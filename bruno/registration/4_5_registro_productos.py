import itk
import numpy as np
import cv2
import os
import open3d as o3d

# --- Entradas --- 
FIXED   = ("bruno/volumenes/superficial/registro1_4/producto_penal_seg.nii", "bruno/volumenes/superficial/registro1_4/producto_penal_limpio.nii")
# (ruta al producto segmentado ; ruta al producto limpio; angulo de rotación en sentido horario del fixed hasta el moving, nro de producto)
MOVINGS = [("bruno/volumenes/superficial/registro5_2/producto_seg.nii", "bruno/volumenes/superficial/registro5_2/producto_limpio.nii", -120, 5),
           ("bruno/volumenes/superficial/registro2_5/producto_seg.nii", "bruno/volumenes/superficial/registro2_5/producto_limpio.nii", 60, 2), 
           ("bruno/volumenes/superficial/registro3_6/producto_seg.nii", "bruno/volumenes/superficial/registro3_6/producto_limpio.nii", 120,3),   
           ("bruno/volumenes/superficial/registro4_1/producto_seg.nii", "bruno/volumenes/superficial/registro4_1/producto_limpio.nii", 180, 4),
           ("bruno/volumenes/superficial/registro6_3/producto_seg.nii", "bruno/volumenes/superficial/registro6_3/producto_limpio.nii", -60, 6),
           ]
OUT = "bruno/volumenes/superficial/1fixed"

# --- ICP ---
UMBRAL_SEG, MAX_DIST_MM, N_MAX = 0.5, 30, 200000

# --- B-spline ---
GRILLA_MM, SCHEDULE_GRILLA = ["10", "10", "20"], ["4", "2", "2"]
PESO_PENAL, PESO_PROPIEDAD, PESO_ORTONORM, PESO_LINEAL = "0", "20", "1", "100"
ITERACIONES = "40"
# -----------------------------
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)

def cargar_par(ruta_f, ruta_m):
    # Carga e iguala frames con [-n:]
    ref = itk.imread(ruta_f, itk.F)
    f = itk.array_from_image(ref)
    m = itk.array_from_image(itk.imread(ruta_m, itk.F))
    n = min(f.shape[0], m.shape[0])
    return ref, f[-n:], m[-n:]

def banda_coincidencia(M, h, w):
    # Región del fixed cubierta por el moving rotado
    Minv = np.linalg.inv(np.vstack([M, [0, 0, 1]]))
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    mx = Minv[0, 0] * xs + Minv[0, 1] * ys + Minv[0, 2]
    my = Minv[1, 0] * xs + Minv[1, 1] * ys + Minv[1, 2]
    return (mx >= -0.5) & (mx <= w - 0.5) & (my >= -0.5) & (my <= h - 0.5)

def icp(ref, fixed_np, moving_np, ang, ruta_txt):
    nz, h, w = fixed_np.shape
    sp, org = np.array(itk.spacing(ref)), np.array(itk.origin(ref))
    D = itk.array_from_matrix(ref.GetDirection())

    M = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), -ang, 1.0)
    banda = banda_coincidencia(M, h, w)

    def a_mm(x, y, z):
        idx = np.stack([x, y, z], axis=1).astype(np.float64)
        return org + (D @ (idx * sp).T).T

    def submuestrear(p):
        return p[rng.choice(len(p), N_MAX, replace=False)] if len(p) > N_MAX else p

    # Nube del fixed (dentro de la banda)
    zf, yf, xf = np.nonzero(fixed_np > UMBRAL_SEG)
    d = banda[yf, xf]
    P_fixed = submuestrear(a_mm(xf[d], yf[d], zf[d]))

    # Nube del moving con la rotación inicial aplicada
    zm, ym, xm = np.nonzero(moving_np > UMBRAL_SEG)
    xr = M[0, 0] * xm + M[0, 1] * ym + M[0, 2]
    yr = M[1, 0] * xm + M[1, 1] * ym + M[1, 2]
    d = (xr >= -0.5) & (xr <= w - 0.5) & (yr >= -0.5) & (yr <= h - 0.5)
    P_moving = submuestrear(a_mm(xr[d], yr[d], zm[d]))

    # ICP rígido
    src = o3d.geometry.PointCloud(); src.points = o3d.utility.Vector3dVector(P_moving)
    tgt = o3d.geometry.PointCloud(); tgt.points = o3d.utility.Vector3dVector(P_fixed)
    res = o3d.pipelines.registration.registration_icp(
        src, tgt, MAX_DIST_MM, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=500))
    T = res.transformation
    print(f"  ICP fitness {res.fitness:.3f} | rmse {res.inlier_rmse:.3f} mm")

    # Transformación total (rotación + ICP) como AffineTransform de elastix
    G = np.eye(4); G[:3, :3] = D @ np.diag(sp); G[:3, 3] = org
    Mi = np.eye(4); Mi[:2, :2] = M[:, :2]; Mi[:2, 3] = M[:, 2]
    F = np.linalg.inv(G @ Mi @ np.linalg.inv(G)) @ np.linalg.inv(T)
    params = " ".join(repr(float(v)) for v in list(F[:3, :3].flatten()) + list(F[:3, 3]))
    dir_str = " ".join(repr(float(v)) for v in D.flatten())
    with open(ruta_txt, "w") as f:
        f.write(f'''(Transform "AffineTransform")
(NumberOfParameters 12)
(TransformParameters {params})
(InitialTransformParameterFileName "NoInitialTransform")
(HowToCombineTransforms "Compose")
(FixedImageDimension 3)
(MovingImageDimension 3)
(FixedInternalImagePixelType "float")
(MovingInternalImagePixelType "float")
(Size {w} {h} {nz})
(Index 0 0 0)
(Spacing {sp[0]} {sp[1]} {sp[2]})
(Origin {org[0]} {org[1]} {org[2]})
(Direction {dir_str})
(UseDirectionCosines "true")
(CenterOfRotationPoint 0 0 0)
(ResampleInterpolator "FinalBSplineInterpolator")
(FinalBSplineInterpolationOrder 1)
(Resampler "DefaultResampler")
(DefaultPixelValue 0)
(ResultImageFormat "nii")
(ResultImagePixelType "float")
''')

def bspline(ref, fixed_np, moving_np, ang, ruta_icp, ruta_vol, ruta_txt):
    nz, h, w = fixed_np.shape
    fixed = itk.image_from_array(fixed_np.astype(np.float32));   fixed.CopyInformation(ref)
    moving = itk.image_from_array(moving_np.astype(np.float32)); moving.CopyInformation(ref)

    # Máscara de coincidencia (misma banda que el ICP)
    M = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), -ang, 1.0)
    m2 = banda_coincidencia(M, h, w).astype(np.uint8)
    mask = itk.image_from_array(np.broadcast_to(m2, (nz, h, w)).copy())
    mask.CopyInformation(ref)

    pm = itk.ParameterObject.New()
    bs = pm.GetDefaultParameterMap("bspline", 3)
    bs["AutomaticTransformInitialization"] = ["false"]
    bs["NumberOfResolutions"] = ["3"]
    bs["ImagePyramidSchedule"] = ["8", "8", "1", "4", "4", "1", "2", "2", "1"]
    bs["FinalGridSpacingInPhysicalUnits"] = GRILLA_MM
    bs["GridSpacingSchedule"] = SCHEDULE_GRILLA
    bs["MaximumNumberOfIterations"] = [ITERACIONES]
    bs["DefaultPixelValue"] = ["0"]
    bs["ErodeMask"] = ["false"]
    bs["ImageSampler"] = ["Full"]
    bs["Registration"] = ["MultiMetricMultiResolutionRegistration"]
    bs["Metric"] = ["AdvancedMeanSquares", "TransformRigidityPenalty"]
    bs["Metric0Weight"] = ["1.0"]
    bs["Metric1Weight"] = [PESO_PENAL]
    bs["PropernessConditionWeight"] = [PESO_PROPIEDAD]
    bs["OrthonormalityConditionWeight"] = [PESO_ORTONORM]
    bs["LinearityConditionWeight"] = [PESO_LINEAL]
    pm.AddParameterMap(bs)

    resultado, params = itk.elastix_registration_method(
        fixed, moving, parameter_object=pm, fixed_mask=mask,
        initial_transform_parameter_file_name=ruta_icp, log_to_console=True)

    itk.imwrite(resultado, ruta_vol)
    po = itk.ParameterObject.New()
    po.AddParameterMap(params.GetParameterMap(0))
    po.WriteParameterFile(ruta_txt)

for i, (seg_m, limpio_m, ang, nro) in enumerate(MOVINGS, 1):
    print(f"Moving {nro} ({ang}°)")
    ruta_icp = os.path.join(OUT, f"moving{nro}_transform_icp.txt")

    # 1) ICP sobre los _seg
    ref, f_seg, m_seg = cargar_par(FIXED[0], seg_m)
    icp(ref, f_seg, m_seg, ang, ruta_icp)

    # 2) B-spline sobre los _limpio, partiendo del ICP
    ref, f_lim, m_lim = cargar_par(FIXED[1], limpio_m)
    bspline(ref, f_lim, m_lim, ang, ruta_icp,
            os.path.join(OUT, f"moving{nro}_registrado.nii.gz"),
            os.path.join(OUT, f"moving{nro}_transform_bspline.txt"))
print("Listo")