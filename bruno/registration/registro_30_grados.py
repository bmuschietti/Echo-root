import itk
import numpy as np
import cv2
import os

# --- Parámetros (completar) ---
RUTA_FIXED  = "bruno/volumenes/superficial/registro2_5/producto.nii"
RUTA_MOVING = "bruno/volumenes/superficial/registro1_4/producto_penal.nii"
OUT = "bruno/volumenes/superficial/final"
ANGULO_HORARIO = -60      # giro del moving en pantalla 
# -----------------------------
os.makedirs(OUT, exist_ok=True)

fixed_np  = itk.array_from_image(itk.imread(RUTA_FIXED,  itk.F))   # (z, y, x)
moving_np = itk.array_from_image(itk.imread(RUTA_MOVING, itk.F))

n = min(fixed_np.shape[0], moving_np.shape[0])
fixed_np  = fixed_np[-n:]      # descarta frames del inicio del más largo
moving_np = moving_np[-n:]

ref = itk.imread(RUTA_FIXED, itk.F)
fixed  = itk.image_from_array(fixed_np);  fixed.CopyInformation(ref)
moving = itk.image_from_array(moving_np); moving.CopyInformation(ref)


# Ambos volúmenes deben compartir grilla (tamaño, spacing, origen)
assert itk.size(fixed) == itk.size(moving), "Distinto tamaño: igualar frames antes"
assert np.allclose(itk.spacing(fixed), itk.spacing(moving))
assert np.allclose(itk.origin(fixed),  itk.origin(moving))

size = np.array(itk.size(fixed), dtype=int)       # (x, y, z)
sp   = np.array(itk.spacing(fixed))
org  = np.array(itk.origin(fixed))
D    = itk.array_from_matrix(fixed.GetDirection())
w, h, nz = size

# --- 1) Rotación en índices: fixed -> moving ---
# cv2 usa la matriz moving -> fixed (la que dio el resultado correcto en 2D).
# Elastix necesita fixed -> moving, es decir, la inversa.
M = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), -ANGULO_HORARIO, 1.0)
M3 = np.vstack([M, [0, 0, 1]])
Minv = np.linalg.inv(M3)
L = np.eye(3)                      # parte lineal en índices (x, y, z); z sin cambio
L[:2, :2] = Minv[:2, :2]
t_idx = np.array([Minv[0, 2], Minv[1, 2], 0.0])

# --- 2) Misma transformación en coordenadas físicas: p_moving = A p_fixed + t ---
S = np.diag(sp)
A = D @ S @ L @ np.linalg.inv(S) @ np.linalg.inv(D)
t = org + D @ S @ t_idx - A @ org

# --- 3) Máscara: píxeles del fixed que caen dentro del dominio del moving ---
xs, ys = np.meshgrid(np.arange(w), np.arange(h))
mx = Minv[0, 0] * xs + Minv[0, 1] * ys + Minv[0, 2]
my = Minv[1, 0] * xs + Minv[1, 1] * ys + Minv[1, 2]
mask2d = ((mx >= -0.5) & (mx <= w - 0.5) & (my >= -0.5) & (my <= h - 0.5)).astype(np.uint8)
mask3d = np.broadcast_to(mask2d, (nz, h, w)).copy()
mask = itk.image_from_array(mask3d)
mask.CopyInformation(fixed)
itk.imwrite(mask, os.path.join(OUT, "mascara_fixed.nii.gz"))
print("Píxeles en la máscara (por frame):", int(mask2d.sum()),
      f"({100 * mask2d.mean():.1f} %)")

# --- 4) Archivo de transformación inicial (afín = rotación pura, centro en 0) ---
params_init = " ".join(repr(float(v)) for v in list(A.flatten()) + list(t))
ruta_init = os.path.join(OUT, "transform_inicial_1_2.txt")
with open(ruta_init, "w") as f:
    f.write(f'''(Transform "AffineTransform")
(NumberOfParameters 12)
(TransformParameters {params_init})
(InitialTransformParametersFileName "NoInitialTransform")
(HowToCombineTransforms "Compose")
(FixedImageDimension 3)
(MovingImageDimension 3)
(FixedInternalImagePixelType "float")
(MovingInternalImagePixelType "float")
(Size {size[0]} {size[1]} {size[2]})
(Index 0 0 0)
(Spacing {sp[0]} {sp[1]} {sp[2]})
(Origin {org[0]} {org[1]} {org[2]})
(Direction {" ".join(repr(float(v)) for v in D.flatten())})
(UseDirectionCosines "true")
(CenterOfRotationPoint 0 0 0)
(ResampleInterpolator "FinalBSplineInterpolator")
(FinalBSplineInterpolationOrder 1)
(Resampler "DefaultResampler")
(DefaultPixelValue 0)
(ResultImageFormat "nii")
(ResultImagePixelType "float")
''')

# --- 5) Registro rígido sobre la zona compartida ---
pm = itk.ParameterObject.New()
rigid = pm.GetDefaultParameterMap("rigid", 3)
rigid["Registration"] = ["MultiResolutionRegistration"]
rigid["Metric"] = ["AdvancedMeanSquares"]
rigid["AutomaticTransformInitialization"] = ["false"]
rigid["AutomaticScalesEstimation"] = ["true"]
rigid["NumberOfResolutions"] = ["3"]
rigid["ImagePyramidSchedule"] = ["8", "8", "1", "4", "4", "1", "2", "2", "1"]
rigid["MaximumNumberOfIterations"] = ["500"]
rigid["NumberOfSpatialSamples"] = ["200000"]
rigid["ErodeMask"] = ["false"]
rigid["DefaultPixelValue"] = ["0"]
pm.AddParameterMap(rigid)

resultado, params = itk.elastix_registration_method(
    fixed, moving,
    parameter_object=pm,
    fixed_mask=mask,
    initial_transform_parameter_file_name=ruta_init,
    log_to_console=True)

itk.imwrite(resultado, os.path.join(OUT, "1_moving_registrado.nii.gz"))
itk.imwrite(fixed, os.path.join(OUT, "2_fixed.nii.gz"))

po_out = itk.ParameterObject.New()
po_out.AddParameterMap(params.GetParameterMap(0))
po_out.WriteParameterFile(os.path.join(OUT, "transform_params_rigid_1_2.txt"))
print("TransformParameters (rx, ry, rz rad; tx, ty, tz mm):",
      params.GetParameterMap(0)["TransformParameters"])