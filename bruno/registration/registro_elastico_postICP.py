import itk
import numpy as np
import cv2
import os

# --- Rutas (completar) ---
RUTA_FIXED     = "bruno/volumenes/superficial/registro3_6/producto_limpio.nii.gz"
RUTA_MOVING    = "bruno/volumenes/superficial/registro1_4/producto_penal_limpio.nii.gz"
RUTA_ICP       = "bruno/volumenes/superficial/final/3fixed/transform_icp_1.txt"      # transformación del ICP (AffineTransform)
OUT            = "bruno/volumenes/superficial/final/3fixed"
ANGULO_HORARIO = -120                                # solo para la máscara de coincidencia
USAR_MASCARA   = True

# =====================================================================
# PARÁMETROS QUE REGULAN LA DEFORMACIÓN (los que se van a ir probando)
# =====================================================================
GRILLA_MM      = ["10", "10", "20"]   # FinalGridSpacingInPhysicalUnits: más grueso = deforma menos
SCHEDULE_GRILLA = ["4", "2", "2"]     # GridSpacingSchedule: evitar "1" al final limita el detalle fino
PENALIZACION   = "rigidez"            # "rigidez" (TransformRigidityPenalty) o "bending"
PESO_PENAL     = "0"                # Metric1Weight: subir por órdenes de magnitud (10, 100, 1000)
PESO_PROPIEDAD = "20"                 # PropernessConditionWeight: penaliza expansión/compresión (det. Jacobiano != 1)
PESO_ORTONORM  = "1"                  # OrthonormalityConditionWeight: penaliza corte y escala
PESO_LINEAL    = "100"                # LinearityConditionWeight
ITERACIONES    = "100"                # MaximumNumberOfIterations: menos = regularización implícita
# =====================================================================
os.makedirs(OUT, exist_ok=True)

# --- Carga e igualación de frames (mismo criterio que en el ICP) ---
ref = itk.imread(RUTA_FIXED, itk.F)
fixed_np  = itk.array_from_image(ref)
moving_np = itk.array_from_image(itk.imread(RUTA_MOVING, itk.F))
n = min(fixed_np.shape[0], moving_np.shape[0])
fixed_np, moving_np = fixed_np[-n:], moving_np[-n:]

fixed = itk.image_from_array(fixed_np.astype(np.float32));   fixed.CopyInformation(ref)
moving = itk.image_from_array(moving_np.astype(np.float32)); moving.CopyInformation(ref)
nz, h, w = fixed_np.shape

# --- Máscara de coincidencia (misma banda que en el ICP) ---
mask = None
if USAR_MASCARA:
    M = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), -ANGULO_HORARIO, 1.0)
    Minv = np.linalg.inv(np.vstack([M, [0, 0, 1]]))
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    mx = Minv[0, 0] * xs + Minv[0, 1] * ys + Minv[0, 2]
    my = Minv[1, 0] * xs + Minv[1, 1] * ys + Minv[1, 2]
    m2 = ((mx >= -0.5) & (mx <= w - 0.5) & (my >= -0.5) & (my <= h - 0.5)).astype(np.uint8)
    mask = itk.image_from_array(np.broadcast_to(m2, (nz, h, w)).copy())
    mask.CopyInformation(ref)

# --- Mapa B-spline ---
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
bs["Metric0Weight"] = ["1.0"]
bs["Metric1Weight"] = [PESO_PENAL]
if PENALIZACION == "rigidez":
    bs["Metric"] = ["AdvancedMeanSquares", "TransformRigidityPenalty"]
    bs["PropernessConditionWeight"] = [PESO_PROPIEDAD]
    bs["OrthonormalityConditionWeight"] = [PESO_ORTONORM]
    bs["LinearityConditionWeight"] = [PESO_LINEAL]
else:
    bs["Metric"] = ["AdvancedMeanSquares", "TransformBendingEnergyPenalty"]
pm.AddParameterMap(bs)

# --- Registro: parte de la transformación del ICP ---
kwargs = dict(parameter_object=pm,
              initial_transform_parameter_file_name=RUTA_ICP,
              log_to_console=True)
if mask is not None:
    kwargs["fixed_mask"] = mask
resultado, params = itk.elastix_registration_method(fixed, moving, **kwargs)

# --- Guardar volumen y transformación ---
itk.imwrite(resultado, os.path.join(OUT, "1moving_registrado_elastico.nii.gz"))
itk.imwrite(fixed, os.path.join(OUT, "3fixed.nii.gz"))

po_out = itk.ParameterObject.New()
po_out.AddParameterMap(params.GetParameterMap(0))
po_out.WriteParameterFile(os.path.join(OUT, "1transform_params_bspline.txt"))
print("Guardado en", OUT)