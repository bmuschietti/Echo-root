import itk
import numpy as np

# --- Parámetros ---
RUTA_FIXED = "bruno/volumenes/superficial/registro1_4/fixed.nii.gz"
RUTA_MOVING = "bruno/volumenes/superficial/registro1_4/moving_registrado_bspline_penal.nii.gz"
RUTA_MOVING_ORIG = "bruno/volumenes/superficial/registro1_4/moving_original.nii.gz"  # grilla de origen del móvil
RUTA_TRANSFORM = ["bruno/volumenes/superficial/registro1_4/transform_params_rigid_penal.txt", 
                  "bruno/volumenes/superficial/registro1_4/transform_params_bspline_penal.txt"]  # último archivo de la cadena (ajustar)
RUTA_SALIDA = "bruno/volumenes/superficial/registro1_4/producto_penal.nii.gz"
RUTA_MASCARA = "bruno/volumenes/superficial/registro1_4/mascara_validez.nii.gz"
P_BAJO, P_ALTO = 1.0, 99.0  # percentiles de normalización (por definir)


def normalizar(vol, valido, p_bajo, p_alto):
    """Normaliza a [0, 1] con percentiles calculados solo en la región válida."""
    lo, hi = np.percentile(vol[valido], [p_bajo, p_alto])
    out = np.clip((vol - lo) / (hi - lo), 0.0, 1.0)
    out[~valido] = 0.0
    return out.astype(np.float32)


# --- Lectura ---
fixed_itk = itk.imread(RUTA_FIXED, itk.F)
moving_itk = itk.imread(RUTA_MOVING, itk.F)
A = itk.array_from_image(fixed_itk)
B = itk.array_from_image(moving_itk)
assert A.shape == B.shape, f"Dimensiones distintas: {A.shape} vs {B.shape}"

# --- Máscara de validez: imagen de unos transformada con el mismo registro ---
mov_orig_itk = itk.imread(RUTA_MOVING_ORIG, itk.F)
unos = itk.image_from_array(np.ones(itk.array_from_image(mov_orig_itk).shape, np.float32))
unos.CopyInformation(mov_orig_itk)

params = itk.ParameterObject.New()
for ruta in RUTA_TRANSFORM:
    params.AddParameterFile(ruta)
params.SetParameter("FinalBSplineInterpolationOrder", "0")
params.SetParameter("DefaultPixelValue", "0")

unos_reg = itk.transformix_filter(unos, params, log_to_console=False)
valido = itk.array_from_image(unos_reg) > 0.5
assert valido.shape == A.shape, f"Máscara {valido.shape} vs fixed {A.shape}"
print(f"Vóxeles inválidos: {(~valido).sum()} ({100 * (~valido).mean():.2f} %)")

# --- Normalización y producto ---
A_norm = normalizar(A, valido, P_BAJO, P_ALTO)
B_norm = normalizar(B, valido, P_BAJO, P_ALTO)
P = (A_norm * B_norm).astype(np.float32)

# --- Guardado (misma grilla que fixed) ---
P_itk = itk.image_from_array(P)
P_itk.CopyInformation(fixed_itk)
itk.imwrite(P_itk, RUTA_SALIDA)

M_itk = itk.image_from_array(valido.astype(np.uint8))
M_itk.CopyInformation(fixed_itk)
itk.imwrite(M_itk, RUTA_MASCARA)
