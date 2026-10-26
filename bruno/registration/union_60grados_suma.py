import itk
import numpy as np
import os

# --- Parámetros (completar) ---
RUTA_FIXED  = "bruno/volumenes/superficial/registro2_5/producto.nii"
RUTA_MOVING = "bruno/volumenes/superficial/registro1_4/producto_penal.nii"
RUTA_MOVING2 = "bruno/volumenes/superficial/registro3_6/producto.nii"
RUTA_PARAMS = "bruno/volumenes/superficial/final/transform_params_rigid_1_2.txt" 
RUTA_PARAMS2 = "bruno/volumenes/superficial/final/transform_params_rigid_3_2.txt"  # el que apunta a transform_inicial.txt
OUT = "bruno/volumenes/superficial/final"
MARGEN_MM = 5      # margen extra alrededor del círculo que contiene cualquier giro
# -----------------------------
os.makedirs(OUT, exist_ok=True)

# --- Cargar y igualar frames (mismo criterio que en el registro) ---
ref = itk.imread(RUTA_FIXED, itk.F)
fixed_np  = itk.array_from_image(ref)                            # (z, y, x)
moving1_np = itk.array_from_image(itk.imread(RUTA_MOVING, itk.F))
n = min(fixed_np.shape[0], moving1_np.shape[0])
fixed_np, moving1_np = fixed_np[-n:], moving1_np[-n:]

moving1 = itk.image_from_array(moving1_np)
moving1.CopyInformation(ref)

moving2_np = itk.array_from_image(itk.imread(RUTA_MOVING2, itk.F))
n = min(fixed_np.shape[0], moving2_np.shape[0])
fixed_np, moving2_np = fixed_np[-n:], moving2_np[-n:]

moving2 = itk.image_from_array(moving2_np)
moving2.CopyInformation(ref)


# --- Geometría del fixed ---
nz, h, w = fixed_np.shape
sp  = np.array(itk.spacing(ref))
org = np.array(itk.origin(ref))
D   = itk.array_from_matrix(ref.GetDirection())

# --- Grilla grande: contiene cualquier giro del rectángulo alrededor de su centro ---
R = np.hypot(w * sp[0] / 2, h * sp[1] / 2) + MARGEN_MM        # mm
N = int(np.ceil(2 * R / sp[0]))                                # px (sp x = sp y)
px = int(np.ceil((N - w) / 2))
py = int(np.ceil((N - h) / 2))
nx, ny = w + 2 * px, h + 2 * py
org_big = org - D @ (sp * np.array([px, py, 0]))
print(f"Grilla grande: {nx} x {ny} x {nz} px")

# --- Fixed: queda alineado a la grilla, solo se coloca con un desplazamiento entero ---
fixed_big = np.zeros((nz, ny, nx), np.float32)
fixed_big[:, py:py + h, px:px + w] = fixed_np

# --- Moving: un solo remuestreo desde el original con la transformación total ---
po = itk.ParameterObject.New()
po.AddParameterFile(RUTA_PARAMS)
po.SetParameter("Size",    [str(nx), str(ny), str(nz)])
po.SetParameter("Origin",  [repr(float(v)) for v in org_big])
po.SetParameter("Spacing", [repr(float(v)) for v in sp])
po.SetParameter("DefaultPixelValue", "0")
po.SetParameter("FinalBSplineInterpolationOrder", "1")
moving1_big = itk.array_from_image(
    itk.transformix_filter(moving1, po)
).astype(np.float32)

# --- Moving: un solo remuestreo desde el original con la transformación total ---
po = itk.ParameterObject.New()
po.AddParameterFile(RUTA_PARAMS2)
po.SetParameter("Size",    [str(nx), str(ny), str(nz)])
po.SetParameter("Origin",  [repr(float(v)) for v in org_big])
po.SetParameter("Spacing", [repr(float(v)) for v in sp])
po.SetParameter("DefaultPixelValue", "0")
po.SetParameter("FinalBSplineInterpolationOrder", "1")
moving2_big = itk.array_from_image(
    itk.transformix_filter(moving2, po)
).astype(np.float32)


# --- Suma directa ---
suma = fixed_big + moving1_big + moving2_big

def guardar(arr, nombre):
    img = itk.image_from_array(arr)
    img.SetSpacing(tuple(float(v) for v in sp))
    img.SetOrigin(tuple(float(v) for v in org_big))
    img.SetDirection(ref.GetDirection())
    itk.imwrite(img, os.path.join(OUT, nombre))

#guardar(fixed_big,  "fixed_grande.nii.gz")
#guardar(moving_big, "moving_grande.nii.gz")
guardar(suma,       "suma_final.nii.gz")