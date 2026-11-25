import itk
import numpy as np
import os

# --- Entradas ---
RUTA_FIXED = "bruno/volumenes/superficial/registro6_3/producto_limpio.nii"
MOVINGS = [("bruno/volumenes/superficial/registro2_5/producto_limpio.nii", 2),   # (limpio, nro)
           ("bruno/volumenes/superficial/registro5_2/producto_limpio.nii", 5),
           ("bruno/volumenes/superficial/registro4_1/producto_limpio.nii", 4),
           ("bruno/volumenes/superficial/registro1_4/producto_penal_limpio.nii", 1),
           ("bruno/volumenes/superficial/registro3_6/producto_limpio.nii", 3)]
CARPETA = "bruno/volumenes/superficial/6fixed"   # donde están los moving{nro}_transform_bspline.txt
SALIDA = "suma_6fixed.nii.gz"
MARGEN_MM = 5
# -----------------------------

ref = itk.imread(RUTA_FIXED, itk.F)
fixed_full = itk.array_from_image(ref)
nz0, h, w = fixed_full.shape
sp, org = np.array(itk.spacing(ref)), np.array(itk.origin(ref))
D = itk.array_from_matrix(ref.GetDirection())

# Grilla grande que contiene cualquier giro
R = np.hypot(w * sp[0] / 2, h * sp[1] / 2) + MARGEN_MM
N = int(np.ceil(2 * R / sp[0]))
px, py = int(np.ceil((N - w) / 2)), int(np.ceil((N - h) / 2))
nx, ny = w + 2 * px, h + 2 * py
org_big = org - D @ (sp * np.array([px, py, 0]))

# Moving remuestreado una sola vez con la cadena completa (ICP + B-spline)
resampleados = []
for ruta, nro in MOVINGS:
    mov_np = itk.array_from_image(itk.imread(ruta, itk.F))
    n = min(nz0, mov_np.shape[0])                       # mismo criterio de frames del registro
    mov = itk.image_from_array(mov_np[-n:].astype(np.float32)); mov.CopyInformation(ref)

    po = itk.ParameterObject.New()
    po.AddParameterFile(os.path.join(CARPETA, f"moving{nro}_transform_bspline.txt"))
    po.SetParameter("Size", [str(nx), str(ny), str(n)])
    po.SetParameter("Origin", [repr(float(v)) for v in org_big])
    po.SetParameter("Spacing", [repr(float(v)) for v in sp])
    po.SetParameter("DefaultPixelValue", "0")
    po.SetParameter("FinalBSplineInterpolationOrder", "1")
    resampleados.append(itk.array_from_image(itk.transformix_filter(mov, po)).astype(np.float32))
    print(f"Moving {nro} remuestreado")

# Frames comunes (últimos nz, como en el resto del pipeline)
nz = min([nz0] + [r.shape[0] for r in resampleados])
fixed_big = np.zeros((nz, ny, nx), np.float32)
fixed_big[:, py:py + h, px:px + w] = fixed_full[-nz:]

# Suma directa (sin normalizar)
suma = fixed_big + sum(r[-nz:] for r in resampleados)

img = itk.image_from_array(suma.astype(np.float32))
img.SetSpacing(tuple(float(v) for v in sp))
img.SetOrigin(tuple(float(v) for v in org_big))
img.SetDirection(ref.GetDirection())
itk.imwrite(img, os.path.join(CARPETA, SALIDA))
print("Suma guardada")