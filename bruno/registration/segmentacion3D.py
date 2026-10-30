import itk
import numpy as np
from scipy.ndimage import gaussian_filter

# --- Parámetros (completar) ---
RUTA = "bruno/volumenes/superficial/registro3_6/producto.nii"
RUTA_LIMPIO = "bruno/volumenes/superficial/registro3_6/producto_limpio.nii.gz"   # producto con paredes en 0 (intensidades)
RUTA_SEG    = "bruno/volumenes/superficial/registro3_6/producto_seg.nii.gz"      # segmentación suavizada (para registrar)
PERCENTIL = 95
FILAS_CERO = [(0, 60), (440, 575)]
SIGMA_PX = 1.5        # suavizado gaussiano en XY (px); 0 = sin suavizar
# -----------------------------

img = itk.imread(RUTA, itk.F)
vol = itk.array_from_image(img).astype(np.float32)   # (z, y, x)

# 1) Paredes en 0 (se mantiene tamaño y centro de la imagen)
for a, b in FILAS_CERO:
    vol[:, a:b, :] = 0

# 2) Umbral: percentil de los vóxeles > 0, ya sin paredes
umbral = np.percentile(vol[vol > 0], PERCENTIL)
seg = (vol > umbral).astype(np.float32)
print(f"Umbral p{PERCENTIL} = {umbral:.4f} | vóxeles segmentados: {int(seg.sum())}")

# 3) Suavizado solo en XY (Z queda sin suavizar)
if SIGMA_PX > 0:
    seg = gaussian_filter(seg, sigma=(0, SIGMA_PX, SIGMA_PX))

def guardar(arr, ruta):
    out = itk.image_from_array(arr.astype(np.float32))
    out.CopyInformation(img)
    itk.imwrite(out, ruta)

guardar(vol, RUTA_LIMPIO)
guardar(seg, RUTA_SEG)