import itk
import numpy as np
from scipy.ndimage import rotate

# --- Parámetros (completar) ---
RUTA_REF = "bruno/volumenes/superficial/final/2fixed/suma_2fixed.nii.gz"      # suma con fixed 2-5 (referencia)
OTRAS = [("bruno/volumenes/superficial/final/3fixed/suma_3fixed.nii.gz", -60)]
OUT_PROMEDIO = "bruno/volumenes/superficial/final/suma_promedio.nii.gz"

# -----------------------------

ref_img = itk.imread(RUTA_REF, itk.F)
ref = itk.array_from_image(ref_img).astype(np.float32)     # (z, y, x)
alineadas = [ref]

for ruta, ang in OTRAS:
    vol = itk.array_from_image(itk.imread(ruta, itk.F)).astype(np.float32)
    assert vol.shape == ref.shape, f"Distinta forma: {ruta}"
    alineadas.append(rotate(vol, ang, axes=(1, 2), reshape=False, order=1,
                            mode="constant", cval=0.0).astype(np.float32))

promedio = np.mean(alineadas, axis=0).astype(np.float32)
out = itk.image_from_array(promedio)
out.CopyInformation(ref_img)
itk.imwrite(out, OUT_PROMEDIO)
print("Promedio guardado en", OUT_PROMEDIO)