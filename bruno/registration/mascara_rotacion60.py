import itk
import numpy as np
import cv2
import matplotlib.pyplot as plt

# --- Parámetros (completar) ---
RUTA_FIXED  = "bruno/volumenes/superficial/registro2_5/producto.nii.gz"
RUTA_MOVING = "bruno/volumenes/superficial/registro1_4/producto_penal.nii.gz"
K = 60                      # índice de frame
ANGULO_HORARIO = -60          # grados
# -----------------------------

fixed  = itk.array_from_image(itk.imread(RUTA_FIXED))   # (z, y, x)
moving = itk.array_from_image(itk.imread(RUTA_MOVING))
print("shapes:", fixed.shape, moving.shape)

f = fixed[K].astype(np.float32)
m = moving[K].astype(np.float32)
h, w = m.shape

# Centro (x, y) en índices de píxel: centro de la imagen recortada
centro = ((w - 1) / 2, (h - 1) / 2)    # (109.5, 287.0) para 220 x 575

# En cv2, ángulo positivo = antihorario en pantalla (origen arriba a la izquierda).
# Horario => ángulo negativo.
R = cv2.getRotationMatrix2D(centro, -ANGULO_HORARIO, 1.0)

# Máscara: dominio del móvil (unos) rotado, vecino más cercano, relleno 0,
# salida con la forma del frame del fixed
unos = np.ones_like(m, dtype=np.uint8)
mascara = cv2.warpAffine(unos, R, (f.shape[1], f.shape[0]),
                         flags=cv2.INTER_NEAREST, borderValue=0)

# Solo visualización del móvil rotado
m_rot = cv2.warpAffine(m, R, (f.shape[1], f.shape[0]),
                       flags=cv2.INTER_LINEAR, borderValue=0)

print("Píxeles en la máscara:", int(mascara.sum()),
      f"({100 * mascara.mean():.1f} % del frame fijo)")

fig, ax = plt.subplots(1, 4, figsize=(16, 7))
ax[0].imshow(f, cmap="gray");      ax[0].set_title("Fixed")
ax[1].imshow(m, cmap="gray");      ax[1].set_title("Moving original")
ax[2].imshow(m_rot, cmap="gray");  ax[2].set_title("Moving rotado")
ax[3].imshow(f, cmap="gray")
ax[3].imshow(np.ma.masked_equal(mascara, 0), cmap="autumn", alpha=0.4)
ax[3].set_title("Fixed + máscara de coincidencia")
for a in ax: a.axis("off")
plt.tight_layout(); plt.show()