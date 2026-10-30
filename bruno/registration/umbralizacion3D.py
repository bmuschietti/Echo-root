import itk
import numpy as np
import matplotlib.pyplot as plt

# --- Parámetros (completar) ---
RUTA = "bruno/volumenes/superficial/registro3_6/producto.nii"
K = 80                                   # frame a visualizar
PERCENTILES = [90, 95, 98, 99, 99.5]      # umbrales candidatos (percentil de vóxeles > 0)
FILAS_CERO = [(0,60),(440,575)]                           # rangos de filas de las paredes, ej. [(0, 40), (535, 575)]
# -----------------------------

vol = itk.array_from_image(itk.imread(RUTA, itk.F)).astype(np.float32)   # (z, y, x)

# Paredes en 0 (solo para este análisis, no modifica el archivo)
for a, b in FILAS_CERO:
    vol[:, a:b, :] = 0

vals = vol[vol > 0]
umbrales = np.percentile(vals, PERCENTILES)
print(f"Vóxeles > 0: {vals.size} ({100 * vals.size / vol.size:.1f} % del volumen)")
for p, u in zip(PERCENTILES, umbrales):
    n = int((vol > u).sum())
    print(f"p{p}: umbral = {u:.4f} | vóxeles sobre el umbral = {n}")

# --- 1) Histograma (escala log) con los umbrales ---
plt.figure(figsize=(8, 4))
plt.hist(vals, bins=200, log=True, color="gray")
for p, u in zip(PERCENTILES, umbrales):
    plt.axvline(u, color="r", lw=0.8)
    plt.text(u, plt.ylim()[1], f"p{p}", rotation=90, va="top", ha="right", fontsize=8)
plt.xlabel("Intensidad"); plt.ylabel("Vóxeles (log)")
plt.title("Histograma de vóxeles > 0")
plt.tight_layout()

# --- 2) Frame K: imagen y segmentación con cada umbral ---
fig, ax = plt.subplots(1, len(umbrales) + 1, figsize=(3.2 * (len(umbrales) + 1), 7))
ax[0].imshow(vol[K], cmap="gray"); ax[0].set_title(f"Frame {K}")
for a, p, u in zip(ax[1:], PERCENTILES, umbrales):
    a.imshow(vol[K] > u, cmap="gray"); a.set_title(f"p{p} ({u:.3f})")
for a in ax: a.axis("off")
plt.tight_layout()

# --- 3) Vóxeles segmentados por frame, para cada umbral ---
plt.figure(figsize=(9, 4))
for p, u in zip(PERCENTILES, umbrales):
    plt.plot((vol > u).sum(axis=(1, 2)), label=f"p{p}")
plt.xlabel("Frame (z)"); plt.ylabel("Vóxeles sobre el umbral")
plt.legend(); plt.title("Frames con señal según el umbral")
plt.tight_layout()

plt.show()