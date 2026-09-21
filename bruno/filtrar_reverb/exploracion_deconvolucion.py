import pydicom
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')



# ============================================================
# CONFIGURACIÓN
# ============================================================

dicom_path = "data/ultrasound/primera_medicion/001.dcm"

# Eje axial/profundidad.
# Para un volumen (236, 768, 1024), inicialmente asumimos eje 1.
AXIAL_AXIS = 1

# Orden del filtro predictivo
PREDICTION_ORDER = 20

# Lag mínimo que consideramos como reverberación
# Se expresa en muestras.
MIN_LAG = 5

# Regularización para evitar inestabilidad
REGULARIZATION = 1e-3


# ============================================================
# DECONVOLUCIÓN PREDICTIVA
# ============================================================

def predictive_deconvolution(trace, order=20,
                             min_lag=5,
                             regularization=1e-3):

    trace = trace.astype(np.float64)

    # quitar componente DC
    mean = np.mean(trace)
    x = trace - mean

    # energía
    energy = np.sum(x ** 2)

    if energy < 1e-12:
        return trace.copy()

    # autocorrelación
    r = np.correlate(x, x, mode="full")

    center = len(x) - 1

    # autocorrelación positiva
    r = r[center:center + order + min_lag + 1]

    # normalización
    r = r / (r[0] + 1e-12)

    # matriz Toeplitz para las ecuaciones de Yule-Walker
    p = order

    R = np.empty((p, p))

    for i in range(p):
        for j in range(p):
            R[i, j] = r[abs(i - j)]

    R += regularization * np.eye(p)

    # vector de autocorrelación
    rhs = r[1:p + 1]

    try:
        a = np.linalg.solve(R, rhs)
    except np.linalg.LinAlgError:
        return trace.copy()

    # filtro predictivo
    predicted = np.zeros_like(x)

    for i in range(p, len(x)):
        predicted[i] = np.dot(a, x[i-p:i][::-1])

    # error de predicción
    residual = x - predicted

    # volver a agregar media
    residual += mean

    return residual


# ============================================================
# PROCESAR VOLUMEN
# ============================================================

def process_volume(volume,
                   axial_axis=1,
                   order=20,
                   min_lag=5,
                   regularization=1e-3):

    volume = volume.astype(np.float64)

    # movemos profundidad al primer eje
    v = np.moveaxis(volume, axial_axis, 0)

    filtered = np.zeros_like(v)

    depth, height, width = v.shape

    print("Procesando volumen:")
    print("Depth:", depth)
    print("Height:", height)
    print("Width:", width)

    for z in range(height):

        print(f"\rCorte {z+1}/{height}", end="")

        for x in range(width):

            trace = v[:, z, x]

            filtered[:, z, x] = predictive_deconvolution(
                trace,
                order=order,
                min_lag=min_lag,
                regularization=regularization
            )

    print()

    # devolver a orientación original
    filtered = np.moveaxis(filtered, 0, axial_axis)

    return filtered


# ============================================================
# MAIN
# ============================================================

ds = pydicom.dcmread(dicom_path)

volume = ds.pixel_array

print("Volumen DICOM:", volume.shape)

# Los 3 canales son idénticos → usamos solamente uno
if volume.ndim == 4 and volume.shape[-1] == 3:
    volume = volume[..., 0]

print("Volumen utilizado:", volume.shape)
print("dtype:", volume.dtype)
print("min:", volume.min())
print("max:", volume.max())

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 3, figsize=(18, 6))

# 1. Un frame/corte
ax[0].imshow(volume[118], cmap="gray")
ax[0].set_title("volume[118]")
ax[0].axis("off")

# 2. Corte atravesando el eje 236
ax[1].imshow(volume[:, 384, :], cmap="gray")
ax[1].set_title("volume[:, 384, :]")
ax[1].axis("off")

# 3. Corte atravesando el otro eje
ax[2].imshow(volume[:, :, 512], cmap="gray")
ax[2].set_title("volume[:, :, 512]")
ax[2].axis("off")

plt.tight_layout()
plt.show()



# filtered_volume = process_volume(
#     volume,
#     axial_axis=AXIAL_AXIS,
#     order=PREDICTION_ORDER,
#     min_lag=MIN_LAG,
#     regularization=REGULARIZATION
# )


# # ============================================================
# # COMPARACIÓN
# # ============================================================

# slice_id = 100

# fig, ax = plt.subplots(1, 2, figsize=(16, 7))

# ax[0].imshow(
#     volume[slice_id],
#     cmap="gray"
# )
# ax[0].set_title("Original")
# ax[0].axis("off")

# ax[1].imshow(
#     filtered_volume[slice_id],
#     cmap="gray"
# )
# ax[1].set_title("Deconvolución predictiva")
# ax[1].axis("off")

# plt.tight_layout()
# plt.show()

