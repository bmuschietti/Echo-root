import numpy as np
import pydicom
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURACIÓN
# ============================================================

dicom_path = "data/ultrasound/primera_medicion/001.dcm"

# Frame que queremos analizar
FRAME = 118

# Región útil del ultrasonido
X_MIN = 365
X_MAX = 650

Y_MIN = 115
Y_MAX = 690

# Período de reverberación encontrado
LAG = 224

# Valores a comparar
ATTENUATIONS = [0.10, 0.20, 0.30, 0.40]


# ============================================================
# DECONVOLUCIÓN DE REVERBERACIÓN
# ============================================================

def reverberation_deconvolution(
    bscan,
    lag=224,
    attenuation=0.25
):
    """
    Deconvolución predictiva para una reverberación periódica.

    Modelo aproximado:

        y[n] = x[n] + a*x[n-lag]

    donde:
        y = señal observada
        x = señal estimada sin reverberación
        a = atenuación de la reverberación
        lag = separación entre ecos
    """

    bscan = bscan.astype(np.float32)

    height, width = bscan.shape

    result = np.zeros_like(bscan)

    for x in range(width):

        signal = bscan[:, x]

        # No podemos corregir los primeros "lag" samples
        result[:lag, x] = signal[:lag]

        # Deconvolución recursiva
        for n in range(lag, height):

            result[n, x] = (
                signal[n]
                - attenuation * result[n - lag, x]
            )

    return result


# ============================================================
# CARGAR DICOM
# ============================================================

ds = pydicom.dcmread(dicom_path)

volume = ds.pixel_array

print("Shape original:", volume.shape)

# El DICOM tiene 3 canales idénticos.
# Nos quedamos con uno.
if volume.ndim == 4 and volume.shape[-1] == 3:
    volume = volume[..., 0]

volume = volume.astype(np.float32)

print("Shape utilizada:", volume.shape)


# ============================================================
# SELECCIONAR FRAME
# ============================================================

bscan = volume[FRAME]

print("B-scan:", bscan.shape)


# ============================================================
# RECORTAR REGIÓN DE ULTRASONIDO
# ============================================================

bscan_crop = bscan[
    Y_MIN:Y_MAX,
    X_MIN:X_MAX
]

print("B-scan recortado:", bscan_crop.shape)


# ============================================================
# APLICAR FILTRO
# ============================================================

filtered_images = []

for attenuation in ATTENUATIONS:

    filtered = reverberation_deconvolution(
        bscan_crop,
        lag=LAG,
        attenuation=attenuation
    )

    filtered_images.append(filtered)


# ============================================================
# VISUALIZACIÓN
# ============================================================

fig, ax = plt.subplots(
    1,
    len(ATTENUATIONS) + 1,
    figsize=(20, 6)
)


# -------------------------
# ORIGINAL
# -------------------------

ax[0].imshow(
    bscan_crop,
    cmap="gray",
    vmin=0,
    vmax=255
)

ax[0].set_title("Original")
ax[0].axis("off")


# -------------------------
# FILTRADOS
# -------------------------

for i, (attenuation, filtered) in enumerate(
    zip(ATTENUATIONS, filtered_images)
):

    ax[i + 1].imshow(
        filtered,
        cmap="gray",
        vmin=0,
        vmax=255
    )

    ax[i + 1].set_title(
        f"Lag = {LAG}\na = {attenuation}"
    )

    ax[i + 1].axis("off")


plt.tight_layout()
plt.show()