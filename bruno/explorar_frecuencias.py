import numpy as np
import pydicom
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


# ============================================================
# CONFIGURACIÓN
# ============================================================

dicom_path = "data/ultrasound/primera_medicion/001.dcm"

FRAME = 118

# Recorte ya validado
X_MIN = 365
X_MAX = 650

Y_MIN = 115
Y_MAX = 690

# Tamaño de las ventanas de análisis
WINDOW_Y = 60
WINDOW_X = 40

# Lags que vamos a buscar
MIN_LAG = 5
MAX_LAG = 50


# ============================================================
# CARGAR DICOM
# ============================================================

ds = pydicom.dcmread(dicom_path)

volume = ds.pixel_array

print("Shape original:", volume.shape)

# Los 3 canales son idénticos
if volume.ndim == 4:
    volume = volume[..., 0]

volume = volume.astype(np.float32)

print("Shape utilizada:", volume.shape)


# ============================================================
# SELECCIONAR FRAME
# ============================================================

bscan = volume[FRAME]


# ============================================================
# RECORTE
# ============================================================

bscan_crop = bscan[
    Y_MIN:Y_MAX,
    X_MIN:X_MAX
]

print("B-scan recortado:", bscan_crop.shape)


# ============================================================
# FUNCIÓN: SCORE DE PERIODICIDAD
# ============================================================

def periodicity_score(region, min_lag=5, max_lag=50):

    # Promedio horizontal.
    # Las líneas horizontales deberían mantenerse
    # al promediar en X.
    profile = np.mean(region, axis=1)

    # Quitar componente DC
    profile = profile - np.mean(profile)

    # Evitar problemas si la región es casi uniforme
    energy = np.sum(profile ** 2)

    if energy < 1e-6:
        return 0.0, None

    # Autocorrelación
    ac = np.correlate(
        profile,
        profile,
        mode="full"
    )

    ac = ac[len(profile)-1:]

    # Normalización
    ac /= ac[0]

    max_lag = min(max_lag, len(ac)-1)

    # Buscar picos
    peaks, properties = find_peaks(
        ac[min_lag:max_lag],
        prominence=0.005
    )

    peaks += min_lag

    if len(peaks) == 0:
        return 0.0, None

    # Elegir el pico positivo más fuerte
    best_idx = np.argmax(ac[peaks])

    best_lag = peaks[best_idx]
    best_score = ac[best_lag]

    return best_score, best_lag


# ============================================================
# RECORRER LA IMAGEN
# ============================================================

height, width = bscan_crop.shape

scores = np.zeros(
    (
        height // WINDOW_Y,
        width // WINDOW_X
    ),
    dtype=np.float32
)

lags = np.zeros_like(scores)


for iy, y in enumerate(
    range(0, height - WINDOW_Y + 1, WINDOW_Y)
):

    for ix, x in enumerate(
        range(0, width - WINDOW_X + 1, WINDOW_X)
    ):

        region = bscan_crop[
            y:y + WINDOW_Y,
            x:x + WINDOW_X
        ]

        score, lag = periodicity_score(
            region,
            MIN_LAG,
            MAX_LAG
        )

        scores[iy, ix] = score

        if lag is not None:
            lags[iy, ix] = lag


# ============================================================
# MOSTRAR ORIGINAL Y MAPA DE PERIODICIDAD
# ============================================================

fig, ax = plt.subplots(
    1,
    2,
    figsize=(14, 7)
)


# Original
ax[0].imshow(
    bscan_crop,
    cmap="gray",
    vmin=0,
    vmax=255
)

ax[0].set_title("B-scan recortado")

ax[0].set_xlabel("X")
ax[0].set_ylabel("Profundidad")


# Score de periodicidad
im = ax[1].imshow(
    scores,
    cmap="viridis",
    origin="upper",
    interpolation="nearest"
)

ax[1].set_title(
    "Score de periodicidad local"
)

ax[1].set_xlabel(
    "Ventana X"
)

ax[1].set_ylabel(
    "Ventana Y"
)

plt.colorbar(
    im,
    ax=ax[1],
    label="Autocorrelación máxima"
)

plt.tight_layout()
plt.show()


# ============================================================
# MOSTRAR LAG DETECTADO
# ============================================================

plt.figure(figsize=(8, 6))

plt.imshow(
    lags,
    cmap="viridis",
    origin="upper",
    interpolation="nearest"
)

plt.colorbar(
    label="Lag detectado [píxeles]"
)

plt.title(
    "Período dominante por región"
)

plt.xlabel("Ventana X")
plt.ylabel("Ventana Y")

plt.tight_layout()
plt.show()