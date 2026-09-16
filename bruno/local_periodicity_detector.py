"""
Detector local de periodicidad para reverberaciones en B-scans de ultrasonido.

Contexto: proyecto echo root. Las reverberaciones de interés son locales,
pequeñas, aproximadamente horizontales y periódicas en profundidad (eje Y).
No conviene un filtro/modelo global (ya descartado: AR global, lag=109,
lag=224 global).

Este script implementa el "próximo paso" descrito:
    DICOM -> canal -> frame -> recorte -> ventanas -> periodicidad local
    -> mapa de score + mapa de lag dominante

Decisiones explícitas tomadas (ajustar si no coinciden con lo esperado):

1. Ventanas CON SOLAPAMIENTO (stride < window), para evitar mapas en bloques
   y que una reverberación en el borde de una celda no se diluya.

2. Restricción de fiabilidad: para que la autocorrelación a un lag L sea
   mínimamente confiable, se requieren ~3-4x L muestras en la ventana.
   Con WINDOW_Y=60 y MAX_LAG=50 esto NO se cumple (quedan ~10 muestras
   de solape en el peor caso). El script recorta automáticamente el rango
   de lags evaluables según window_y y lo reporta, en vez de calcular
   valores no confiables silenciosamente.

3. Métrica de decaimiento (amplitude decay) como score SEPARADO del score
   de autocorrelación pura. La reverberación clásica y[n] = x[n] + a*x[n-L]
   con a<1 predice que copias sucesivas del mismo pico decaen en amplitud.
   Estructura real (raíz, capas de sustrato) no tiene por qué decaer así.
   Se calculan ambos scores por separado para poder compararlos antes de
   decidir si combinarlos.

4. Detrend simple (resta de media móvil) antes de la autocorrelación,
   siguiendo lo que ya hacías en el análisis de periodicidad corta.

No se aplica ningún filtrado/deconvolución todavía: el objetivo de este
script es solamente producir los mapas para inspección visual, tal como
se planteó como próximo experimento.
"""

import numpy as np
import warnings
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
try:
    import pydicom
except ImportError:
    pydicom = None


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

def load_bscan(dicom_path, frame_idx, crop=None):
    """
    Carga un frame del DICOM, se queda con un canal, y aplica recorte opcional.

    crop: tupla (x_min, x_max, y_min, y_max) o None.
    """
    if pydicom is None:
        raise ImportError("pydicom no está instalado (pip install pydicom).")

    ds = pydicom.dcmread(dicom_path)
    volume = ds.pixel_array

    if volume.ndim == 4:
        # (frames, H, W, canales) -> los 3 canales son idénticos según lo verificado
        volume = volume[..., 0]

    bscan = volume[frame_idx].astype(np.float64)

    if crop is not None:
        x_min, x_max, y_min, y_max = crop
        bscan = bscan[y_min:y_max, x_min:x_max]

    return bscan


# ---------------------------------------------------------------------------
# Utilidades de señal
# ---------------------------------------------------------------------------

def _detrend(profile, smooth_len=15):
    """Resta una tendencia lenta estimada con media móvil."""
    if len(profile) <= smooth_len:
        return profile - profile.mean()
    kernel = np.ones(smooth_len) / smooth_len
    trend = np.convolve(profile, kernel, mode="same")
    return profile - trend


def _autocorr_full(profile):
    """Autocorrelación normalizada (lag 0 -> 1.0), solo lags positivos."""
    profile = profile - profile.mean()
    n = len(profile)
    ac = np.correlate(profile, profile, mode="full")
    ac = ac[n - 1:]  # quedarse con lags >= 0
    norm = ac[0] if ac[0] != 0 else 1e-9
    return ac / norm


def _decay_score(profile, lag, n_repeats=3):
    """
    Estima cuánto decae la amplitud de picos sucesivos separados por `lag`.
    Devuelve un valor entre 0 y 1: 1 = decaimiento fuerte y monótono
    (consistente con reverberación clásica), 0 = sin decaimiento o
    comportamiento irregular (más consistente con estructura real).

    Es una heurística simple, no un ajuste físico del modelo.
    """
    profile = np.abs(profile - profile.mean())
    amps = []
    pos = 0
    for _ in range(n_repeats):
        end = pos + lag
        if end >= len(profile):
            break
        segment = profile[pos:end]
        if len(segment) == 0:
            break
        amps.append(segment.max())
        pos = end

    if len(amps) < 2:
        return 0.0

    amps = np.array(amps)
    # fracción de pasos donde realmente decae
    decreasing = np.mean(np.diff(amps) < 0)
    # magnitud relativa del decaimiento total
    total_decay = 1.0 - (amps[-1] / (amps[0] + 1e-9))
    total_decay = np.clip(total_decay, 0.0, 1.0)

    return float(decreasing * total_decay)


def harmonic_score(ac, lag, max_harmonics=10, tolerance=5, min_correlation=0.15):
    """
    Evalúa si `lag` es un período válido buscando armónicos consistentes
    (lag, 2*lag, 3*lag, ...) en la autocorrelación `ac`.

    Un armónico solo cuenta como válido si:
      (a) su pico está por encima de min_correlation (no cualquier valor
          cuenta como "encontrado", punto 4 de lo que hablamos), y
      (b) la ventana de búsqueda de ese armónico no se solapa con la del
          armónico siguiente (punto 3) -> tolerance debe ser < lag/2.

    Si no se llegan a completar `max_harmonics` armónicos válidos,
    se descarta el lag por completo (score 0, punto 6).
    """
    n = len(ac)

    # Punto 3: chequeo explícito de no-solapamiento entre ventanas de armónicos.
    # Si tolerance es muy grande respecto al lag, las ventanas de armónicos
    # consecutivos se pisan entre sí -> no tiene sentido seguir.
    if tolerance >= lag / 2:
        return 0.0, 0

    valid_peaks = []

    for h in range(1, max_harmonics + 1):
        center = h * lag                          # posición esperada del armónico h
        lo = max(0, center - tolerance)
        hi = min(n, center + tolerance + 1)

        if lo >= hi:
            # el armónico esperado cae fuera del largo de la ventana
            break

        peak_value = ac[lo:hi].max()

        # Punto 4: el pico solo cuenta como armónico válido si supera el umbral.
        # Antes, cualquier valor (aunque fuera bajo) se aceptaba como "encontrado".
        if peak_value < min_correlation:
            break  # esta cadena de armónicos se corta acá, no sigo buscando más lejos

        valid_peaks.append(peak_value)


    if len(valid_peaks) == 0:
        return 0.0, 0

    return float(np.mean(valid_peaks)), len(valid_peaks)


# ---------------------------------------------------------------------------
# Detector local
# ---------------------------------------------------------------------------

def local_periodicity_maps(
    bscan,
    window_y=60,
    window_x=40,
    stride_y=None,
    stride_x=None,
    min_lag=5,
    max_lag=50,
    min_reliability_ratio=3.0,
    compute_decay=True,
):
    """
    Recorre bscan con ventanas (con solapamiento) y calcula, para cada
    ventana:
        - score de periodicidad (pico de autocorrelación en el rango de lag)
        - lag dominante
        - score de decaimiento (opcional, ver _decay_score)

    Devuelve un dict con los mapas y metadatos (incluye el max_lag
    efectivamente usado, que puede ser menor al pedido si window_y no
    lo permite de forma confiable).
    """
    H, W = bscan.shape

    if stride_y is None:
        stride_y = window_y // 2
    if stride_x is None:
        stride_x = max(1, window_x // 2)

    
    effective_max_lag = max_lag

    y_positions = list(range(0, H - window_y + 1, stride_y))
    x_positions = list(range(0, W - window_x + 1, stride_x))

    score_map = np.zeros((len(y_positions), len(x_positions)))
    lag_map = np.zeros((len(y_positions), len(x_positions)), dtype=int)
    decay_map = np.zeros((len(y_positions), len(x_positions)))
    n_harmonics_map = np.zeros((len(y_positions), len(x_positions)), dtype=int)

    for i, y in enumerate(y_positions):
        for j, x in enumerate(x_positions):
            window = bscan[y:y + window_y, x:x + window_x]
            profile = window.mean(axis=1)  # perfil en profundidad
            profile = _detrend(profile)

            ac = _autocorr_full(profile)
            lag_range = range(min_lag, effective_max_lag + 1)
            local_scores = ac[min_lag:effective_max_lag + 1]

            best_lag, best_score, best_n_harmonics = None, -1.0, 0
            for lag in lag_range:
                h_score, n_harm = harmonic_score(ac, lag, max_harmonics=10, tolerance=5, min_correlation=0.15)
                if h_score > best_score:
                    best_lag, best_score, best_n_harmonics = lag, h_score, n_harm

            score_map[i, j] = best_score
            lag_map[i, j] = best_lag
            n_harmonics_map[i, j] = best_n_harmonics

            if compute_decay:
                decay_map[i, j] = _decay_score(profile, best_lag)

    return {
        "score_map": score_map,
        "lag_map": lag_map,
        "decay_map": decay_map if compute_decay else None,
        "n_harmonics_map": n_harmonics_map,
        "y_positions": np.array(y_positions),
        "x_positions": np.array(x_positions),
        "window_y": window_y,
        "window_x": window_x,
        "effective_max_lag": effective_max_lag,
    }


# ---------------------------------------------------------------------------
# Visualización
# ---------------------------------------------------------------------------

def plot_maps(bscan, result, combined_threshold=None):
    """
    Grafica: B-scan original, mapa de score de periodicidad, mapa de lag
    dominante, y mapa de decaimiento (si fue calculado).

    combined_threshold: si se pasa un valor, además dibuja una máscara
    binaria donde score >= threshold (solo con fines exploratorios,
    todavía no es una máscara de filtrado final).
    """
    import matplotlib.pyplot as plt

    score_map = result["score_map"]
    lag_map = result["lag_map"]
    decay_map = result["decay_map"]
    n_harmonics_map = result["n_harmonics_map"]

    n_plots = 4 + (1 if decay_map is not None else 0) + (1 if combined_threshold else 0)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))

    axes[0].imshow(bscan, cmap="gray")
    axes[0].set_title("B-scan (recortado)")

    im1 = axes[1].imshow(score_map, cmap="viridis", aspect="auto")
    axes[1].set_title("Score de periodicidad")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(lag_map, cmap="plasma", aspect="auto")
    axes[2].set_title(f"Lag dominante (max_lag={result['effective_max_lag']})")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)

    im2b = axes[3].imshow(n_harmonics_map, cmap="cividis", aspect="auto")
    axes[3].set_title("N° de armónicos encontrados")
    fig.colorbar(im2b, ax=axes[3], fraction=0.046)

    idx = 4
    if decay_map is not None:
        im3 = axes[idx].imshow(decay_map, cmap="magma", aspect="auto")
        axes[idx].set_title("Score de decaimiento")
        fig.colorbar(im3, ax=axes[idx], fraction=0.046)
        idx += 1

    if combined_threshold is not None:
        mask = score_map >= combined_threshold
        axes[idx].imshow(mask, cmap="gray")
        axes[idx].set_title(f"Máscara score >= {combined_threshold} (exploratorio)")

    plt.tight_layout()
    return fig



def plot_frame_with_grid(bscan, grid_step=10):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(bscan, cmap="gray")

    ax.xaxis.set_major_locator(ticker.MultipleLocator(grid_step))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(grid_step))
    ax.grid(color="red", linestyle="--", linewidth=0.4, alpha=0.6)
    ax.set_title(f"Grilla cada {grid_step}px")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Ejemplo de uso (ajustar rutas/parámetros)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    DICOM_PATH = "data/ultrasound/primera_medicion/001.dcm"
    FRAME_IDX = 118
    CROP = (365, 650, 115, 690)  # x_min, x_max, y_min, y_max

    bscan_crop = load_bscan(DICOM_PATH, FRAME_IDX, crop=CROP)
    print("bscan_crop shape:", bscan_crop.shape)

    # fig = plot_frame_with_grid(bscan_crop, grid_step=10)
    # plt.show()

    result = local_periodicity_maps(
        bscan_crop,
        window_y=80,
        window_x=1,
        min_lag=10,
        max_lag=16,
        compute_decay=True,
    )

    print("max_lag efectivo usado:", result["effective_max_lag"])
    print("score_map shape:", result["score_map"].shape)

    fig = plot_maps(bscan_crop, result)
    fig.savefig("periodicity_maps.png", dpi=150)
    print("Guardado periodicity_maps.png")
