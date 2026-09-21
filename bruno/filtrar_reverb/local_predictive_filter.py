"""
Filtro predictivo local para atenuar reverberación usando los lags
detectados por local_periodicity_maps.

Método (por ventana, todo-o-nada por umbral de score):
    1. Tomar el lag dominante y el score de esa ventana.
    2. Si score >= SCORE_THRESHOLD:
         - estimar alpha por mínimos cuadrados entre x[n] y x[n-lag]
           (alpha = <x[n], x[n-lag]> / <x[n-lag], x[n-lag]>)
         - residuo = x[n] - alpha * x[n-lag]
       Si score < SCORE_THRESHOLD:
         - residuo = x[n] (ventana sin modificar)
    3. Recombinar todas las ventanas con overlap-add (promediando solapes)
       para evitar discontinuidades de bloque.

NOTA: esta es la versión simple, todo-o-nada. No distingue todavía
zonas de raíz real mezcladas con reverb (limitación conocida, sin
resolver en esta primera prueba).
"""

import numpy as np
import matplotlib.pyplot as plt

from local_periodicity_detector import load_bscan, local_periodicity_maps  

# ----------------------------------------------------------------------
# PARÁMETROS - ajustar acá
# ----------------------------------------------------------------------
DICOM_PATH = "data/ultrasound/primera_medicion/001.dcm"      # ajustar
FRAME_IDX = 118                                        # ajustar
CROP = (365, 650, 115, 690)                             # ej: (x_min, x_max, y_min, y_max)

WINDOW_Y = 40
STRIDE_Y = 40
WINDOW_X = 20
STRIDE_X = 10

SCORE_THRESHOLD = 0.3  # umbral para decidir si se filtra la ventana o no


def apply_local_predictive_filter(bscan, maps, score_threshold):
    """
    Aplica el filtro predictivo local ventana por ventana y recombina
    con overlap-add.

    bscan: array 2D (H, W), float64.
    maps: dict devuelto por local_periodicity_maps (debe incluir
          score_map, lag_map, y_positions, x_positions, window_y, window_x).
    """
    H, W = bscan.shape

    score_map = maps["score_map"]
    lag_map = maps["lag_map"]
    y_positions = maps["y_positions"]
    x_positions = maps["x_positions"]
    window_y = maps["window_y"]
    window_x = maps["window_x"]

    filtered_sum = np.zeros((H, W), dtype=np.float64)
    weight_sum = np.zeros((H, W), dtype=np.float64)

    # ventana de pesado simple (rectangular); se puede cambiar a Hanning
    # si el overlap-add da artefactos en los bordes de ventana
    win_weight = np.ones((window_y, window_x), dtype=np.float64)

    n_filtered = 0
    n_total = 0
    alphas = []  # DIAGNÓSTICO: guardamos todos los alpha calculados

    for i, y0 in enumerate(y_positions):
        for j, x0 in enumerate(x_positions):
            n_total += 1
            y1 = y0 + window_y
            x1 = x0 + window_x

            block = bscan[y0:y1, x0:x1]
            score = score_map[i, j]
            lag = int(lag_map[i, j])

            if score >= score_threshold and lag > 0 and lag < window_y:
                # separar la señal retrasada dentro del mismo bloque
                x_n = block[lag:, :]
                x_lag = block[:-lag, :]

                denom = np.sum(x_lag * x_lag)
                if denom > 1e-8:
                    alpha = np.sum(x_n * x_lag) / denom
                else:
                    alpha = 0.0

                alphas.append(alpha)

                residual = block.copy()
                residual[lag:, :] = x_n - alpha * x_lag
                # las primeras `lag` filas del bloque no tienen x[n-lag]
                # disponible dentro del bloque; se dejan sin modificar
                n_filtered += 1
            else:
                residual = block

            filtered_sum[y0:y1, x0:x1] += residual * win_weight
            weight_sum[y0:y1, x0:x1] += win_weight

    # evitar división por cero en zonas no cubiertas por ninguna ventana
    weight_sum[weight_sum == 0] = 1.0
    filtered = filtered_sum / weight_sum

    print(f"Ventanas filtradas: {n_filtered}/{n_total} "
          f"({100 * n_filtered / n_total:.1f}%) con score >= {score_threshold}")


    # DIAGNÓSTICO: estadísticas de alpha
    alphas = np.array(alphas)
    if len(alphas) > 0:
        print(f"alpha -> min={alphas.min():.3f}  max={alphas.max():.3f}  "
              f"mean={alphas.mean():.3f}  median={np.median(alphas):.3f}")
        print(f"  fracción con |alpha| > 1: "
              f"{100 * np.mean(np.abs(alphas) > 1):.1f}%")
    else:
        print("No hubo ventanas filtradas, no hay alphas para reportar.")


    return filtered


def main():
    bscan = load_bscan(DICOM_PATH, FRAME_IDX, crop=CROP)

    maps = local_periodicity_maps(
        bscan,
        window_y=WINDOW_Y,
        stride_y=STRIDE_Y,
        window_x=WINDOW_X,
        stride_x=STRIDE_X,
        min_lag=10,
        max_lag=16,
        compute_decay=True,
    )

    filtered = apply_local_predictive_filter(bscan, maps, SCORE_THRESHOLD)

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    axes[0].imshow(bscan, cmap="gray", aspect="auto")
    axes[0].set_title("Original")

    axes[1].imshow(filtered, cmap="gray", aspect="auto")
    axes[1].set_title(f"Filtrado (score >= {SCORE_THRESHOLD})")

    diff = bscan - filtered
    im = axes[2].imshow(diff, cmap="gray", aspect="auto")
    axes[2].set_title("Diferencia (removido)")
    plt.colorbar(im, ax=axes[2])

    plt.tight_layout()
    #plt.savefig("local_predictive_filter_result.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
