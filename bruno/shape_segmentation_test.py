"""
Prueba rápida de la hipótesis: ¿la relación de aspecto (ancho/alto) de
los blobs segmentados separa reverb (elongada en x) de raíz (extensión
en ambos ejes / diagonal)?

Método:
    1. Umbralizar el B-scan (Otsu).
    2. Limpieza morfológica opcional (para no quedarnos con ruido suelto).
    3. Etiquetar componentes conexas.
    4. Para cada blob: bounding box, relación de aspecto (ancho/alto),
       y orientación/elongación real vía momentos (PCA / regionprops),
       que es más robusta que el bounding box si el blob es diagonal.
    5. Graficar el B-scan con cada blob coloreado según su relación de
       aspecto, y un histograma de esa métrica.

Esto NO es la solución final, es solo para ver rápido si el criterio de
forma separa algo antes de invertir en algo más pesado (SAM, etc.).
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, morphology, measure

from local_periodicity_detector import load_bscan  

# ----------------------------------------------------------------------
# PARÁMETROS - ajustar acá
# ----------------------------------------------------------------------
DICOM_PATH = "data/ultrasound/primera_medicion/001.dcm"      # ajustar
FRAME_IDX = 118                           # ajustar
CROP = (365, 650, 115, 690)                              # ej: (x_min, x_max, y_min, y_max)

MIN_BLOB_AREA = 0       # descartar blobs más chicos que esto (ruido suelto)
OPENING_RADIUS = 1       # limpieza morfológica opcional (0 para desactivar)


def segment_and_measure(bscan):
    """
    Devuelve label_image y una lista de dicts con métricas por blob.
    """
    # Otsu sobre el bscan (asumido float64, se normaliza a 0-1 para el umbral)
    norm = (bscan - bscan.min()) / (bscan.max() - bscan.min() + 1e-8)
    thresh = filters.threshold_otsu(norm)
    binary = norm > thresh

    if OPENING_RADIUS > 0:
        binary = morphology.binary_opening(
            binary, morphology.disk(OPENING_RADIUS)
        )

    label_image = measure.label(binary)
    regions = measure.regionprops(label_image, intensity_image=bscan)

    blobs = []
    for r in regions:
        if r.area < MIN_BLOB_AREA:
            continue

        # bounding box: ancho/alto crudo
        min_row, min_col, max_row, max_col = r.bbox
        h = max_row - min_row
        w = max_col - min_col
        bbox_aspect = w / h if h > 0 else np.inf

        # elongación real vía ejes principales (más robusta a diagonales
        # que el bounding box, que sobreestima el aspect ratio de blobs
        # inclinados)
        major = r.major_axis_length
        minor = r.minor_axis_length if r.minor_axis_length > 0 else 1e-3
        axis_ratio = major / minor

        blobs.append({
            "label": r.label,
            "area": r.area,
            "bbox": r.bbox,
            "bbox_aspect": bbox_aspect,   # w/h del bounding box
            "axis_ratio": axis_ratio,     # elongación real (eje mayor/menor)
            "orientation_deg": np.degrees(r.orientation),  # 0 = horizontal
            "centroid": r.centroid,
        })

    return label_image, blobs


def main():
    bscan = load_bscan(DICOM_PATH, FRAME_IDX, crop=CROP)
    label_image, blobs = segment_and_measure(bscan)

    print(f"Blobs encontrados (area >= {MIN_BLOB_AREA}): {len(blobs)}")
    for b in sorted(blobs, key=lambda x: -x["area"])[:15]:
        print(f"  label={int(b['label']):4d}  area={int(b['area']):5d}    "
              f"bbox_aspect={b['bbox_aspect']:6.2f}  "
              f"axis_ratio={b['axis_ratio']:6.2f}  "
              f"orient={b['orientation_deg']:6.1f}deg  "
              f"centroid={tuple(round(c, 1) for c in b['centroid'])}")

    # --- Visualización ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(bscan, cmap="gray", aspect="auto")
    axes[0].set_title("B-scan original")

    axes[1].imshow(label_image, cmap="nipy_spectral", aspect="auto")
    axes[1].set_title(f"Componentes conexas ({len(blobs)} blobs)")

    # colorear el bscan según axis_ratio de cada blob para ver
    # visualmente si las bandas de reverb se distinguen de la raíz
    aspect_overlay = np.zeros_like(bscan, dtype=np.float64)
    for b in blobs:
        aspect_overlay[label_image == b["label"]] = b["axis_ratio"]

    im = axes[2].imshow(aspect_overlay, cmap="viridis", aspect="auto")
    axes[2].set_title("axis_ratio por blob (mayor = más elongado)")
    plt.colorbar(im, ax=axes[2])

    plt.tight_layout()
    plt.savefig("shape_segmentation_test.png", dpi=150)
    plt.show()

    # --- Histograma de axis_ratio, para ver si hay separación clara ---
    ratios = [b["axis_ratio"] for b in blobs]
    plt.figure(figsize=(6, 4))
    plt.hist(ratios, bins=30)
    plt.xlabel("axis_ratio (eje mayor / eje menor)")
    plt.ylabel("n° de blobs")
    plt.title("Distribución de elongación de blobs")
    plt.tight_layout()
    plt.savefig("axis_ratio_histogram.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
