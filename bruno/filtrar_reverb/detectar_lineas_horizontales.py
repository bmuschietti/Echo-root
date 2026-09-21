import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import white_tophat

from local_periodicity_detector import load_bscan

DICOM_PATH = "data/ultrasound/raices reales/1/001.dcm"
FRAME_IDX = 109
CROP = (365, 650, 115, 690)

# Longitud horizontal mínima que queremos resaltar
LINE_LENGTH = 25

def main():
    bscan = load_bscan(DICOM_PATH, FRAME_IDX, crop=CROP)
    img = bscan.astype(np.float32)

    # Estructura horizontal
    footprint = np.ones((1, LINE_LENGTH), dtype=bool)

    # Resalta estructuras brillantes horizontales
    response = white_tophat(img, footprint=footprint)

    # Umbral robusto
    threshold = np.percentile(response, 99)
    mask = response >= threshold

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    axes[0].imshow(img, cmap="gray", aspect="auto")
    axes[0].set_title("Original")

    axes[1].imshow(response, cmap="gray", aspect="auto")
    axes[1].set_title("Respuesta horizontal")

    axes[2].imshow(mask, cmap="gray", aspect="auto")
    axes[2].set_title(f"Detectado (percentil 99.5)")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()