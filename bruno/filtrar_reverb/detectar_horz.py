import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

from local_periodicity_detector import load_bscan

DICOM_PATH = "data/ultrasound/primera_medicion/001.dcm"
FRAME_IDX = 118
CROP = (365, 650, 115, 690)

def main():
    bscan = load_bscan(DICOM_PATH, FRAME_IDX, crop=CROP)
    img = bscan.astype(np.float32)

    # Detector de una línea brillante horizontal.
    # Centro positivo, entorno vertical negativo.
    width = 21
    kernel = np.zeros((5, width), dtype=np.float32)

    kernel[0, :] = -1
    kernel[1, :] = -1
    kernel[2, :] = 4
    kernel[3, :] = -1
    kernel[4, :] = -1

    response = convolve(img, kernel, mode="reflect")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Original
    axes[0].imshow(img, cmap="gray", aspect="auto")
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Respuesta del detector
    max_val = np.max(np.abs(response))
    im = axes[1].imshow(
        response,
        cmap="coolwarm",
        vmin=-max_val,
        vmax=max_val,
        aspect="auto"
    )
    axes[1].set_title("Respuesta del detector")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()