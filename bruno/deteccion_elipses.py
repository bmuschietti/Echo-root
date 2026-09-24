import cv2
import numpy as np
import numpy as np
from scipy.ndimage import map_coordinates
import pydicom
import nibabel as nib
from PIL import Image

import matplotlib.pyplot as plt

from fusion_hexagonal import (
    coords_globales_a_locales,
    muestrear_volumen,
    construir_grilla_global,
    cargar_volumen,
    calcular_mm_por_frame_z,
    MM_PER_PIXEL_XY,
    APOTEMA_MM,
)

def detectar_circulos_elipses(frame, dp=1.5, min_dist=20,
                               param1=50, param2=30,
                               min_radius=3, max_radius=20):
    # frame: array 2D, escala de grises, uint8
    img = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)

    # --- Hough circles ---
    circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, dp=dp,
                                minDist=min_dist, param1=param1, param2=param2,
                                minRadius=min_radius, maxRadius=max_radius)

    # --- Elipses vía contornos ---
    _, thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    elipses = [cv2.fitEllipse(c) for c in contours if len(c) >= 5]

    # --- Plot ---
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if circles is not None:
        for x, y, r in np.round(circles[0]).astype(int):
            cv2.circle(out, (x, y), r, (0, 0, 255), 1)
    for e in elipses:
        cv2.ellipse(out, e, (0, 255, 0), 1)

    plt.figure(figsize=(5, 8))
    plt.imshow(out[..., ::-1])
    plt.title(f"circulos(rojo)={0 if circles is None else len(circles[0])}, elipses(verde)={len(elipses)}")
    plt.axis("off")
    plt.show()

    return circles, elipses

# uso:
# detectar_circulos_elipses(vol_a[118])

if __name__ == "__main__":

    vol_a = cargar_volumen("data/ultrasound/primera_medicion/001.dcm")
    detectar_circulos_elipses(vol_a[118])


