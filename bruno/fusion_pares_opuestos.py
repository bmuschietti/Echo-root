"""
Prueba de fusión por PARES DE CARAS OPUESTAS (separadas 180°).

Hipótesis a probar: la cola de reverberación de una cara corre en
sentido opuesto a la de la cara opuesta (en coordenadas globales),
así que promediar solo el par, y descartar lo visto por una sola
cara, podría reducir el artefacto mejor que fusionar las 6 directo.

Este script NO modifica fusion_hexagonal.py; solo importa lo que
necesita de ahí.
"""

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

UMBRAL_INTENSIDAD = 15 #UMBRAL QUE AMBOS VOLUMENES DEBEN SUPERAR PARA QUE SE CONSIDERE RAIZ Y NO REVERB


def fusionar_par_opuesto(vol_a, vol_b, umbral_intensidad):
    """
    Fusiona dos volúmenes de caras opuestas (separadas 180°).

    Se conserva un voxel solo si AMBOS (A y B rotado) superan
    `umbral_intensidad` en esa posición; si no, se descarta (queda en 0).

    vol_a, vol_b: arrays numpy, mismo shape (Z, Rows, Cols).
    """
    assert vol_a.shape == vol_b.shape, "Los dos volúmenes deben tener el mismo shape"

    vol_b_rotado = vol_b[:, ::-1, ::-1]  # 180° en el plano del frame

    ambos_superan = (vol_a > umbral_intensidad) & (vol_b_rotado > umbral_intensidad)

    fusionado = np.mean([vol_a, vol_b_rotado], axis=0)
    fusionado[~ambos_superan] = 0.0

    return fusionado


def ver_corte_axial(vol_a, vol_b, fusionado, z=None,):
    """
    Grafica el mismo corte (índice z) de vol_a, vol_b y el volumen
    fusionado, lado a lado, para comparar visualmente.
    """
    import matplotlib.pyplot as plt

    if z is None:
        z = vol_a.shape[0] // 2

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(vol_a[z, :, :], cmap="gray")
    axes[0].set_title(f"Vol A (z={z})")

    axes[1].imshow(vol_b[z, :, :], cmap="gray")
    axes[1].set_title(f"Vol B (z={z})")

    axes[2].imshow(fusionado[z, :, :], cmap="gray")
    axes[2].set_title(f"Fusionado (z={z})")


    plt.tight_layout()
    plt.show()

def overlay(a, b):
    rgb = np.zeros((*a.shape, 3))
    rgb[...,0] = a / a.max()
    rgb[...,1] = b / b.max()
    return rgb


def n_frames(path):
    ds = pydicom.dcmread(path)
    return ds.pixel_array.shape[0]

if __name__ == "__main__":
    #ds0 = pydicom.dcmread("data/ultrasound/primera_medicion/001.dcm")
    #mm_por_frame_z = calcular_mm_por_frame_z(ds0.FrameTimeVector)
    
    vol_a = cargar_volumen("data/ultrasound/primera_medicion/001.dcm")
    vol_b = cargar_volumen("data/ultrasound/primera_medicion/004.dcm")

    print(n_frames("data/ultrasound/raices reales/1/001.dcm"), n_frames("data/ultrasound/raices reales/1/004.dcm"))
    
    resultado_par = fusionar_par_opuesto(
        vol_a, vol_b,
        umbral_intensidad= UMBRAL_INTENSIDAD
    )

    vol_b_rotado = vol_b[:, ::-1, ::-1]

    for z in [10, vol_a.shape[0]//2, vol_a.shape[0]-10]:
        plt.figure()
        plt.imshow(overlay(vol_a[z], vol_b_rotado[z]))
        plt.title(f"z={z}")
    plt.show()
    

    ver_corte_axial(vol_a, vol_b_rotado, resultado_par, z=90)

