import numpy as np
from scipy.ndimage import map_coordinates
import pydicom
import nibabel as nib
from PIL import Image

import matplotlib.pyplot as plt
import json
import os
import numpy as np


MM_PX = 0.121847  # mm/pixel en x e y
ARCHIVO_MARCAS = "marcas_centroides.json"


def cargar_volumen(path_dcm):
    ds = pydicom.dcmread(path_dcm)
    pixel_array = ds.pixel_array

    if pixel_array.ndim == 4 and pixel_array.shape[-1] == 3:
        volumen = pixel_array.astype(np.float32).mean(axis=-1)
    elif pixel_array.ndim == 3:
        volumen = pixel_array.astype(np.float32)
    else:
        raise ValueError(f"Forma de pixel_array no esperada: {pixel_array.shape}")

    CROP_FILA_MIN, CROP_FILA_MAX = 113, 694
    CROP_COL_MIN, CROP_COL_MAX = 365, 652

    volumen = volumen[:, CROP_FILA_MIN:CROP_FILA_MAX, CROP_COL_MIN:CROP_COL_MAX]
    return volumen


def marcar_frame(vol_a, vol_b_rotado, z):
    """
    Muestra A y B-rotado en el corte z. Clic en el centro de cada elipse en A
    (panel izquierdo), Enter para terminar; luego lo mismo en B-rotado
    (panel derecho), en el mismo orden. Clic derecho borra el último punto.
    Devuelve (pts_a, pts_b) como listas de (x, y) en píxeles.
    """
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10, 9))
    ax_a.imshow(vol_a[z], cmap="gray", aspect="equal")
    ax_b.imshow(vol_b_rotado[z], cmap="gray", aspect="equal")
    ax_a.set_title(f"A (z={z}): marcar elipses, Enter para terminar")
    ax_b.set_title(f"B rotado (z={z})")
    plt.tight_layout()

    plt.sca(ax_a)
    pts_a = plt.ginput(n=-1, timeout=0, show_clicks=True)
    ax_a.plot([p[0] for p in pts_a], [p[1] for p in pts_a], "r+", ms=12)
    ax_b.set_title(f"B rotado (z={z}): mismo orden, Enter para terminar")
    fig.canvas.draw()

    plt.sca(ax_b)
    pts_b = plt.ginput(n=-1, timeout=0, show_clicks=True)
    plt.close(fig)

    if len(pts_a) != len(pts_b):
        raise ValueError(
            f"Cantidad distinta de marcas en z={z}: A={len(pts_a)}, B={len(pts_b)}"
        )
    return pts_a, pts_b


def guardar_marcas(z, pts_a, pts_b, archivo=ARCHIVO_MARCAS):
    datos = {}
    if os.path.exists(archivo):
        with open(archivo, "r") as f:
            datos = json.load(f)
    datos[str(z)] = {"A": [list(p) for p in pts_a], "B_rotado": [list(p) for p in pts_b]}
    with open(archivo, "w") as f:
        json.dump(datos, f, indent=2)


def marcar_varios_z(vol_a, vol_b_rotado, lista_z, archivo=ARCHIVO_MARCAS):
    for z in lista_z:
        pts_a, pts_b = marcar_frame(vol_a, vol_b_rotado, z)
        if len(pts_a) == 0:
            continue
        guardar_marcas(z, pts_a, pts_b, archivo)


def calcular_corrimientos(archivo=ARCHIVO_MARCAS):
    """
    dx = x_B - x_A, dy = y_B - y_A por elipse. Imprime estadísticas por frame
    y globales (en píxeles y en mm). Devuelve dict z -> array (n_elipses, 2).
    """
    with open(archivo, "r") as f:
        datos = json.load(f)

    por_frame = {}
    todos = []
    for z, d in sorted(datos.items(), key=lambda kv: int(kv[0])):
        a = np.array(d["A"], dtype=float)
        b = np.array(d["B_rotado"], dtype=float)
        delta = b - a  # columnas: dx, dy
        por_frame[int(z)] = delta
        todos.append(delta)
        media = delta.mean(axis=0)
        std = delta.std(axis=0)
        print(
            f"z={z}: n={len(delta)} | media dx={media[0]:.1f}, dy={media[1]:.1f} px"
            f" | std dx={std[0]:.1f}, dy={std[1]:.1f} px"
        )

    todos = np.vstack(todos)
    media = todos.mean(axis=0)
    std = todos.std(axis=0)
    print(
        f"\nGlobal (n={len(todos)}): media dx={media[0]:.1f}, dy={media[1]:.1f} px"
        f" ({media[0]*MM_PX:.2f}, {media[1]*MM_PX:.2f} mm)"
        f" | std dx={std[0]:.1f}, dy={std[1]:.1f} px"
    )
    return por_frame


# Uso (en tu script, con vol_a y vol_b_rotado ya cargados):

if __name__ == "__main__":
    
    vol_a = cargar_volumen("data/ultrasound/primera_medicion/001.dcm")
    vol_b = cargar_volumen("data/ultrasound/primera_medicion/004.dcm")
    vol_b_rotado = vol_b[:, ::-1, ::-1]

    marcar_varios_z(vol_a, vol_b_rotado, [109])
    calcular_corrimientos()