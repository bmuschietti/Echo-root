"""
Fusión de 6 volúmenes (uno por cara de la maceta hexagonal) en un único
volumen global, usando la transformación geométrica CONOCIDA
(rotación de 60° por cara + traslación por apotema).

No se hace registro/estimación de pose: la transformación es determinista.
"""

import numpy as np
from scipy.ndimage import map_coordinates
import pydicom
import nibabel as nib
from PIL import Image
import napari
import matplotlib.pyplot as plt


# =========================================================================
# PARÁMETROS FÍSICOS (algunos confirmados, otros PENDIENTES de medir)
# =========================================================================

MM_PER_PIXEL_XY = 0.121847   # confirmado (PhysicalDeltaX/Y del DICOM)
VELOCIDAD_Z_MM_S = 25.0      # confirmado (velocidad del carro vertical)
# MM_PER_PIXEL_Z ya no es constante fija: se calcula por volumen a partir
# del FrameTimeVector real de cada DICOM (ver calcular_mm_por_frame_z).

LADO_HEXAGONO_MM = 41.0      # confirmado (medida interior, plano maceta)
APOTEMA_MM = (LADO_HEXAGONO_MM / 2) * np.sqrt(3)  # ≈ 35.5 mm (hexágono regular)

N_CARAS = 6
ROTACION_GRADOS = 60.0       # confirmado, calibración mecánica exacta

# Dimensiones de cada frame (confirmado por metadata DICOM)
# FRAME_ROWS = 768   # eje y_local (profundidad)
# FRAME_COLS = 1024  # eje x_local (lateral)

def guardar_como_nifti(volumen, mm_per_pixel_xy, mm_por_frame_z, path_salida):
    """
    Guarda un volumen numpy (Z, Rows, Cols) como NIfTI (.nii.gz)
    para visualizar en ITK-SNAP, con el espaciado real de vóxel.
    """
    # nibabel espera orden de ejes (X, Y, Z), hay que transponer
    volumen_nifti = np.transpose(volumen, (2, 1, 0))  # (Z,Y,X) -> (X,Y,Z)

    affine = np.diag([mm_per_pixel_xy, mm_per_pixel_xy, mm_por_frame_z, 1.0])

    img = nib.Nifti1Image(volumen_nifti.astype(np.float32), affine)
    nib.save(img, path_salida)


CROP_FILA_MIN, CROP_FILA_MAX = 113, 694
CROP_COL_MIN, CROP_COL_MAX = 365, 652

def cargar_volumen(path_dcm):
    ds = pydicom.dcmread(path_dcm)
    pixel_array = ds.pixel_array

    if pixel_array.ndim == 4 and pixel_array.shape[-1] == 3:
        volumen = pixel_array.astype(np.float32).mean(axis=-1)
    elif pixel_array.ndim == 3:
        volumen = pixel_array.astype(np.float32)
    else:
        raise ValueError(f"Forma de pixel_array no esperada: {pixel_array.shape}")

    volumen = volumen[:, CROP_FILA_MIN:CROP_FILA_MAX, CROP_COL_MIN:CROP_COL_MAX]
    return volumen

def calcular_mm_por_frame_z(frame_time_vector_ms, velocidad_mm_s=VELOCIDAD_Z_MM_S):
    """
    Calcula el espaciado real en z (mm/frame) a partir del FrameTimeVector
    del DICOM (tiempo entre frames en ms) y la velocidad conocida del carro.

    frame_time_vector_ms: lista/array de tiempos entre frames (el primer
    valor suele ser 0, correspondiente al frame inicial).
    """
    tiempos = np.array(frame_time_vector_ms, dtype=float)
    tiempos_validos = tiempos[tiempos > 0]  # descarta el 0 inicial
    frame_time_promedio_s = np.mean(tiempos_validos) / 1000.0
    mm_por_frame = velocidad_mm_s * frame_time_promedio_s
    return mm_por_frame


def construir_grilla_global(apotema_mm, mm_per_pixel_xy, mm_per_pixel_z,
                             n_frames_z, margen_mm=5.0):
    """
    Define el tamaño y resolución del volumen global de salida.
    Cubre un círculo de radio ~apotema (+margen) centrado en el eje z.
    """
    radio_mm = apotema_mm + margen_mm
    lado_px = int(np.ceil(2 * radio_mm / mm_per_pixel_xy))

    z_total_mm = n_frames_z * mm_per_pixel_z
    z_px = n_frames_z  # misma resolución en z que los volúmenes originales

    shape_global = (z_px, lado_px, lado_px)  # (Z, Y, X)
    origen_mm = (-radio_mm, -radio_mm)       # esquina (X,Y) = (0,0) del array global
    return shape_global, origen_mm


def coords_globales_a_locales(X_mm, Y_mm, indice_cara, apotema_mm,
                               ancho_frame_mm, rotacion_grados=ROTACION_GRADOS):
    theta = np.radians(rotacion_grados * indice_cara)
    cos_t, sin_t = np.cos(-theta), np.sin(-theta)
    X_rot = X_mm * cos_t - Y_mm * sin_t
    Y_rot = X_mm * sin_t + Y_mm * cos_t

    x_local_mm = X_rot + (ancho_frame_mm / 2)  
    y_local_mm = apotema_mm - Y_rot

    return x_local_mm, y_local_mm


def muestrear_volumen(volumen, x_local_mm, y_local_mm, z_idx,
                       mm_per_pixel_xy):
    """
    Interpola el valor de intensidad de `volumen` en la posición
    (x_local_mm, y_local_mm) para cada plano z_idx.
    Devuelve NaN donde la posición cae fuera del frame.
    """
    x_px = x_local_mm / mm_per_pixel_xy
    y_px = y_local_mm / mm_per_pixel_xy

    fuera_de_rango = (
        (x_px < 0) | (x_px >= volumen.shape[2]) |
        (y_px < 0) | (y_px >= volumen.shape[1])
    )

    coords = np.array([z_idx, y_px, x_px])
    valores = map_coordinates(
        volumen, coords, order=1, mode="constant", cval=np.nan
    )
    valores[fuera_de_rango] = np.nan
    return valores


def fusionar_volumenes(volumenes, apotema_mm, mm_per_pixel_xy,
                        mm_per_pixel_z, metodo="promedio"):
    """
    volumenes: lista de 6 arrays numpy, cada uno shape (Z, Rows, Cols)
    Devuelve: volumen global fusionado, shape (Z, lado_px, lado_px)
    """
    assert len(volumenes) == N_CARAS

    n_z = volumenes[0].shape[0]
    shape_global, origen_mm = construir_grilla_global(
        apotema_mm, mm_per_pixel_xy, mm_per_pixel_z, n_z
    )
    z_px, lado_px, _ = shape_global

    # Grilla de coordenadas globales (X, Y) en mm, para un plano z
    x0, y0 = origen_mm
    xs = x0 + np.arange(lado_px) * mm_per_pixel_xy
    ys = y0 + np.arange(lado_px) * mm_per_pixel_xy
    X_mm, Y_mm = np.meshgrid(xs, ys)  # shape (lado_px, lado_px)

    X_flat = X_mm.ravel()
    Y_flat = Y_mm.ravel()

    acumulador = np.full((N_CARAS, z_px, lado_px * lado_px), np.nan,
                          dtype=np.float32)

    ancho_frame_mm = volumenes[0].shape[2] * mm_per_pixel_xy
    for i, vol in enumerate(volumenes):
        x_local_mm, y_local_mm = coords_globales_a_locales(
            X_flat, Y_flat, i, apotema_mm, ancho_frame_mm
        )

        for z in range(z_px):
            z_idx_array = np.full_like(x_local_mm, z)
            valores = muestrear_volumen(
                vol, x_local_mm, y_local_mm, z_idx_array, mm_per_pixel_xy
            )
            acumulador[i, z, :] = valores

    if metodo == "promedio":
        fusionado = np.nanmean(acumulador, axis=0)
    elif metodo == "mediana":
        fusionado = np.nanmedian(acumulador, axis=0)
    elif metodo == "maximo":
        fusionado = np.nanmax(acumulador, axis=0)
    else:
        raise ValueError(f"Método desconocido: {metodo}")

    fusionado = fusionado.reshape(z_px, lado_px, lado_px)
    fusionado = np.nan_to_num(fusionado, nan=0.0)

    return fusionado

def ver_cortes_ortogonales(volumen, titulo=""):
    z, y, x = [s // 2 for s in volumen.shape]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(volumen[z, :, :], cmap="gray")
    axes[0].set_title(f"Axial (z={z})")

    axes[1].imshow(volumen[:, y, :], cmap="gray", aspect="auto")
    axes[1].set_title(f"Coronal (y={y})")

    axes[2].imshow(volumen[:, :, x], cmap="gray", aspect="auto")
    axes[2].set_title(f"Sagital (x={x})")

    fig.suptitle(titulo)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    #Ejemplo de uso (completar con datos reales):
    ds = pydicom.dcmread("data/ultrasound/primera_medicion/001.dcm")
    mm_por_frame_z = calcular_mm_por_frame_z(ds.FrameTimeVector)
    
    volumenes = [cargar_volumen(f"data/ultrasound/primera_medicion/00{i}.dcm") for i in range(1,7)]
    resultado = fusionar_volumenes(
        volumenes,
        apotema_mm=APOTEMA_MM,
        mm_per_pixel_xy=MM_PER_PIXEL_XY,
        mm_per_pixel_z=mm_por_frame_z,
        metodo="promedio",
    )
    print(resultado.shape)

    ver_cortes_ortogonales(resultado, "volumen fusionado")


    guardar_como_nifti(
     resultado,
     mm_per_pixel_xy=MM_PER_PIXEL_XY,
     mm_por_frame_z=mm_por_frame_z,
     path_salida="bruno/volumenes/volumen_fusionado.nii.gz",
 )
    
