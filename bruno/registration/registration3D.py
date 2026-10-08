import numpy as np
import numpy as np
from scipy.ndimage import map_coordinates
import pydicom
import nibabel as nib
from PIL import Image
import matplotlib.pyplot as plt
from bruno.fusion_hexagonal import (
    coords_globales_a_locales,
    muestrear_volumen,
    construir_grilla_global,
    cargar_volumen,
    calcular_mm_por_frame_z,
    MM_PER_PIXEL_XY,
    APOTEMA_MM,
)