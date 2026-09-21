import pydicom
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def extraer_metadata_echoroot(path):
    ds = pydicom.dcmread(path)
    datos = {}

    # --- Escala espacial (x, y) ---
    if hasattr(ds, "SequenceOfUltrasoundRegions"):
        for i, region in enumerate(ds.SequenceOfUltrasoundRegions):
            datos[f"region_{i}_PhysicalDeltaX"] = getattr(region, "PhysicalDeltaX", None)
            datos[f"region_{i}_PhysicalDeltaY"] = getattr(region, "PhysicalDeltaY", None)
            datos[f"region_{i}_UnitsX"] = getattr(region, "PhysicalUnitsXDirection", None)
            datos[f"region_{i}_UnitsY"] = getattr(region, "PhysicalUnitsYDirection", None)
            datos[f"region_{i}_RegionLocationMinX0"] = getattr(region, "RegionLocationMinX0", None)
            datos[f"region_{i}_RegionLocationMaxX1"] = getattr(region, "RegionLocationMaxX1", None)
            datos[f"region_{i}_RegionLocationMinY0"] = getattr(region, "RegionLocationMinY0", None)
            datos[f"region_{i}_RegionLocationMaxY1"] = getattr(region, "RegionLocationMaxY1", None)
    else:
        datos["SequenceOfUltrasoundRegions"] = None

    # --- Dimensiones del frame (para saber tamaño en pixeles) ---
    datos["Rows"] = getattr(ds, "Rows", None)
    datos["Columns"] = getattr(ds, "Columns", None)
    datos["NumberOfFrames"] = getattr(ds, "NumberOfFrames", None)

    # --- Timing (para escala z, si el barrido es a velocidad constante) ---
    datos["FrameTime"] = getattr(ds, "FrameTime", None)              # ms entre frames
    datos["FrameTimeVector"] = getattr(ds, "FrameTimeVector", None)  # si el tiempo entre frames varía
    datos["CineRate"] = getattr(ds, "CineRate", None)                # frames/seg
    datos["RecommendedDisplayFrameRate"] = getattr(ds, "RecommendedDisplayFrameRate", None)
    datos["ActualFrameDuration"] = getattr(ds, "ActualFrameDuration", None)

    # --- Info general del equipo/estudio (útil como metadata de referencia) ---
    datos["Manufacturer"] = getattr(ds, "Manufacturer", None)
    datos["TransducerType"] = getattr(ds, "TransducerType", None)
    datos["PhotometricInterpretation"] = getattr(ds, "PhotometricInterpretation", None)

    return datos

if __name__ == "__main__":
    path = "data/ultrasound/primera_medicion/001.dcm"  # reemplazar
    datos = extraer_metadata_echoroot(path)

    path_png = 'data/crop_masks/mascara.png'
    mascara = np.array(Image.open(path_png).convert("L"))
    filas = np.any(mascara, axis=1)
    columnas = np.any(mascara, axis=0)
    fila_min, fila_max = np.where(filas)[0][[0, -1]]
    col_min, col_max = np.where(columnas)[0][[0, -1]]

    print (fila_min, fila_max, col_min, col_max)

    ds = pydicom.dcmread("data/ultrasound/primera_medicion/001.dcm")
    pixel_array = ds.pixel_array

    frame = pixel_array[0]  # primer frame; cambiar índice para ver otro

    plt.imshow(frame)
    plt.title(f"Frame 0 - shape {frame.shape}")
    plt.colorbar()
    plt.show()
    # for k, v in datos.items():
    #     print(f"{k}: {v}")