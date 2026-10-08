import pydicom
import numpy as np

# Ruta al DICOM
dcm = pydicom.dcmread("data/ultrasound/primera_medicion/004.dcm")

print("=== DICOM ===")
print("Rows:", getattr(dcm, "Rows", None))
print("Columns:", getattr(dcm, "Columns", None))
print("Number of Frames:", getattr(dcm, "NumberOfFrames", None))

print("\n=== SPACING ===")
print("Pixel Spacing:", getattr(dcm, "PixelSpacing", None))
print("Slice Thickness:", getattr(dcm, "SliceThickness", None))
print("Spacing Between Slices:", getattr(dcm, "SpacingBetweenSlices", None))

print("\n=== ORIENTACIÓN ===")
print("Image Orientation Patient:",
      getattr(dcm, "ImageOrientationPatient", None))
print("Image Position Patient:",
      getattr(dcm, "ImagePositionPatient", None))

print("\n=== DIMENSIÓN DEL ARRAY ===")
print("pixel_array.shape:", dcm.pixel_array.shape)

VELOCIDAD_Z_MM_S = 25.0

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

mm_por_frame_z = calcular_mm_por_frame_z(dcm.FrameTimeVector)
print("\n=== ===")
print(mm_por_frame_z)