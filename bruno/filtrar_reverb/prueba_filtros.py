import os
import sys
import cv2
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pydicom
import SimpleITK as sitk
from utils.dicom_utils import extract_video, rename_dicom_files_sequentially
# from src.volumeReconstructor import VolumeReconstructor
# from src.volumeRegistrator import VolumeRegistrator
# from src.preprocessing import frangi_3d_filter

# # ------------------- SET UP ------------------
# # Extract videos from DICOM files

print("\n" + "="*50)
print("VIDEO EXTRACTION")
print("="*50)

input_dir = 'data/ultrasound/primera_medicion'
rename_dicom_files_sequentially(input_dir)

for filename in os.listdir(input_dir):
        
    input_path = os.path.join(input_dir, filename)
    if os.path.isfile(input_path):
        try:
            video_path = extract_video(input_path)
            print(f"Video extracted in: {video_path}")
        except Exception as e:
            print(f"Could not extract video from {input_path}: {e}")

extracted_videos_dir = 'data/videos'


# # ------------------- ADQUIRIR FRAMES DE EJEMPLO ------------------

ds = pydicom.dcmread("data/ultrasound/primera_medicion/001.dcm")
frames = ds.pixel_array  # forma: (n_frames, alto, ancho) o (n_frames, alto, ancho, 3)

print("Forma del array:", frames.shape)
print("Cantidad de frames:", frames.shape[0])

# # si viene a color, pasar a escala de grises
if frames.ndim == 4:
   frames = np.array([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames])

# # elegir 5 frames espaciados (primero, y otros 4 repartidos)
n = frames.shape[0]
indices = np.linspace(0, n - 1, 5, dtype=int)

for i, idx in enumerate(indices):
    cv2.imwrite(f"bruno/test/mid_frame_{i:03d}.png", frames[idx])


# # ------------------- PROBAR FILTROS ------------------
