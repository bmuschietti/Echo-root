import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pydicom
import SimpleITK as sitk

from exploration.fourier_exploration.export_frames import export_frames_every_n
from exploration.fourier_exploration.fourier_filter_per_frame import remove_horizontal_reverb_fft_smooth, dereverb_1d_cepstrum
from utils.dicom_utils import extract_video, rename_dicom_files_sequentially
from src.VolumeReconstructor import VolumeReconstructor
from src.volumeRegistrator import VolumeRegistrator
from src.preprocessing import frangi_3d_filter


#--------------------CONVERTIR DICOM A .AVI----------------------

# input_dir = 'data/ultrasound/primera_medicion'

# rename_dicom_files_sequentially(input_dir)

# for filename in os.listdir(input_dir):
        
#     input_path = os.path.join(input_dir, filename)
#     if os.path.isfile(input_path):
#         try:
#             video_path = extract_video(input_path)
#             print(f"Video extracted in: {video_path}")
#         except Exception as e:
#             print(f"Could not extract video from {input_path}: {e}")

extracted_videos_dir = 'data/videos'


#--------------------EXTRAER FRAMES DE UN VIDEO----------------------

# frames_dir = 'bruno/frames'
# video1 = 'data/videos/001.AVI'
# export_frames_every_n(video1,frames_dir, n = 20)


#--------------------FILTROS SOBRE UN FRAME----------------------

path = "bruno/frames/frame_0009.png"  # <-- your image filename
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Image not found at path: " + path)

imgf = img.astype(np.float32)
imgf /= imgf.max()

# call the improved filter
cleaned, detected_peaks = remove_horizontal_reverb_fft_smooth(
    imgf,
    prominence=0.04,      # raise to be more selective (0..1)
    max_peaks=9,
    notch_sigma=9.0,      # float allowed and recommended
    notch_strength=1,  # how strongly to suppress the peak (0..1)
    debug=True
    )

# save and show comparison
cv2.imwrite("ultrasound_cleaned_smooth.png", (cleaned * 255).astype(np.uint8))
print("Saved ultrasound_cleaned_smooth.png, detected peak rows (indices):", detected_peaks)
plt.figure(figsize=(8,6))
plt.subplot(1,2,1); plt.imshow(imgf, cmap='gray'); plt.title("Original"); plt.axis('off')
plt.subplot(1,2,2); plt.imshow(cleaned, cmap='gray'); plt.title("Smoothed Notch Result"); plt.axis('off')
plt.show()

# # Apply after your smooth notch result:
dereverb = dereverb_1d_cepstrum(cleaned, qmin=5, qmax=50, alpha=1)
cv2.imwrite("notch_filter_frame_003.png", (dereverb*255).astype(np.uint8))

