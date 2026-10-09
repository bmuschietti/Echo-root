import numpy as np
import numpy as np
from scipy.ndimage import map_coordinates
import pydicom
import nibabel as nib
from PIL import Image
import matplotlib.pyplot as plt



"""Registro rígido 3D (Euler) basado en intensidad: vol_a (fijo) vs vol_b rotado 180° (móvil).
Métrica: MSE. Volúmenes crudos, un canal, sin máscara.

Uso: python registro_baseline.py ruta_a.dcm ruta_b.dcm [--out salida]
"""
import argparse
import os
import numpy as np
import pydicom
import itk

# ---------------- Configuración ----------------
MM_PER_PIXEL_XY = 0.121847
MM_PER_FRAME_Z = 1.049983510638298   # calculado con un volumen; ver nota abajo
CROP_FILA_MIN, CROP_FILA_MAX = 113, 694
CROP_COL_MIN, CROP_COL_MAX = 365, 652
# ------------------------------------------------

SPACING = (MM_PER_PIXEL_XY, MM_PER_PIXEL_XY, MM_PER_FRAME_Z)  # orden ITK: x, y, z


def cargar_dicom(path):
    """Devuelve array (Z, Y, X) float32, un solo canal."""
    arr = pydicom.dcmread(path).pixel_array      # (236, 768, 1024, 3)
    return arr[..., 0].astype(np.float32)


def recortar(vol):
    return vol[:, CROP_FILA_MIN:CROP_FILA_MAX, CROP_COL_MIN:CROP_COL_MAX]


def a_itk(vol):
    img = itk.image_from_array(np.ascontiguousarray(vol, dtype=np.float32))
    img.SetSpacing(SPACING)
    return img


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def main():
    vol_a = cargar_dicom("data/ultrasound/primera_medicion/001.dcm")
    vol_b = cargar_dicom("data/ultrasound/primera_medicion/004.dcm")
    out = "bruno/volumenes"
    os.makedirs(out, exist_ok=True)

    print("shapes:", vol_a.shape, vol_b.shape)

    # Flip 180° en XY sobre la imagen completa, y recién después el recorte
    fixed_np = recortar(vol_a)
    moving_np = recortar(vol_b)[:, ::-1, ::-1]

    fixed = a_itk(fixed_np)
    moving = a_itk(moving_np)

    mse_antes = mse(fixed_np, moving_np)
    print(f"MSE antes del registro: {mse_antes:.4f}")

    # Parámetros Elastix
    po = itk.ParameterObject.New()
    p = po.GetDefaultParameterMap("rigid")   # EulerTransform 3D
    p["Metric"] = ["AdvancedMeanSquares"]
    p["AutomaticTransformInitialization"] = ["true"]
    p["AutomaticTransformInitializationMethod"] = ["GeometricalCenter"]
    p["AutomaticScalesEstimation"] = ["true"]
    p["NumberOfResolutions"] = ["3"]
    p["ImagePyramidSchedule"] = ["8", "8", "1", "4", "4", "1", "2", "2", "1"]
    p["MaximumNumberOfIterations"] = ["500"]
    p["NumberOfSpatialSamples"] = ["50000"]
    p["ResultImagePixelType"] = ["float"]
    po.AddParameterMap(p)

    resultado, params = itk.elastix_registration_method(
        fixed, moving, parameter_object=po, log_to_console=True)

    res_np = itk.array_from_image(resultado)
    mse_despues = mse(fixed_np, res_np)

    tp = [float(x) for x in params.GetParameter(0, "TransformParameters")]
    centro = [float(x) for x in params.GetParameter(0, "CenterOfRotationPoint")]
    print("\n=== Resultados ===")
    print(f"MSE antes:   {mse_antes:.4f}")
    print(f"MSE después: {mse_despues:.4f}")
    print(f"rx, ry, rz (rad): {tp[0]:.6f}, {tp[1]:.6f}, {tp[2]:.6f}")
    print(f"rx, ry, rz (deg): {np.degrees(tp[0]):.4f}, {np.degrees(tp[1]):.4f}, {np.degrees(tp[2]):.4f}")
    print(f"tx, ty, tz (mm):  {tp[3]:.4f}, {tp[4]:.4f}, {tp[5]:.4f}")
    print(f"Centro de rotación (mm): {centro}")

    # Guardados
    itk.imwrite(resultado, os.path.join(out, "moving_registrado.nii.gz"))
    itk.imwrite(fixed, os.path.join(out, "fixed.nii.gz"))
    itk.imwrite(moving, os.path.join(out, "moving_original.nii.gz"))
    params.WriteParameterFile(params.GetParameterMap(0),
                              os.path.join(out, "transform_params.txt"))


if __name__ == "__main__":
    main()