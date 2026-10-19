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
MM_PER_PIXEL_XY = 0.156661
MM_PER_FRAME_Z = 1.318
CROP_FILA_MIN, CROP_FILA_MAX = 115, 690
CROP_COL_MIN, CROP_COL_MAX = 400, 620
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
    vol_a = cargar_dicom("data/Escaneo 22-7/Superficial/001.dcm")
    vol_b = cargar_dicom("data/Escaneo 22-7/Superficial/004.dcm")
    out = "bruno/volumenes/superficial/registro1_4"
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
    p["NumberOfSpatialSamples"] = ["200000"]
    p["ResultImagePixelType"] = ["float"]
    #p["ImageSampler"] = ["Full"] # mas costoso
    po.AddParameterMap(p)
    # --- B-spline (segunda etapa, parte del resultado rígido) ---
    pb = po.GetDefaultParameterMap("bspline")
    pb["Metric"] = ["AdvancedMeanSquares", "TransformBendingEnergyPenalty"]
    pb["Metric0Weight"] = ["1.0"]
    pb["Metric1Weight"] = ["10"]          # peso de la regularización (a ajustar)
    pb["FinalGridSpacingInPhysicalUnits"] = ["10", "10", "20"]   # mm entre nodos
    pb["NumberOfResolutions"] = ["3"]
    pb["GridSpacingSchedule"] = ["4.0", "2.0", "1.0"]
    pb["ImagePyramidSchedule"] = ["8", "8", "1", "4", "4", "1", "2", "2", "1"]
    pb["MaximumNumberOfIterations"] = ["500"]
    pb["NumberOfSpatialSamples"] = ["200000"]
    #pb["ImageSampler"] = ["Full"]
    pb["ResultImagePixelType"] = ["float"]
    po.AddParameterMap(pb)

    resultado, params = itk.elastix_registration_method(
        fixed, moving, parameter_object=po, log_to_console=True,)

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
    itk.imwrite(resultado, os.path.join(out, "moving_registrado_bspline.nii.gz"))
    itk.imwrite(fixed, os.path.join(out, "fixed.nii.gz"))
    itk.imwrite(moving, os.path.join(out, "moving_original.nii.gz"))
    params.WriteParameterFile(os.path.join(out, "transform_params_bspline.txt"))


if __name__ == "__main__":
    main()

#producto entre volumene opuestos (pasar a (0,1) y hacer producto para limpiar reverb)
#hacer esto mismo para el reto de volumenes
#tratar de registrar los 3 volumenes entre si, capaz usando mascara solo en el centro que comparten

#ademas pasar el stl a volumen, primero se puede hacer en cualquier dimension olvidandonos del escaneo
#luego con las medidad reales de la raices reescalar a las medidas del escaneo y usar el spacing de este 