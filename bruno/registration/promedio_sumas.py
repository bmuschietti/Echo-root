import itk
import numpy as np
import cv2
import os

# --- Parámetros ---
RUTA_REF = "bruno/volumenes/superficial/3fixed/suma_3fixed.nii.gz"                  # suma elegida como referencia
OTRAS = [("bruno/volumenes/superficial/2fixed/suma_2fixed.nii.gz", -60, 2),             # (ruta, ángulo horario de la referencia a esa suma)
         ("bruno/volumenes/superficial/5fixed/suma_5fixed.nii.gz", 120, 5),
         ("bruno/volumenes/superficial/6fixed/suma_6fixed.nii.gz", 180, 6),
         ("bruno/volumenes/superficial/4fixed/suma_4fixed.nii.gz", 60, 4)]
OUT = "bruno/volumenes/superficial/final/atlas"
ITERACIONES = 30   # confirmar
# -----------------------------
os.makedirs(OUT, exist_ok=True)

ref = itk.imread(RUTA_REF, itk.F)
ref_np = itk.array_from_image(ref).astype(np.float32)   # (z, y, x)
nz, h, w = ref_np.shape
sp, org = np.array(itk.spacing(ref)), np.array(itk.origin(ref))
D = itk.array_from_matrix(ref.GetDirection())
G = np.eye(4); G[:3, :3] = D @ np.diag(sp); G[:3, 3] = org   # índice -> mm

def guardar_rotacion(ang, ruta):
    # Rotación inicial (cv2, centro de la grilla) como AffineTransform fixed(mm) -> moving(mm)
    M = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), -ang, 1.0)
    Mi = np.eye(4); Mi[:2, :2] = M[:, :2]; Mi[:2, 3] = M[:, 2]
    F = np.linalg.inv(G @ Mi @ np.linalg.inv(G))
    params = " ".join(repr(float(v)) for v in list(F[:3, :3].flatten()) + list(F[:3, 3]))
    dir_str = " ".join(repr(float(v)) for v in D.flatten())
    with open(ruta, "w") as f:
        f.write(f'''(Transform "AffineTransform")
(NumberOfParameters 12)
(TransformParameters {params})
(InitialTransformParameterFileName "NoInitialTransform")
(HowToCombineTransforms "Compose")
(FixedImageDimension 3)
(MovingImageDimension 3)
(FixedInternalImagePixelType "float")
(MovingInternalImagePixelType "float")
(Size {w} {h} {nz})
(Index 0 0 0)
(Spacing {sp[0]} {sp[1]} {sp[2]})
(Origin {org[0]} {org[1]} {org[2]})
(Direction {dir_str})
(UseDirectionCosines "true")
(CenterOfRotationPoint 0 0 0)
(ResampleInterpolator "FinalBSplineInterpolator")
(FinalBSplineInterpolationOrder 1)
(Resampler "DefaultResampler")
(DefaultPixelValue 0)
(ResultImageFormat "nii")
(ResultImagePixelType "float")
''')

# Rígido 3D: Euler, MSE, pirámide 8-4-2 sin submuestrear Z
pm = itk.ParameterObject.New()
p = pm.GetDefaultParameterMap("rigid")
p["Metric"] = ["AdvancedMeanSquares"]
p["AutomaticTransformInitialization"] = ["false"]   # la inicial viene del archivo
p["AutomaticScalesEstimation"] = ["true"]
p["NumberOfResolutions"] = ["3"]
p["ImagePyramidSchedule"] = ["8", "8", "1", "4", "4", "1", "2", "2", "1"]
p["MaximumNumberOfIterations"] = [str(ITERACIONES)]
p["NumberOfSpatialSamples"] = ["200000"]
p["DefaultPixelValue"] = ["0"]
p["ImageSampler"] = ["Full"]
p["ResultImagePixelType"] = ["float"]
pm.AddParameterMap(p)

alineadas = [ref_np]
for i, (ruta, ang, nro) in enumerate(OTRAS, 1):
    mov = itk.imread(ruta, itk.F)
    assert itk.array_from_image(mov).shape == ref_np.shape, f"Distinta forma: {ruta}"
    mov.CopyInformation(ref)                         # misma geometría que la referencia

    ruta_init = os.path.join(OUT, f"rotacion_inicial_{nro}.txt")
    guardar_rotacion(ang, ruta_init)

    # Rígido desde la rotación inicial; un solo remuestreo con la cadena completa
    reg, tp = itk.elastix_registration_method(
        ref, mov, parameter_object=pm,
        initial_transform_parameter_file_name=ruta_init,
        log_to_console=True)
    alineadas.append(itk.array_from_image(reg).astype(np.float32))

    # Guardar cada suma registrada por separado
    r = itk.image_from_array(alineadas[-1]); r.CopyInformation(ref)
    itk.imwrite(r, os.path.join(OUT, f"suma_registrada_{nro}.nii.gz"))

    po = itk.ParameterObject.New()
    po.AddParameterMap(tp.GetParameterMap(0))
    po.WriteParameterFile(os.path.join(OUT, f"transform_rigido_{nro}.txt"))
    print(f"Suma {nro} ({ang}°) registrada")

# Suma directa (sin normalizar)
atlas = np.sum(alineadas, axis=0).astype(np.float32)
out = itk.image_from_array(atlas)
out.CopyInformation(ref)
itk.imwrite(out, os.path.join(OUT, "atlas_3fixed.nii.gz"))
print("Atlas guardado en", OUT)


#pipeline final
##tomar las sumas con los 3 o 6 fixed distintos y registrarlas 
##luego promediarlas
##una vez que se tiene el ATLAS:
###empezar todo de nuevo pero registrando solo al atlas y no entre sí

#bueno quiero que me armes un archivo de texto .md con todos los detalles necesarios para resolver esto. 
#que describa el pipeline completo con cada codigo con su input/output, paso por paso que se hace. 
#me gustaria que en cada paso exista la opcion de guardar el volumen producto de ese paso, quizas comentando o descomentando una linea. para algo estilo debug. 
#que el archivo especifique si hay que armar algun sciprt secundario con funciones que luego usamos, 
#porque algunos de los scripts actuales no estan con funciones sino todo suelto. 
#sino que aclare que script importar, todos estan en la misma crptea y no te preocupes por los nombres de los archivos, eso lo agrego yo