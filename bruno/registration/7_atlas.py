import itk
import numpy as np
import cv2

# --- Parámetros ---
RUTA_REF = "bruno/volumenes/superficial/final/2fixed/suma_2fixed.nii.gz"
OTRAS = [("bruno/volumenes/superficial/final/3fixed/suma_3fixed.nii.gz", 60),
            ]  # ángulo horario, como en el ICP
OUT_PROMEDIO = "bruno/volumenes/superficial/final/atlas.nii.gz"
ITERACIONES = 500   # confirmar
# -----------------------------

ref = itk.imread(RUTA_REF, itk.F)
nz, h, w = itk.array_from_image(ref).shape
sp = np.array(itk.spacing(ref)); org = np.array(itk.origin(ref))
D = itk.array_from_matrix(ref.GetDirection())

# Índice -> mm
G = np.eye(4); G[:3, :3] = D @ np.diag(sp); G[:3, 3] = org

def guardar_rotacion(angulo, ruta):
    # Rotación inicial (cv2, alrededor del centro de la grilla) como AffineTransform fixed(mm) -> moving(mm)
    M = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), -angulo, 1.0)
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

# Parámetros del rígido: Euler 3D, MSE, pirámide 8-4-2 sin submuestrear Z
pm = itk.ParameterObject.New()
rigido = pm.GetDefaultParameterMap("rigid")
rigido["Metric"] = ["AdvancedMeanSquares"]
rigido["NumberOfResolutions"] = ["3"]
rigido["ImagePyramidSchedule"] = ["8", "8", "1", "4", "4", "1", "2", "2", "1"]
rigido["MaximumNumberOfIterations"] = [str(ITERACIONES)]
rigido["NumberOfSpatialSamples"] = ["200000"]
rigido["ResultImagePixelType"] = ["float"]
#p["ImageSampler"] = ["Full"] # mas costoso
pm.AddParameterMap(rigido)

alineadas = [itk.array_from_image(ref).astype(np.float32)]

for i, (ruta, ang) in enumerate(OTRAS):
    mov = itk.imread(ruta, itk.F)
    ruta_init = f"transform_inicial_{i}.txt"
    guardar_rotacion(ang, ruta_init)

    # Rígido partiendo de la rotación inicial
    _, tp = itk.elastix_registration_method(
        ref, mov, parameter_object=pm,
        initial_transform_parameter_file_name=ruta_init,
        log_to_console=True)

    # Un solo remuestreo con la cadena completa (rotación + rígido)
    reg = itk.transformix_filter(mov, tp)
    alineadas.append(itk.array_from_image(reg).astype(np.float32))

suma_de_fixeds = np.sum(alineadas, axis=0).astype(np.float32)
out = itk.image_from_array(suma_de_fixeds)
out.CopyInformation(ref)
itk.imwrite(out, OUT_PROMEDIO)
print("Promedio guardado en", OUT_PROMEDIO)

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