import numpy as np
import itk  # pip install itk-elastix
from marcar_centroides import cargar_volumen
import numpy as np
import matplotlib.pyplot as plt


def norm(img):
    img = img.astype(np.float32)
    return (img - img.min()) / (img.max() - img.min() + 1e-8)

def overlay(a, b):
    # A en rojo, B en verde, amarillo donde coinciden
    rgb = np.zeros((*a.shape, 3), dtype=np.float32)
    rgb[..., 0] = norm(a)
    rgb[..., 1] = norm(b)
    return rgb

def registrar_traslacion_2d(frame_fijo, frame_movil, y_min=100, y_max=500):
    """
    Registra frame_movil contra frame_fijo con una traslación (2D).
    La métrica se calcula solo en las filas [y_min, y_max) del frame fijo,
    en todo el ancho. Devuelve (tx, ty, imagen_alineada).
    """
    fijo = itk.GetImageFromArray(frame_fijo.astype(np.float32))
    movil = itk.GetImageFromArray(frame_movil.astype(np.float32))

    mascara = np.zeros(frame_fijo.shape, dtype=np.uint8)
    mascara[y_min:y_max, :] = 1
    mascara_itk = itk.GetImageFromArray(mascara)

    params = itk.ParameterObject.New()
    params.AddParameterMap(params.GetDefaultParameterMap("translation"))

    resultado, params_res = itk.elastix_registration_method(
        fijo, movil,
        parameter_object=params,
        fixed_mask=mascara_itk,
        log_to_console=False,
    )
    tx, ty = (float(v) for v in params_res.GetParameterMap(0)["TransformParameters"])
    imagen_alineada = itk.GetArrayFromImage(resultado)
    return tx, ty, imagen_alineada


def registrar_volumen_por_frame(vol_a, vol_b_rotado, lista_z=None):
    """Devuelve (dict z -> (tx, ty), dict z -> imagen alineada)."""
    if lista_z is None:
        lista_z = range(vol_a.shape[0])

    resultados = {}
    alineados = {}
    for z in lista_z:
        try:
            tx, ty, img = registrar_traslacion_2d(vol_a[z], vol_b_rotado[z])
            resultados[z] = (tx, ty)
            alineados[z] = img
        except Exception as e:
            print(f"z={z}: falló el registro ({e})")
    return resultados, alineados


def resumen(resultados):
    arr = np.array(list(resultados.values()))
    print(f"n frames: {len(arr)}")
    print(f"tx: media={arr[:, 0].mean():.1f}, std={arr[:, 0].std():.1f} px")
    print(f"ty: media={arr[:, 1].mean():.1f}, std={arr[:, 1].std():.1f} px")


# Uso:
#   res = registrar_volumen_por_frame(vol_a, vol_b_rotado, lista_z=[50, 100, 150, 200])
#   for z, (tx, ty) in res.items(): print(z, round(tx, 1), round(ty, 1))
#   resumen(res)

if __name__ == "__main__":
    
    vol_a = cargar_volumen("data/ultrasound/primera_medicion/001.dcm")
    vol_b = cargar_volumen("data/ultrasound/primera_medicion/004.dcm")
    vol_b_rotado = vol_b[:, ::-1, ::-1]

    res, alineados = registrar_volumen_por_frame(vol_a, vol_b_rotado, lista_z=[100])
    for z, (tx, ty) in res.items(): print(z, round(tx, 1), round(ty, 1))
    resumen(res)

    for z, (tx, ty) in res.items():
        print(f"z={z}: tx={tx:.1f}, ty={ty:.1f} px")


    for z in res:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 9))
        ax1.imshow(overlay(vol_a[z], vol_b_rotado[z]))
        ax1.set_title(f"Antes (z={z})")
        ax2.imshow(overlay(vol_a[z], alineados[z]))
        ax2.set_title(f"Después (z={z})")
        plt.tight_layout()
        plt.show()
