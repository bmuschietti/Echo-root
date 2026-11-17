import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.morphology import white_tophat
from marcar_centroides import cargar_volumen

LINE_LENGTH = 25   # top-hat
PERC = 99.5          #top-hat
MIN_AREA = 8     # px; descarta blobs más chicos (valor a ajustar)
TOL = 30        # px; tolerancia para considerar que dos puntos coinciden (a ajustar)
MIN_PARES = 3      # mínimo de pares para aceptar la estimación de un frame


def detectar_centroides(frame, line_length=LINE_LENGTH, perc=PERC, min_area=MIN_AREA):
    """Top-hat + umbral por percentil + componentes conexas. Devuelve array (n, 2) de (x, y)."""
    img = frame.astype(np.float32)
    footprint = np.ones((1, line_length), dtype=bool)
    response = white_tophat(img, footprint=footprint)
    mask = response >= np.percentile(response, perc)

    etiquetas, n = ndi.label(mask)
    if n == 0:
        return np.empty((0, 2))
    idx = np.arange(1, n + 1)
    areas = ndi.sum(mask, etiquetas, idx)
    idx = idx[areas >= min_area]
    if len(idx) == 0:
        return np.empty((0, 2))
    centros = np.array(ndi.center_of_mass(mask, etiquetas, idx))  # (y, x)

    return centros[:, ::-1]  # (x, y)


def emparejar(pts_a, pts_b, tol=TOL):
    """
    Prueba cada diferencia (b_j - a_i) como traslación candidata y se queda con la
    que deja más puntos de A cerca de algún punto de B (desempate: menor distancia media).
    Devuelve (idx_a, idx_b) de los pares aceptados.
    """
    if len(pts_a) == 0 or len(pts_b) == 0:
        return np.array([], int), np.array([], int)

    arbol_b = cKDTree(pts_b)
    mejor = None
    for pa in pts_a:
        for pb in pts_b:
            t = pb - pa
            dist, idx = arbol_b.query(pts_a + t)
            ok = dist <= tol
            n = int(ok.sum())
            if n == 0:
                continue
            clave = (n, -dist[ok].mean())
            if mejor is None or clave > mejor[0]:
                mejor = (clave, ok, idx, dist)

    if mejor is None:
        return np.array([], int), np.array([], int)

    _, ok, idx, dist = mejor
    idx_a = np.where(ok)[0]
    idx_b = idx[ok]

    # si dos puntos de A caen en el mismo punto de B, se conserva el más cercano
    mantener = []
    for b in np.unique(idx_b):
        candidatos = idx_a[idx_b == b]
        mantener.append(candidatos[np.argmin(dist[candidatos])])
    mantener = np.array(sorted(mantener))
    idx_a_final = mantener
    idx_b_final = idx[mantener]
    return idx_a_final, idx_b_final


def estimar_traslacion(pts_a, pts_b):
    """Traslación por mínimos cuadrados (= media de las diferencias). dx = x_B - x_A."""
    diffs = pts_b - pts_a
    t = diffs.mean(axis=0)
    residuos = np.linalg.norm(diffs - t, axis=1)
    return t[0], t[1], residuos


def aplicar_traslacion(frame, dx, dy):
    """Lleva B sobre A desplazando (-dx, -dy)."""
    return ndi.shift(frame, (-dy, -dx), order=1, mode="constant", cval=0)


def registrar_frame(frame_a, frame_b_rotado, tol=TOL, min_pares=MIN_PARES):
    pts_a = detectar_centroides(frame_a)
    pts_b = detectar_centroides(frame_b_rotado)
    idx_a, idx_b = emparejar(pts_a, pts_b, tol)

    info = {"pts_a": pts_a, "pts_b": pts_b, "idx_a": idx_a, "idx_b": idx_b}
    if len(idx_a) < min_pares:
        info.update(dx=None, dy=None, residuos=None, alineado=None)
        return info

    dx, dy, residuos = estimar_traslacion(pts_a[idx_a], pts_b[idx_b])
    info.update(
        dx=dx, dy=dy, residuos=residuos,
        alineado=aplicar_traslacion(frame_b_rotado, dx, dy),
    )
    return info


def registrar_volumen_por_frame(vol_a, vol_b_rotado, lista_z=None, tol=TOL):
    if lista_z is None:
        lista_z = range(vol_a.shape[0])
    return {z: registrar_frame(vol_a[z], vol_b_rotado[z], tol=tol) for z in lista_z}


def resumen(resultados):
    validos = {z: r for z, r in resultados.items() if r["dx"] is not None}
    for z, r in resultados.items():
        if r["dx"] is None:
            print(f"z={z}: sin estimación (A={len(r['pts_a'])} blobs, "
                  f"B={len(r['pts_b'])} blobs, pares={len(r['idx_a'])})")
        else:
            print(f"z={z}: dx={r['dx']:.1f}, dy={r['dy']:.1f} px | pares={len(r['idx_a'])} "
                  f"| residuo medio={r['residuos'].mean():.1f}, max={r['residuos'].max():.1f} px")
    if validos:
        dxs = np.array([r["dx"] for r in validos.values()])
        dys = np.array([r["dy"] for r in validos.values()])
        print(f"\nFrames válidos: {len(validos)}/{len(resultados)}")
        print(f"dx: mediana={np.median(dxs):.1f}, std={dxs.std():.1f} px")
        print(f"dy: mediana={np.median(dys):.1f}, std={dys.std():.1f} px")


def _norm(img):
    img = img.astype(np.float32)
    return (img - img.min()) / (img.max() - img.min() + 1e-8)


def _overlay(a, b):
    rgb = np.zeros((*a.shape, 3), dtype=np.float32)
    rgb[..., 0] = _norm(a)
    rgb[..., 1] = _norm(b)
    return rgb


def mostrar(vol_a, vol_b_rotado, resultados):
    """Por cada z: overlay antes, puntos detectados/emparejados, overlay después."""
    for z, r in resultados.items():
        fig, axes = plt.subplots(1, 3, figsize=(15, 8))
        axes[0].imshow(_overlay(vol_a[z], vol_b_rotado[z]))
        axes[0].set_title(f"Antes (z={z})")

        axes[1].imshow(_overlay(vol_a[z], vol_b_rotado[z]))
        axes[1].plot(*r["pts_a"].T, "wo", mfc="none", ms=10, label="A")
        axes[1].plot(*r["pts_b"].T, "c^", mfc="none", ms=10, label="B")
        for ia, ib in zip(r["idx_a"], r["idx_b"]):
            axes[1].plot(*zip(r["pts_a"][ia], r["pts_b"][ib]), "y-", lw=1)
        axes[1].legend()
        axes[1].set_title(f"Puntos y pares (z={z})")

        if r["alineado"] is not None:
            axes[2].imshow(_overlay(vol_a[z], r["alineado"]))
            axes[2].set_title(f"Después: dx={r['dx']:.1f}, dy={r['dy']:.1f}")
        else:
            axes[2].imshow(_overlay(vol_a[z], vol_b_rotado[z]))
            axes[2].set_title("Sin estimación")
        plt.tight_layout()
        plt.show()


# Uso:
#   res = registrar_volumen_por_frame(vol_a, vol_b_rotado, lista_z=[50, 100, 150, 200])
#   resumen(res)
#   mostrar(vol_a, vol_b_rotado, res)

if __name__ == "__main__":
    vol_a = cargar_volumen("data/ultrasound/primera_medicion/001.dcm")
    vol_b = cargar_volumen("data/ultrasound/primera_medicion/004.dcm")
    vol_b_rotado = vol_b[:, ::-1, ::-1]

    lista_z = [120]
    res = registrar_volumen_por_frame(vol_a, vol_b_rotado, lista_z=lista_z)

    resumen(res)
    mostrar(vol_a, vol_b_rotado, res)
