import itk
import numpy as np
import cv2
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.ndimage import affine_transform

# --- Parámetros (completar) ---
RUTA_FIXED_SEG  = "bruno/volumenes/superficial/registro2_5/producto_seg.nii"
RUTA_MOVING_SEG = "bruno/volumenes/superficial/registro1_4/producto_penal_seg.nii"
RUTA_SALIDA     = "bruno/volumenes/superficial/final/seg"
ANGULO_HORARIO  = -60      # el mismo valor que usaste en el registro del 1-4
UMBRAL_SEG      = 0.5      # el _seg está suavizado: vóxeles > umbral cuentan como raíz
MAX_DIST_MM     = 30       # distancia máxima de correspondencia del ICP
N_MAX           = 200000   # máximo de puntos por nube (submuestreo aleatorio)
# -----------------------------
rng = np.random.default_rng(0)

ref = itk.imread(RUTA_FIXED_SEG, itk.F)
fixed_np  = itk.array_from_image(ref)                                   # (z, y, x)
moving_np = itk.array_from_image(itk.imread(RUTA_MOVING_SEG, itk.F))
n = min(fixed_np.shape[0], moving_np.shape[0])                          # mismo criterio de frames
fixed_np, moving_np = fixed_np[-n:], moving_np[-n:]

nz, h, w = fixed_np.shape
sp  = np.array(itk.spacing(ref))
org = np.array(itk.origin(ref))
D   = itk.array_from_matrix(ref.GetDirection())

# Rotación inicial: misma matriz de cv2 que en el script 2D (moving -> fixed, en píxeles)
M = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), -ANGULO_HORARIO, 1.0)
M3 = np.vstack([M, [0, 0, 1]])
Minv = np.linalg.inv(M3)

# Banda de coincidencia en el fixed
xs, ys = np.meshgrid(np.arange(w), np.arange(h))
mx = Minv[0, 0] * xs + Minv[0, 1] * ys + Minv[0, 2]
my = Minv[1, 0] * xs + Minv[1, 1] * ys + Minv[1, 2]
banda = (mx >= -0.5) & (mx <= w - 0.5) & (my >= -0.5) & (my <= h - 0.5)

def a_mm(x, y, z):
    idx = np.stack([x, y, z], axis=1).astype(np.float64)
    return org + (D @ (idx * sp).T).T

def submuestrear(p):
    if len(p) > N_MAX:
        p = p[rng.choice(len(p), N_MAX, replace=False)]
    return p

# --- Puntos del fixed (dentro de la banda) ---
zf, yf, xf = np.nonzero(fixed_np > UMBRAL_SEG)
dentro = banda[yf, xf]
P_fixed = submuestrear(a_mm(xf[dentro], yf[dentro], zf[dentro]))

# --- Puntos del moving: rotación inicial en píxeles, luego a mm ---
zm, ym, xm = np.nonzero(moving_np > UMBRAL_SEG)
xr = M[0, 0] * xm + M[0, 1] * ym + M[0, 2]
yr = M[1, 0] * xm + M[1, 1] * ym + M[1, 2]
dentro = (xr >= -0.5) & (xr <= w - 0.5) & (yr >= -0.5) & (yr <= h - 0.5)
P_moving = submuestrear(a_mm(xr[dentro], yr[dentro], zm[dentro]))

print("Puntos fixed:", len(P_fixed), "| moving:", len(P_moving))

# --- Distancia media al vecino más cercano (antes del ICP) ---
tree = cKDTree(P_fixed)
d_antes = tree.query(P_moving)[0].mean()

# --- ICP rígido (6 grados de libertad) ---
src = o3d.geometry.PointCloud(); src.points = o3d.utility.Vector3dVector(P_moving)
tgt = o3d.geometry.PointCloud(); tgt.points = o3d.utility.Vector3dVector(P_fixed)
res = o3d.pipelines.registration.registration_icp(
    src, tgt, MAX_DIST_MM, np.eye(4),
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=500))
T = res.transformation

P_final = (T[:3, :3] @ P_moving.T).T + T[:3, 3]
d_despues = tree.query(P_final)[0].mean()

rx, ry, rz = Rotation.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True)
print(f"Distancia media: antes {d_antes:.2f} mm -> después {d_despues:.2f} mm")
print(f"fitness (fracción de puntos con pareja): {res.fitness:.3f} | rmse: {res.inlier_rmse:.3f} mm")
print(f"Rotación ICP (rx, ry, rz) en grados: {rx:.2f}, {ry:.2f}, {rz:.2f}")
print(f"Traslación ICP (tx, ty, tz) en mm: {T[0, 3]:.2f}, {T[1, 3]:.2f}, {T[2, 3]:.2f}")

np.savez(RUTA_SALIDA, T_icp_mm=T, M_rotacion_inicial_px=M, angulo_horario=ANGULO_HORARIO)

#CREA Y GUARDA EL VOLUMEN REGISTRADO 

RUTA_SALIDA_VOL = "bruno/volumenes/superficial/final/seg/1seg_registrado_icp.nii.gz"

# Índice (x, y, z) -> mm
G = np.eye(4)
G[:3, :3] = D @ np.diag(sp)
G[:3, 3] = org

# Rotación inicial en píxeles (moving -> fixed), como matriz 4x4 en índices (x, y, z)
Mi = np.eye(4)
Mi[:2, :2] = M[:, :2]
Mi[:2, 3] = M[:, 2]

# Total moving(idx) -> fixed(mm) = T · G · Mi ; se necesita fixed(idx) -> moving(idx)
A = np.linalg.inv(Mi) @ np.linalg.inv(G) @ np.linalg.inv(T) @ G

# affine_transform trabaja con (z, y, x): se permutan los ejes
P = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=float)
mat = P @ A[:3, :3] @ P.T
off = P @ A[:3, 3]

reg = affine_transform(moving_np, mat, offset=off, output_shape=fixed_np.shape,
                       order=1, mode="constant", cval=0.0).astype(np.float32)

out = itk.image_from_array(reg)
out.CopyInformation(ref)
itk.imwrite(out, RUTA_SALIDA_VOL)
print("Volumen registrado guardado en", RUTA_SALIDA_VOL)