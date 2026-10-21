import numpy as np
import vtk
import itk
from vtk.util import numpy_support

# --- Parámetros ---
RUTA_STL = "bruno/raiz_recortada.stl"
SPACING = (0.156661, 0.156661, 1.318305)   # (x, y, z) mm, escaneo Superficial 001
GRILLA_ESCANEO = (220, 575, 188)           # (x, y, z) vóxeles del recorte
FLIP_Z = False                             # True si la tapa quedó abajo respecto del escaneo
RUTA_COMPLETO = "bruno/volumenes/superficial/stl_volumen_completo.nii.gz"
RUTA_EN_GRILLA = "bruno/volumenes/superficial/stl_volumen_grilla_escaneo.nii.gz"

# --- Leer STL ---
reader = vtk.vtkSTLReader()
reader.SetFileName(RUTA_STL)
reader.Update()
poly = reader.GetOutput()
b = poly.GetBounds()

# --- Grilla que cubre todo el STL con el spacing del escaneo ---
dims = [int(np.ceil((b[2 * i + 1] - b[2 * i]) / SPACING[i])) + 1 for i in range(3)]
img = vtk.vtkImageData()
img.SetSpacing(SPACING)
img.SetDimensions(dims)
img.SetOrigin(b[0], b[2], b[4])
img.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
img.GetPointData().GetScalars().Fill(1)

# --- Voxelizar (interior = 1) ---
p2s = vtk.vtkPolyDataToImageStencil()
p2s.SetInputData(poly)
p2s.SetOutputOrigin(img.GetOrigin())
p2s.SetOutputSpacing(img.GetSpacing())
p2s.SetOutputWholeExtent(img.GetExtent())
p2s.Update()

st = vtk.vtkImageStencil()
st.SetInputData(img)
st.SetStencilConnection(p2s.GetOutputPort())
st.ReverseStencilOff()
st.SetBackgroundValue(0)
st.Update()

arr = numpy_support.vtk_to_numpy(st.GetOutput().GetPointData().GetScalars())
arr = arr.reshape(dims[2], dims[1], dims[0]).astype(np.uint8)   # (z, y, x)
if FLIP_Z:
    arr = arr[::-1]
print("Volumen STL (z, y, x):", arr.shape, "| vóxeles = 1:", int(arr.sum()))


def centro(shape_zyx, spacing_xyz):
    z, y, x = shape_zyx
    return np.array([(x - 1) * spacing_xyz[0], (y - 1) * spacing_xyz[1],
                     (z - 1) * spacing_xyz[2]]) / 2.0


# --- 1) Volumen completo, con el mismo centro que la grilla del escaneo ---
shape_esc = GRILLA_ESCANEO[::-1]                       # (z, y, x)
origen = centro(shape_esc, SPACING) - centro(arr.shape, SPACING)
vol = itk.image_from_array(arr)
vol.SetSpacing(SPACING)
vol.SetOrigin(tuple(float(o) for o in origen))
itk.imwrite(vol, RUTA_COMPLETO)

# --- 2) Mismo volumen en la grilla exacta del escaneo (centrado; lo que sobra se recorta) ---
out = np.zeros(shape_esc, dtype=np.uint8)
src, dst = [], []
for n_src, n_dst in zip(arr.shape, shape_esc):
    n = min(n_src, n_dst)
    s0 = (n_src - n) // 2
    d0 = (n_dst - n) // 2
    src.append(slice(s0, s0 + n))
    dst.append(slice(d0, d0 + n))
out[tuple(dst)] = arr[tuple(src)]
vol2 = itk.image_from_array(out)
vol2.SetSpacing(SPACING)
itk.imwrite(vol2, RUTA_EN_GRILLA)
print("Listo:", RUTA_COMPLETO, "y", RUTA_EN_GRILLA)


