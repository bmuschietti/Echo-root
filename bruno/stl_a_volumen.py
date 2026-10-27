import numpy as np
import vtk
import itk
from vtk.util import numpy_support

# --- Parámetros ---
RUTA_STL = "bruno/volumenes/superficial/raiz_recortada.stl"
SPACING = (0.156661, 0.156661, 1.318305)   # (x, y, z) mm, escaneo Superficial 001
GRILLA_ESCANEO = (680, 681, 187)           # (x, y, z) vóxeles del recorte
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


RUTA_SUMA = "bruno/volumenes/superficial/final/suma_final.nii.gz"

ref = itk.imread(RUTA_SUMA, itk.F)
shape_ref = itk.array_from_image(ref).shape        # (z, y, x)

out = np.zeros(shape_ref, dtype=np.uint8)

# Z: alineado al último frame (techo). XY: centrado.
nz_s, ny_s, nx_s = shape_ref
nz_a, ny_a, nx_a = arr.shape

nz = min(nz_a, nz_s)
ny = min(ny_a, ny_s)
nx = min(nx_a, nx_s)

# Z: se conservan los últimos frames del STL y se ponen al final de la suma
sz_src = slice(nz_a - nz, nz_a)
sz_dst = slice(nz_s - nz, nz_s)
# XY: centrado
sy_src = slice((ny_a - ny) // 2, (ny_a - ny) // 2 + ny)
sy_dst = slice((ny_s - ny) // 2, (ny_s - ny) // 2 + ny)
sx_src = slice((nx_a - nx) // 2, (nx_a - nx) // 2 + nx)
sx_dst = slice((nx_s - nx) // 2, (nx_s - nx) // 2 + nx)

out[sz_dst, sy_dst, sx_dst] = arr[sz_src, sy_src, sx_src]

vol = itk.image_from_array(out)
vol.CopyInformation(ref)                            # origen, spacing y dirección de la suma
itk.imwrite(vol, RUTA_EN_GRILLA)


