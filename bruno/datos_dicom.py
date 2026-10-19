import pydicom, numpy as np

def resumen(ruta, vel_z=25.0):
    d = pydicom.dcmread(ruta)
    print("\n###", ruta)
    print("Frames:", getattr(d, "NumberOfFrames", None), "| shape:", d.pixel_array.shape)

    # Equipo y preset
    for t in ["Manufacturer", "ManufacturerModelName", "SeriesDescription", "ProtocolName",
              "TransducerType", "TransducerFrequency", "DepthOfScanField",
              "MechanicalIndex", "ThermalIndex"]:
        print(f"{t}: {getattr(d, t, None)}")

    # Spacing real y región ultrasónica
    for i, r in enumerate(getattr(d, "SequenceOfUltrasoundRegions", [])):
        print(f"Región {i}: DeltaX={getattr(r,'PhysicalDeltaX',None)}, "
              f"DeltaY={getattr(r,'PhysicalDeltaY',None)}, "
              f"unidades X/Y={getattr(r,'PhysicalUnitsXDirection',None)}/"
              f"{getattr(r,'PhysicalUnitsYDirection',None)}, "
              f"x=[{getattr(r,'RegionLocationMinX0',None)},{getattr(r,'RegionLocationMaxX1',None)}], "
              f"y=[{getattr(r,'RegionLocationMinY0',None)},{getattr(r,'RegionLocationMaxY1',None)}], "
              f"transductor={getattr(r,'TransducerFrequency',None)}")

    # Z: tiempo entre frames y su variabilidad
    ft = np.array(d.FrameTimeVector, float)
    ft = ft[ft > 0]
    print(f"FrameTime (ms): media={ft.mean():.3f}, std={ft.std():.3f}, "
          f"min={ft.min():.3f}, max={ft.max():.3f}, n={ft.size}")
    print(f"mm_por_frame_z: {vel_z * ft.mean() / 1000:.6f}")

    # Volcado completo (incluye tags privados, donde puede estar el preset o la ganancia)
    print("--- Todos los tags ---")
    for el in d:
        if el.tag != (0x7FE0, 0x0010):
            print(el.tag, el.name, "=", str(el.value)[:80])

for ruta in ["data/Escaneo 22-7/Ocular/001.dcm", "data/Escaneo 22-7/Suerficial/001.dcm", "data/Escaneo 22-7/Vascular/001.dcm"]:
    resumen(ruta)
