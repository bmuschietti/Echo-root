# Resumen — Fusión de volúmenes echoroot (maceta hexagonal)

## Objetivo
Fusionar los 6 volúmenes de ultrasonido (uno por cara de la maceta hexagonal)
en un único volumen 3D, usando la transformación geométrica conocida
(rotación 60° + apotema), con la hipótesis de que la raíz aparece consistente
en las 6 vistas mientras que la reverberación no, permitiendo filtrarla en el
proceso de fusión.

## Datos físicos confirmados
- `mm/pixel` en x e y: **0.121847 mm/pixel** (tag DICOM `PhysicalDeltaX/Y`)
- Lado del hexágono (interior): **41 mm** (plano de la maceta)
- Apotema calculada: **≈35.5 mm** `(lado/2)*√3`
- Rotación entre caras: **60° exactos**, calibración mecánica confiable
- Velocidad del carro vertical: **25 mm/s**
- `mm/frame` en z: calculado dinámicamente a partir de `FrameTimeVector` del
  DICOM (≈1.05 mm/frame con ~42ms entre frames)
- Recorte real de máscara (elimina texto/UI del equipo):
  `CROP_FILA_MIN, CROP_FILA_MAX = 113, 694`
  `CROP_COL_MIN, CROP_COL_MAX = 365, 652`
  → profundidad recortada ≈70.8mm (coincide con 2×apotema≈71mm)
  → ancho lateral recortado ≈35.0mm (menor a los 41mm de la cara completa)

## Lo que se probó (fusión de las 6 caras)
1. **Transformación geométrica** (`coords_globales_a_locales`): rotación
   inversa por cara + traslación por apotema, para llevar cada volumen a
   coordenadas globales comunes.
2. **Bug encontrado y corregido**: faltaba centrar el eje lateral
   (`x_local = X_rot + ancho_frame/2`); sin esto, las 6 caras quedaban
   desplazadas y superpuestas en vez de encajar.
3. **Métodos de combinación probados**: promedio, mediana, mediana + factor
   de atenuación por cobertura (`cantidad_vistas_validas / 6`).
   - La mediana redujo notablemente las líneas de reverb en zonas de alta
     cobertura (vistas por varias caras).
   - Persistía reverb sin filtrar en zonas vistas por 1 sola cara (ahí no
     hay consenso posible entre vistas).
4. **Manchas centrales**: confirmadas visualmente como correspondientes a
   la posición real de las raíces sintéticas del phantom.
5. **Huecos en los vértices** del hexágono fusionado: esperados, porque el
   transductor no cubre el ancho completo de la cara (35mm de 41mm).

## Nueva hipótesis: pares de caras opuestas
- Idea: la cola de reverberación de una cara corre en sentido opuesto a la
  de su cara opuesta (mismo eje, direcciones inversas); fusionar solo el
  par podría cancelar mejor el artefacto.
- Como cada volumen cubre casi exactamente la distancia entre caras
  opuestas (~71mm ≈ 2×apotema), se simplificó el método: en vez de
  resamplear a grilla global, alcanza con **rotar 180° uno de los dos
  volúmenes** (flip de filas y columnas) para que ambos queden mirando el
  mismo espacio físico.
- Se armó `fusion_pares_opuestos.py` (script aparte, no modifica el
  pipeline principal) con:
  - `fusionar_par_opuesto(vol_a, vol_b, umbral_intensidad)`: promedia A y
    B-rotado, pero descarta (pone en 0) todo voxel donde no *ambos*
    superen `umbral_intensidad` (parámetro a calibrar a mano).
  - `ver_corte_axial(vol_a, vol_b_rotado, fusionado, z, titulo)`: grafica
    lado a lado el mismo corte de A, B-rotado y el resultado fusionado.

## Problema actual (sin resolver)
Al comparar visualmente A vs. B-rotado (mismo z), **las estructuras
(picos/manchas) no coinciden en posición x** entre ambos — más allá de que
las colas de reverb sí apuntan en sentidos opuestos (esperado por la
hipótesis). Si la alineación fuera correcta, algo real (la raíz) debería
aparecer en la misma posición x en ambas vistas.

Dos causas candidatas, sin confirmar todavía:
1. **Orden de frames en z no verificado entre archivos distintos** — no se
   confirmó que el frame índice N sea la misma altura física real en los
   6 archivos DICOM (podría haber inversión de orden en algún escaneo).
2. **Transductor no centrado lateralmente en la cara** (asunción usada
   desde el principio, nunca validada) — generaría un corrimiento lateral
   sistemático que el flip de 180° no compensa.

## Pendientes para seguir
- [ ] Verificar orden temporal/físico de los frames entre los 6 archivos
      (¿frame 0 = misma altura real en todos?).
- [ ] Confirmar o descartar si el transductor está centrado lateralmente
      en cada cara.
- [ ] Una vez resuelta la alineación, recalibrar `umbral_intensidad` en
      `fusionar_par_opuesto` con datos reales.
- [ ] Definir si el enfoque final será: fusión directa de 6, fusión de 3
      pares opuestos + combinación final, u otro método (no descartado:
      consenso por varianza, fusión multiplicativa).
- [ ] Pendiente de terceros: confirmar si hay error en la calibración de
      rotación mecánica (se asumió exacta, 60° sin margen de error).
- [ ] Segmentación real (Otsu/Frangi) queda pospuesta hasta tener un
      volumen fusionado confiable — no se avanzó en eso todavía.
