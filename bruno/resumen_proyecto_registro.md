# Echo Root: resumen del proyecto, registro 3D y tarea actual

## 1. Condiciones de trabajo

- Trabajo de investigación (doctorado en ciencia de datos). Tono objetivo, sin emoción.
- Respuestas cortas y precisas; extenderse solo si se pide.
- Preguntar ante dudas, no asumir.
- No generar código ni tomar decisiones por adelantado sin que se pida.

## 2. Información inicial del proyecto

**Sistema:** Echo Root. Máquina que adquiere ultrasonido de raíces dentro de una maceta. Genera un archivo `.dcm` (DICOM) por escaneo.

**Idea general:** tomar 6 mediciones por planta (rotaciones de 60°) y crear con ellas un único volumen.

**Problema vigente:** reverberación y ruido en el ultrasonido que afectan el análisis. El objetivo es llegar a un procesamiento más limpio.

### Datos de los DICOM

- Shape: `(236, 768, 1024, 3)`; los 3 canales son idénticos (se usa un solo canal).
- `MM_PER_PIXEL_XY = 0.121847` (confirmado, PhysicalDeltaX/Y).
- `VELOCIDAD_Z_MM_S = 25.0` (velocidad del carro vertical, confirmada).
- `MM_PER_FRAME_Z = 1.049983510638298` (calculado con un volumen a partir de `FrameTimeVector`; en general se calcula por volumen).
- `LADO_HEXAGONO_MM = 41.0` (medida interior, plano maceta); apotema ≈ 35.5 mm.
- `N_CARAS = 6`, `ROTACION_GRADOS = 60`.
- Recorte: filas 113–694, columnas 365–652 (excluye texto/metadata del frame).
- Tamaño del recorte: ≈ 35.0 mm (X) × 70.8 mm (Y) × 248 mm (Z).

### Estructuras en las imágenes

1. **Bordes de la maceta:** líneas muy intensas arriba y abajo, con algo de reverberación.
2. **Raíces:** elipses (corte transversal), con colas largas hacia la pared opuesta al ecógrafo. Las colas son reverberación de ondas que rebotan en la raíz. A veces las elipses se ven más como arcos.
3. **Reverberaciones locales:** regiones con varias líneas horizontales brillantes, aproximadamente periódicas en profundidad, locales, en distintas posiciones y a veces superpuestas con raíces. Son las que se quieren eliminar.

### Trabajo previo

Filtrado de reverberación sobre frames por separado.

## 3. Registro de volúmenes opuestos

**Idea:** el volumen de una cara y el de la cara opuesta cubren el mismo espacio físico. Rotando uno 180° en XY deberían verse iguales, pero están algo desfasados. Se pidió (director) registro 3D basado en intensidad (itk-elastix), con transformación rígida (traslación + rotación) y una métrica tipo error cuadrático medio.

### Preprocesamiento

- Orden correcto: **recortar ambos volúmenes con los mismos índices y luego hacer el flip 180° del móvil**:
  ```python
  fixed_np  = recortar(vol_a)
  moving_np = recortar(vol_b)[:, ::-1, ::-1]
  ```
- El orden inverso (flip y luego recorte) daba una región parcialmente equivocada, porque la imagen útil no está centrada en el frame (768×1024).
- Spacing ITK (x, y, z): `(0.121847, 0.121847, 1.04998)` mm.

### Configuración Elastix (baseline)

- Mapa `rigid` (EulerTransform 3D, implícito en `GetDefaultParameterMap("rigid")`).
- Métrica `AdvancedMeanSquares`, sin máscara.
- Inicialización por centro geométrico, escalas automáticas.
- 3 resoluciones, `ImagePyramidSchedule = 8 8 1, 4 4 1, 2 2 1` (Z no se submuestrea).
- `MaximumNumberOfIterations = 500`; sampler `Full` en las últimas corridas (`NumberOfSpatialSamples = 50000` se ignora con `Full`).
- Salidas: `fixed.nii.gz`, `moving_original.nii.gz`, `moving_registrado.nii.gz`, parámetros de transformación.

### Resultados (par 001 vs 004)

| | Flip → recorte (incorrecto) | Recorte → flip (correcto) |
|---|---|---|
| MSE antes | 1995.2 | 1450.0 |
| MSE después | 1822.0 | 1070.0 |
| Reducción | ~8.7% | ~26.2% |
| rx, ry, rz (°) | 1.04, 0.82, 0.16 | -0.14, 0.40, 0.72 |
| tx, ty, tz (mm) | 0.49, 2.80, -0.02 | 2.19, 0.79, 0.006 |

El MSE "después" incluye relleno con 0 fuera del solapamiento, por lo que está algo inflado.

### Observaciones

- El registro acercó las elipses, pero en frames con colas largas tiende a alinear una elipse con la cola de la otra (las colas van en sentidos opuestos entre caras). El objetivo actual tolera esto: si la cola de una coincide con la elipse de la otra, el producto no elimina ninguna de las dos.
- El optimizador termina por máximo de iteraciones; la métrica oscila alrededor de un valor sin tendencia clara. La causa de la oscilación no está determinada.
- Si las elipses son arcos, cada cara ve el borde cercano a su transductor, y los volúmenes no se superponen en intensidad aun con alineación perfecta (hipótesis, no verificada).
- Pocos casos: elipses que no coinciden y casi sin cola. Un caso medido: desfase ≈ (10, 40) px en (x, y) ≈ (1.2, 4.9) mm, principalmente en profundidad. Identificado como **desfase residual local**.
- Pendiente: medir el desfase en otras 2–3 raíces y ver si el signo en Y es sistemático (arcos) o variable (desfase real).
- Se probará MSE; NCC queda como control opcional. Si ambas dan transformaciones similares, la métrica no sería el factor limitante.

### B-spline (en prueba)

Segunda etapa encadenada tras el rígido, para corregir desfases locales.

Parámetros de partida (no verificados):

| Parámetro | Valor |
|---|---|
| Métricas | `AdvancedMeanSquares` + `TransformBendingEnergyPenalty` |
| `FinalGridSpacingInPhysicalUnits` | 10, 10, 20 mm |
| `GridSpacingSchedule` | 4.0, 2.0, 1.0 (corrige error "Invalid GridSpacingSchedule") |
| `Metric1Weight` | 10 (probar 1, 10, 100) |
| `ImageSampler` | `RandomCoordinate` o `Full` (abierto) |
| `NumberOfSpatialSamples` | 200000 |
| `MaximumNumberOfIterations` | 500 |

Riesgos: la deformación puede ajustarse a colas, Z está poco muestreado (1.05 mm/frame), y si el desfase viene de arcos el B-spline lo "corregiría" artificialmente.

## 4. Tarea actual: volumen de intersección por producto

**Objetivo:** limpiar la reverberación multiplicando vóxel a vóxel los dos volúmenes opuestos ya registrados. Solo sobrevive lo brillante en ambos (elipses); la reverberación, que no coincide entre caras, se atenúa.

### Procedimiento

1. Registrar el par opuesto (`fixed` y `moving_registrado`, en la misma grilla).
2. Normalizar cada volumen a [0, 1] **por percentiles** (valores de percentil por definir, por ejemplo p1–p99 o p0.5–p99.5, con recorte). Calcularlos solo en la región válida, para que el relleno con 0 no sesgue el percentil bajo.
3. Crear un volumen nuevo de la misma grilla que `fixed`, donde cada vóxel es `A_norm × B_norm`. Trabajar en float32.
4. Guardar como NIfTI.

### Zonas fuera de la transformación

Donde el móvil quedó fuera de la grilla, `moving_registrado` vale 0, y el producto ahí también. Opciones consideradas:

1. **Máscara de validez** (transformar una imagen de unos con la misma transformación): se usa para percentiles y producto; fuera de ella se marca "sin dato". Se sugirió esta opción, **pendiente de decisión**.
2. Dejar el 0 del producto (confunde "sin dato" con "sin señal").
3. Recortar al solapamiento.

Conviene medir cuántos vóxeles quedan en la zona inválida.

### Limitaciones conocidas del producto

- Actúa como AND: si uno de los dos es 0 en una zona, la elimina, ya sea por desfase local o porque la estructura solo se ve desde una cara.
- Depende de la calidad del registro: un desfase de pocos píxeles atenúa también las elipses.
- Para los pocos casos con desfase local (~5 mm) se evaluaron: dilatación (solo XY; poco atractiva con ~40 px) y B-spline.

### Siguientes pasos (notas del director)

1. Producto entre volúmenes opuestos (normalizar a [0,1], multiplicar).
2. Repetir con los otros pares opuestos (60°–240°, 120°–300°). Registro del resto de pares en curso.
3. Registrar los 3 volúmenes (productos) entre sí, quizá con máscara solo en el centro que comparten (a definir: qué son exactamente los "3 volúmenes" y la forma de la máscara).
4. Pasar el STL a volumen (primero en cualquier dimensión, sin relación con el escaneo).
5. Con medidas reales de las raíces, reescalar el STL a las del escaneo y usar su spacing.

## 5. STL (ground truth) — estado

- Contiene raíces y una base hexagonal (no se sabe si la base es parte de la maceta o un soporte).
- Tamaño en el visor: X = 53.96, Y = 52.84, Z = 124.24 (unidades sin confirmar). Lado de la base medido ≈ 25 mm, contra 41 mm de la maceta.
- Un factor uniforme 41/25 ≈ 1.64 no es consistente (las raíces se saldrían de la maceta); no está resuelto.
- En 3D Slicer se cargó y aparece invertido (base arriba); alinear con sliders resultó impracticable.
- Se solicitaron medidas reales: unidades/escala del modelo impreso, lado exterior e interior y altura de la base, altura total, posición de la base en la maceta, orientación respecto de la cara 001, diámetros de 2–3 raíces y si el STL es exactamente lo impreso.
- Voxelizar es simple (Slicer o Python); lo difícil es escalar y alinear. Pendiente de esas medidas.

## 6. Notas técnicas

- En el equipo Windows, Smart App Control bloqueaba las DLL de ITK. Se trabajó en Google Colab y luego en Linux.
- `params.WriteParameterFile(ruta)` acepta solo la ruta (un argumento). Con dos mapas encadenados conviene revisar qué archivos se generan.
- Visualización: ITK-SNAP con `fixed` como imagen principal y los otros volúmenes como capas adicionales.
