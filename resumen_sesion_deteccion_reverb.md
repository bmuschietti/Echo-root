RESUMEN DE SESIÓN - Detección local de reverberación (echo root)
==================================================================

1. PUNTO DE PARTIDA
--------------------
Se retomó el trabajo documentado en "ultrasonido_deconvolucion_hallazgos.md":
reverberaciones locales (no globales) en B-scans, ya se habían descartado
la deconvolución AR global y modelos con lag fijo (109, 224). La conclusión
vigente era que había que detectar periodicidad en ventanas locales antes
de intentar filtrar.

2. REVISIÓN INICIAL DEL PROCESO ANTERIOR
------------------------------------------
Se señalaron 4 puntos de riesgo sobre el enfoque documentado:
  - Relación ventana/lag: con window_y=60 y max_lag=50 la autocorrelación
    en los lags más grandes quedaba mal estimada (pocas muestras de
    solapamiento).
  - Ventanas sin solapamiento generan mapas en bloques.
  - La periodicidad sola no distingue reverberación de estructura real
    (raíz, capas de sustrato) — faltaba un criterio adicional.
  - El lag ~220-230 aparecía repetido en varios análisis; se sugirió no
    descartarlo del todo sin revisar si correspondía a un artefacto de
    sistema (paredes/fondo de maceta) en vez de reverb local.

Se armó un script inicial (local_periodicity_detector.py) implementando
el detector por ventanas descrito en el md, con auto-ajuste de max_lag
según el tamaño de ventana (esto luego se sacó, ver punto 5).

3. IMAGEN DE EJEMPLO Y CORRECCIÓN DE INTERPRETACIÓN
-----------------------------------------------------
Se analizó un frame de ejemplo (frame_0009.png). Primera lectura errónea:
se interpretaron las bandas superior/inferior como posible artefacto de
sistema. El usuario corrigió: esas bandas SÍ son reverberación conocida
por bordes de maceta (no prioritaria); el objetivo real son líneas
horizontales repetidas (~5-6 veces) que aparecen en zonas puntuales del
centro de la imagen, junto a las raíces.

4. DECISIÓN DE ENFOQUE: 2D vs 3D
-----------------------------------
Se plantearon dos líneas posibles:
  a) Seguir con detección 2D por frame (ventanas + periodicidad).
  b) Comparar consistencia entre las 6 rotaciones ya registradas en 3D
     (una raíz real debería aparecer en las 6 vistas, una reverberación
     probablemente no).
Se evaluó que 3D es conceptualmente más sólida (usa info física real),
pero depende de que el registro (VolumeRegistrator) sea preciso.

Validación intentada: se generó un volumen a partir de 6 DICOM de raíces
sintéticas (ground truth conocido). Resultado en ITK-SNAP: la forma
esperada de la raíz sintética NO se vio coherente — se observó un patrón
de bandas verticales tipo rayado, compatible con desalineación entre
vistas. Conclusión: el registro no está listo para usarse como validación
todavía. SE PAUSA la línea 3D por falta de información sobre los escaneos
usados; se retoma más adelante.

5. VUELTA A 2D: AJUSTES SOBRE EL DETECTOR
--------------------------------------------
Se fijó window_y=80 de forma manual (regla usada: window_y >= 3*max_lag +
tolerancia), eliminando el auto-ajuste dinámico y sus warnings.

Se decidió agregar un filtro de armónicos: en vez de tomar el pico de
autocorrelación en un solo lag, exigir picos consistentes en lag, 2*lag,
3*lag (evidencia de que el patrón se repite varias veces, no un pico
aislado que podría ser estructura real).

Parámetros iniciales del filtro de armónicos (estimados a ojo, no
medidos): mínimo 3 armónicos, tolerancia ±10px (grosor de línea asumido).

6. MEDICIÓN VISUAL SOBRE GRILLA
-----------------------------------
Se generó un plot del frame con grilla de referencia cada 10px para medir
en vez de estimar a ciegas. Con eso se corrigieron los parámetros:
  - Grosor de línea observado: ~5-8px -> tolerance=5
  - Separación entre líneas observada: ~10-15px -> min_lag=10, max_lag=16
Se identificaron también 3 formas triangulares/cónicas en la imagen que
inicialmente generaron duda; el usuario confirmó que corresponden a
raíces sintéticas (estructuras a preservar, no reverb).

7. BUG DETECTADO EN EL FILTRO DE ARMÓNICOS (v1)
---------------------------------------------------
Al correr el detector con estos parámetros, el mapa de score y el de
"n° de armónicos" salieron casi uniformes en toda la imagen (sin
distinguir zonas) — señal de que el filtro no estaba discriminando nada.

Causa encontrada: la función tomaba el máximo de la ventana de tolerancia
para cada armónico SIN exigir un mínimo de correlación real, y además
para lags chicos las ventanas de tolerancia de armónicos consecutivos se
solapaban entre sí. Resultado: "encontraba" armónicos artificialmente en
todos lados.

8. REDISEÑO DEL FILTRO DE ARMÓNICOS (v2)
---------------------------------------------
Cambios acordados e implementados:
  - Umbral mínimo de correlación (min_correlation=0.15) para que un pico
    cuente como armónico válido.
  - Chequeo explícito de no-solapamiento entre ventanas de armónicos
    (tolerance < lag/2).
  - Se corta la búsqueda de armónicos en cuanto uno no supera el umbral.

Resultado: el mapa dejó de ser uniforme; las dos bandas de reverberación
ya identificadas a ojo se destacaron con score y n° de armónicos más
altos. Se detectó un posible falso positivo puntual: una celda coincidente
con una de las raíces sintéticas (parcialmente tapada por reverb) también
dio score alto.

9. CAMBIO DE ESCALA: DE BINARIO A CONTEO CONTINUO
-------------------------------------------------------
Se identificó que el filtro v2 devolvía resultado binario (3 armónicos o
0, sin valores intermedios), perdiendo información útil para decidir
umbrales. Se modificó para:
  - Subir max_harmonics a 10 (rango de conteo más amplio).
  - Eliminar el corte que forzaba score=0 si no se llegaba al máximo;
    ahora siempre se devuelve el conteo real de armónicos válidos
    encontrados (0 a 10) y el score promedio de esos picos.

Resultado: aparecieron valores intermedios (0 a ~4-5) en el mapa de
armónicos, permitiendo ver gradiente en vez de todo-o-nada. Se señaló
como pendiente: con max_lag=16, los armónicos más altos (7mo en adelante)
exceden window_y=80, por lo que el conteo real está limitado por el
tamaño de ventana, no solo por ausencia de periodicidad.

10. AJUSTE DE VENTANA EN X
-------------------------------
Se probó reducir window_x de 40 a 20, y luego al extremo window_x=1
(A-line por A-line), motivado por sospecha de que promediar muchas
columnas mezcla información de raíz y reverberación vecinas.

Se corrigió un bug menor (ValueError por stride_x=0 al usar window_x=1;
se agregó max(1, window_x//2) al cálculo del stride) y un problema de
visualización (mapas ilegibles por aspecto de imagen; se agregó
aspect="auto" en los imshow).

Con window_x=1 se observó una franja vertical angosta de score bajo en
medio de una banda de score alto, compatible con una raíz real rodeada
de reverberación lateral, según una imagen de zoom analizada aparte.
Pendiente de confirmar si esa franja coincide en posición exacta con la
raíz observada visualmente.

11. ESTADO ACTUAL / PENDIENTES
------------------------------------
  - Falso positivo sobre raíz sintética parcialmente tapada por reverb:
    sin resolver, pendiente de decisión sobre cómo tratarlo.
  - max_lag=16 puede estar limitando la detección en zonas con periodo
    mayor (ej. pared superior de la maceta, con líneas más juntas o con
    otro período no cubierto en el rango probado).
  - Línea 3D (multi-vista) pausada por registro no confiable todavía;
    retomar cuando haya más información sobre los escaneos.
  - No se aplicó ningún filtrado/deconvolución todavía; todo el trabajo
    de hoy fue de detección y visualización exploratoria.
