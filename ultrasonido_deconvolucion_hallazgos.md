# Exploración de deconvolución para ultrasonido de raíces

## Objetivo

El objetivo del análisis es procesar un ultrasonido de una maceta con una máquina diseñada para reconstruir imágenes/volumen 3D y tratar de eliminar **reverberaciones locales periódicas** que aparecen como pequeñas zonas con líneas horizontales repetidas.

La finalidad es conservar las estructuras correspondientes a las **raíces naturales o sintéticas**, eliminando únicamente esos artefactos.

---

## Datos del DICOM

El DICOM fue cargado con `pydicom` mediante:

```python
ds = pydicom.dcmread(dicom_path)
volume = ds.pixel_array
```

La forma obtenida fue:

```text
(236, 768, 1024, 3)
```

y no `(236, 768, 1024)` como se esperaba inicialmente.

Características observadas:

```text
dtype: uint8
min: 0
max: 255
```

Los tres canales resultaron ser idénticos en estadísticas:

```text
Canal 0: min=0, max=255, mean=5.92
Canal 1: min=0, max=255, mean=5.92
Canal 2: min=0, max=255, mean=5.92
```

Visualmente también se comprobó que los tres canales contienen la misma información.

Por lo tanto, se decidió trabajar con un único canal:

```python
volume = volume[..., 0]
```

obteniendo:

```text
(236, 768, 1024)
```

### Importante

Se decidió **no convertir el DICOM a AVI** para este procesamiento. Es preferible trabajar directamente con los datos del DICOM para evitar una conversión intermedia innecesaria y conservar la información disponible.

---

## Orientación y estructura de la imagen

Se visualizaron distintas secciones del volumen para entender la orientación.

Para un frame:

```python
bscan = volume[118]
```

la dimensión es:

```text
(768, 1024)
```

Las líneas de reverberación de interés aparecen como **líneas horizontales** en el B-scan.

El análisis se está realizando sobre el eje vertical/profundidad de la imagen.

---

## Problema observado en las imágenes

Se identificaron tres tipos de estructuras:

### 1. Bordes de la maceta

En los laterales aparecen líneas muy intensas, asociadas aparentemente al material de la maceta, con algo de reverberación.

Por ahora **no son la prioridad** porque son fáciles de eliminar mediante un recorte espacial.

### 2. Raíces

En el centro del ultrasonido aparecen objetos que corresponden a raíces naturales o sintéticas.

Estas estructuras son las que se quieren conservar.

### 3. Reverberaciones locales

En distintas posiciones del B-scan aparecen pequeñas regiones donde se observan varias líneas horizontales brillantes y aproximadamente repetidas.

Estas son las estructuras que se quieren eliminar.

Características importantes:

- son locales;
- no ocupan toda la imagen;
- aparecen en distintas posiciones;
- pueden estar superpuestas con raíces;
- tienen una repetición aproximadamente periódica en profundidad.

Por este motivo, se concluyó que **no conviene aplicar un filtro global a todo el B-scan**.

---

## Recorte utilizado

Se definió una región de interés para eliminar principalmente los bordes y el texto/fondo del equipo.

Coordenadas:

```text
Esquina superior izquierda: (365, 115)
Esquina inferior derecha:   (650, 690)
```

Interpretando las coordenadas como `(x, y)`:

```python
X_MIN = 365
X_MAX = 650

Y_MIN = 115
Y_MAX = 690
```

El recorte se realiza como:

```python
bscan_crop = bscan[
    Y_MIN:Y_MAX,
    X_MIN:X_MAX
]
```

Dimensiones obtenidas:

```text
B-scan original:   (768, 1024)
B-scan recortado:  (575, 285)
```

Se verificó visualmente que el recorte es correcto. Puede quedar algún píxel de fondo negro en los bordes, pero se considera irrelevante para el análisis actual.

---

## Primer intento: deconvolución predictiva AR

Inicialmente se implementó una deconvolución predictiva basada en un modelo autoregresivo.

El resultado fue negativo.

La imagen filtrada perdió gran parte de la información útil y quedaron principalmente bordes/estructuras horizontales. Esto mostró que el filtro estaba eliminando estructuras predecibles de la imagen en lugar de atacar específicamente las reverberaciones.

Además, la implementación inicial tenía un problema: el parámetro `MIN_LAG` no se utilizaba realmente para construir el predictor.

### Conclusión

No utilizar el filtro AR global sobre todo el B-scan.

---

## Segundo intento: modelo explícito de reverberación

Se planteó un modelo:

\[
y[n] = x[n] + a x[n-L]
\]

con una inversión aproximada:

\[
x[n] = y[n] - a x[n-L]
\]

La intención era modelar una reverberación periódica como copias separadas por un lag `L`.

---

## Análisis inicial de autocorrelación

Sobre el B-scan completo se encontró:

```text
Lag = 109 muestras   autocorrelación = 0.245
Lag = 229 muestras   autocorrelación = 0.073
Lag =  81 muestras   autocorrelación = -0.079
Lag = 144 muestras   autocorrelación = -0.005
Lag = 200 muestras   autocorrelación = 0.018
Lag =  64 muestras   autocorrelación = -0.087
```

Inicialmente se consideró `109` como candidato.

Sin embargo, después de realizar el recorte, `109` dejó de ser el pico dominante.

Esto sugirió que el resultado inicial estaba influenciado por elementos fuera de la región útil, como texto, bordes u otras estructuras.

---

## Análisis de autocorrelación después del recorte

Sobre el B-scan recortado se obtuvo:

```text
Lag = 224 muestras   autocorrelación = 0.148
Lag =  81 muestras   autocorrelación = -0.015
Lag =  95 muestras   autocorrelación = -0.038
Lag = 290 muestras   autocorrelación = -0.037
Lag =  67 muestras   autocorrelación = -0.021
```

Se probó `lag=224` como posible período.

---

## Prueba de deconvolución con lag=224

Se implementó:

```python
result[n, x] = (
    signal[n]
    - attenuation * result[n - lag, x]
)
```

con:

```text
lag = 224
attenuation = 0.10, 0.20, 0.30, 0.40
```

El resultado fue prácticamente igual para los distintos valores de `attenuation`.

Además, el intento anterior con `attenuation=0.25` y luego `0.50` no produjo una eliminación útil de las líneas y generó **fantasmas del texto** cuando se aplicaba sobre la imagen sin recortar.

### Conclusión

El modelo global:

```text
y[n] = x[n] + a*x[n-224]
```

no parece describir correctamente las reverberaciones de interés.

El valor `224` probablemente estaba capturando estructuras distintas a distintas profundidades, y no copias reverberadas de una misma estructura.

---

## Análisis de A-lines

Se graficaron A-lines individuales del B-scan recortado:

```python
bscan_crop[:, x]
```

para:

```text
x = 70
x = 140
x = 210
```

En las A-lines se observaron estructuras fuertes aproximadamente separadas por unos 220-225 píxeles en algunos casos.

Por ejemplo, en una de ellas se observaron estructuras alrededor de:

```text
~10
~230
~455
```

Esto inicialmente parecía compatible con `L≈224`, pero se concluyó que esas estructuras no necesariamente eran copias reverberadas entre sí; podían ser estructuras físicas distintas a diferentes profundidades.

Por ello no se consideró suficiente para justificar una deconvolución con `L=224`.

---

## Análisis de periodicidad local

Se decidió cambiar el enfoque.

La reverberación de interés es:

- local;
- pequeña;
- aproximadamente horizontal;
- repetitiva en profundidad.

Por lo tanto, no se debe buscar un único período global de toda la imagen.

La estrategia propuesta es:

```text
DICOM
  ↓
selección de canal
  ↓
selección de frame
  ↓
recorte
  ↓
división en pequeñas ventanas
  ↓
medición de periodicidad local
  ↓
detección de regiones candidatas
  ↓
estimación del período local
  ↓
deconvolución solamente en esas regiones
```

---

## Primer análisis de periodicidad corta

Sobre el B-scan recortado se eliminó una tendencia lenta del perfil vertical y se calculó autocorrelación para buscar períodos cortos.

Resultado:

```text
Lag =  15 muestras   autocorrelación = -0.0043
Lag =  81 muestras   autocorrelación =  0.0826
Lag =  39 muestras   autocorrelación =  0.0390
Lag =  67 muestras   autocorrelación =  0.0379
Lag =  53 muestras   autocorrelación =  0.0120
Lag =  96 muestras   autocorrelación =  0.0527
Lag =  34 muestras   autocorrelación =  0.0389
```

`81` fue el candidato positivo más fuerte, pero su autocorrelación fue solamente `0.0826`, por lo que no se consideró suficientemente convincente.

Los distintos lags (`34`, `39`, `67`, `81`, `96`) tampoco forman una secuencia armónica clara.

### Conclusión

No conviene elegir `81` como período global.

---

# Estrategia actual

La estrategia que se considera más adecuada actualmente es **detección local de regiones periódicas**.

Se propuso recorrer el B-scan recortado utilizando ventanas pequeñas, inicialmente:

```text
WINDOW_Y = 60
WINDOW_X = 40
```

Para cada ventana:

1. promediar horizontalmente;
2. analizar la variación de intensidad en profundidad;
3. calcular autocorrelación;
4. buscar un pico positivo de periodicidad;
5. registrar:
   - score de periodicidad;
   - lag dominante.

Esto genera dos mapas:

### Mapa 1

Score de periodicidad local.

Permite identificar dónde existen regiones con comportamiento repetitivo.

### Mapa 2

Lag dominante por región.

Permite saber qué período tiene cada región candidata.

---

## Próximo paso

El próximo experimento pendiente es ejecutar el detector local de periodicidad sobre:

```text
bscan_crop
shape = (575, 285)
```

usando aproximadamente:

```python
WINDOW_Y = 60
WINDOW_X = 40

MIN_LAG = 5
MAX_LAG = 50
```

El objetivo no es todavía filtrar.

Primero hay que comprobar visualmente que las regiones con score alto coinciden con las pequeñas zonas de líneas horizontales que se identifican visualmente como reverberaciones.

Si la detección funciona:

```text
B-scan
   ↓
regiones periódicas detectadas
   ↓
máscara
   ↓
deconvolución local
```

La deconvolución se aplicaría únicamente dentro de esas regiones, para minimizar la alteración de las raíces y del resto del sustrato.

---

## Consideraciones sobre el tipo de dato

El DICOM proporciona imágenes:

```text
uint8
0–255
```

Esto sugiere que estamos trabajando con una representación B-mode ya procesada por el equipo, y no necesariamente con datos RF crudos.

Por lo tanto, la deconvolución no debe interpretarse todavía como una reconstrucción física exacta de la señal ultrasónica. Es más apropiado considerarla, por ahora, como una técnica de **supresión de patrones de reverberación en la imagen B-mode**.

Si en algún momento se consigue acceso a RF/raw ultrasound data del equipo, la deconvolución predictiva podría plantearse de una manera físicamente más rigurosa.

---

## Estado actual resumido

### Confirmado

- DICOM leído correctamente con `pydicom`.
- Shape original: `(236, 768, 1024, 3)`.
- Los tres canales son idénticos.
- Se trabaja con un canal: `(236, 768, 1024)`.
- El frame utilizado para pruebas es `118`.
- Recorte válido:
  - `x = 365:650`
  - `y = 115:690`
- Recorte resultante: `(575, 285)`.
- Las líneas de interés son reverberaciones locales, no un patrón global.
- Los bordes de la maceta pueden ignorarse mediante el recorte.

### Descartado o no recomendado

- Convertir DICOM → AVI para este procesamiento.
- Deconvolución AR global.
- Deconvolución global con `lag=109`.
- Deconvolución global con `lag=224`.
- Elegir un único período de reverberación para toda la imagen.

### Hipótesis actual

Las reverberaciones pueden detectarse por su **periodicidad local en profundidad**, y luego filtrarse únicamente en las regiones donde ese patrón aparece.

### Próxima tarea

Ejecutar el detector local de periodicidad y comprobar si las regiones detectadas coinciden con las reverberaciones visibles.

