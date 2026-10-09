# Detección de formas geométricas con YOLO

Electiva IV – Deep Computer Vision · USTA Bucaramanga · Mecatrónica

Detecta **círculo, cuadrado y triángulo** en tiempo real con **YOLOv8n** (Ultralytics). La ventana de la webcam muestra, para cada figura, el *bounding box*, la clase, el **color** (por HSV, no por YOLO) y la confianza, más un overlay fijo con el **Acc_val** del modelo.

## Estructura

```
proyecto_yolo/
├── data/
│   ├── generate_dataset.py   # genera el dataset sintético (imágenes 640x640 + etiquetas YOLO TXT)
│   └── add_real_data.py      # mezcla las fotos reales de la webcam con el dataset
├── models/
│   ├── best.pt               # mejores pesos entrenados
│   └── metrics.json          # métricas (las lee el overlay de la demo)
├── mis_fotos/
│   ├── vacio/                # 145 fotos reales SIN figuras (etiqueta vacía)
│   ├── roboflow_figuras/     # 110 fotos reales CON figuras, ya etiquetadas (formato YOLOv8 + data.yaml)
│   └── figuras_sin_etiquetar/# fotos originales sin etiquetar (solo respaldo)
├── docs/                     # video de la demo, matriz de confusión, muestras del dataset
├── shapes_utils.py           # clases, paleta, detección de color, métricas IoU
├── capture_frames.py         # captura fotos con la webcam para ampliar el dataset real
├── train.py                  # entrenamiento
├── evaluate.py               # mAP@0.5, precisión, recall, matriz de confusión y Acc_val
├── demo_webcam.py            # demo en tiempo real
└── requirements.txt
```

## 0. Instalación

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
```

## 1. Preparación de datos

El dataset final mezcla **imágenes sintéticas** con **fotos reales de la webcam**, todo en formato **YOLO TXT** (`clase cx cy w h` normalizado) y partido 70 % train / 20 % val / 10 % test.

**a) Dataset sintético (1800 imágenes de 640x640).** Figuras con fondos, colores, luz, ruido y desenfoque variados, triángulos girados 0–360° y figuras "distractoras" sin etiqueta; ~12 % de las imágenes no tienen figuras.

```bash
python data/generate_dataset.py --n 1800 --out data/shapes_yolo
```

**b) Fotos reales.** Se agregan las fotos vacías (etiqueta vacía, enseñan al modelo que una lámpara o un monitor *no* es una figura) y las fotos con figuras (celular/tablet frente a la webcam) etiquetadas:

```bash
python data/add_real_data.py --vacio mis_fotos/vacio --roboflow mis_fotos/roboflow_figuras --repeat 5
```

- Reparte las fotos reales al azar 70/20/10. En **train** cada foto real se repite `--repeat 5` veces (son pocas frente a las sintéticas); en **val y test** va una sola copia, para que las métricas no se inflen.
- Las etiquetas de las fotos con figuras se generaron automáticamente (máscara HSV + contornos) y no se revisaron una por una a mano.
- `generate_dataset.py` **borra** `data/shapes_yolo` al correr: si se vuelve a ejecutar, hay que volver a correr `add_real_data.py`.
- Opcional (mejora para triángulos girados, no usada en el `best.pt` entregado): `--rot 2` agrega copias giradas 90/180/270° de las fotos con figuras en train.

**Composición final del dataset usado para entrenar `best.pt`:**

| Partición | Total | Sintéticas | Fotos vacías | Fotos con figuras |
|---|---|---|---|---|
| train | 2155 | 1260 | 102 x5 | 77 x5 |
| val | 411 | 360 | 29 | 22 |
| test | 205 | 180 | 14 | 11 |

Para ampliar las fotos reales: `python capture_frames.py` (webcam) y volver a correr el paso b.

## 2. Cómo entrenar

```bash
python train.py --model yolov8n.pt --epochs 50 --batch 16 --patience 10
```

- YOLOv8n con pesos preentrenados (*transfer learning*), `imgsz=640`, 50 épocas, **early stopping** (`patience=10`), semilla 42. Usa GPU si hay; en CPU es muy lento (en Colab con T4 tardó unos 26 min).
- Guarda los mejores pesos en `models/best.pt` (la corrida completa queda en `runs/shapes/`).
- Opcional: `--flipud 0.5` voltea verticalmente la mitad de las imágenes (ayuda con triángulos de punta abajo; no usado en el `best.pt` entregado).
- Si la GPU se queda sin memoria: `--batch 8`.

## 3. Cómo reproducir las métricas

```bash
python evaluate.py                     # val y test; conf=0.25 para Acc_val
```

Calcula mAP@0.5, precisión, recall y matriz de confusión por clase (se guardan en `runs/eval_val` y `runs/eval_test`), y el **Acc_val**, que se escribe en `models/metrics.json`.

**Acc_val** = TP / (TP + FP + FN), donde una detección es correcta (TP) si coincide con una figura real con **IoU ≥ 0.5 y la clase correcta**, con confianza ≥ 0.25. Este umbral es solo para el cálculo de Acc_val; el de la demo es independiente (`--conf` de `demo_webcam.py`).

Resultados del `best.pt` entregado (el mAP@0.5 de 0.995 es el máximo que reporta Ultralytics):

| Conjunto | Imágenes | mAP@0.5 | Precisión | Recall | TP | FP | FN | Acc_val |
|---|---|---|---|---|---|---|---|---|
| val | 411 | 0.995 | 0.999 | 0.999 | 963 | 3 | 0 | 99.7 % |
| test | 205 | 0.995 | 0.997 | 0.998 | 451 | 1 | 1 | 99.6 % |

Por clase, el mAP@0.5 es 0.995 en círculo, cuadrado y triángulo. Separando el Acc_val de val: **100 %** en imágenes sintéticas (933 TP) y **90.9 %** en fotos reales (30 TP, 3 FP, 0 FN en 51 imágenes), así que las fotos reales son la prueba más exigente.

## 4. Cómo ejecutar la demo en tiempo real

```bash
python demo_webcam.py                       # webcam 0, conf=0.7
python demo_webcam.py --conf 0.5            # más sensible (detecta más, más riesgo de falsos positivos)
python demo_webcam.py --cam 1               # otra cámara
python demo_webcam.py --source docs/clip_prueba.mp4        # probar con un video
python demo_webcam.py --save docs/demo_salida.mp4          # graba el video de evidencia
```

Se cierra con `q`. Cada figura muestra bbox, clase, color y confianza; arriba queda fijo el overlay `Acc_val` (leído de `models/metrics.json`). En una laptop sin GPU corre a ~3–4 FPS.

## Limitaciones conocidas

- Brecha sintético → real: en la webcam el modelo falla más que en las métricas (Acc_val real 90.9 % vs 100 % sintético).
- Las fotos reales tienen todos los triángulos con la punta arriba y pocas combinaciones forma x color (p. ej. sin círculos amarillos ni morados); por eso pueden fallar triángulos girados y algunos círculos de color poco frecuente. Las opciones `--rot` y `--flipud` apuntan a mejorar esto.
- El color se estima con umbrales HSV sobre los píxeles de la figura; con poca luz un amarillo oscuro puede verse verde oliva (hay una regla de brillo para ese caso).
