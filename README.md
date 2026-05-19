# Airplane Detection with YOLOv8

Proyecto de visión por computador enfocado en la detección de aviones mediante un modelo YOLOv8. El sistema fue construido a partir de videos fuente, extracción automática de frames, selección de imágenes útiles, etiquetado manual y entrenamiento supervisado.

## Descripción del proyecto

El objetivo de este proyecto fue desarrollar un sistema capaz de detectar aviones en imágenes de forma automática.

Para construir el dataset, primero se seleccionaron videos donde aparecían aviones en diferentes condiciones de captura, como variaciones de ángulo, distancia, iluminación y fondo. A partir de estos videos se implementó un proceso de extracción de frames, permitiendo convertir el material audiovisual en imágenes individuales.

Posteriormente, se realizó una depuración de los frames obtenidos para conservar únicamente las imágenes más útiles para el entrenamiento. Luego se llevó a cabo el etiquetado manual de los aviones mediante bounding boxes, generando así el conjunto de datos necesario para el entrenamiento del modelo.

Finalmente, se entrenó un modelo YOLOv8 con el dataset organizado, obteniendo como resultado un archivo `best.pt`, correspondiente al mejor peso del entrenamiento. Este modelo fue validado mediante pruebas con imágenes nuevas para confirmar su capacidad de detección.

## Flujo de construcción

1. Selección de videos con presencia de aviones.
2. Extracción automática de frames desde los videos.
3. Filtrado de imágenes útiles.
4. Etiquetado manual de objetos.
5. Organización del dataset en formato YOLO.
6. Entrenamiento del modelo YOLOv8.
7. Pruebas de inferencia con imágenes nuevas.
8. Generación de evidencia visual y video de demostración.

## Estructura del repositorio

```bash
airplane-detection-yolov8/
├── README.md
├── .gitignore
├── data/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── labels/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── data.yaml
├── model/
│   └── best.pt
├── notebooks/
│   └── tarea_aviones_yolov8.ipynb
├── results/
│   └── sample_predictions/
└── media/
    └── demo_video.mp4
```

## Contenido

- `data/`: contiene las imágenes, etiquetas y configuración del dataset.
- `model/`: almacena el modelo entrenado final.
- `notebooks/`: incluye el notebook principal usado en el desarrollo.
- `src/`: contiene scripts auxiliares para extracción de frames, entrenamiento o pruebas.
- `results/`: guarda ejemplos de predicciones y resultados visuales.
- `media/`: incluye el video de demostración del sistema funcionando.

## Requisitos

```bash
pip install ultralytics opencv-python matplotlib
```

## Ejemplo de inferencia

```python
from ultralytics import YOLO

model = YOLO("model/best.pt")
results = model.predict(source="data/images/test/imagen_prueba.jpg", conf=0.25, save=True)
print("Detecciones:", len(results.boxes))
```

## Resultado esperado

El sistema debe detectar la presencia del avión en la imagen de prueba, dibujando la bounding box correspondiente y generando la salida visual de la predicción.

## Video demostrativo

El repositorio incluye un video donde se muestra:

- La selección de videos fuente.
- La extracción de frames.
- La construcción del dataset.
- El proceso de entrenamiento.
- La prueba final del modelo con imágenes nuevas.

## Posibles mejoras

- Extender la detección a video en tiempo real.
- Incluir más clases de aeronaves.
- Aumentar el tamaño y variedad del dataset.
- Comparar el rendimiento con otros modelos de detección.

## Autor

Daniel Silva  
Ingeniería en Multimedia
