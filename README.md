# Agrupación de casas con K-Means: California Housing

Proyecto del módulo «Modelos de clasificación» de 4Geeks (*Proyecto Tutorial K-Medias*), creado a partir del template [`machine-learning-python-template`](https://github.com/4GeeksAcademy/machine-learning-python-template).

Un K-Means agrupa en 6 clusters las casas de California según su ubicación (`Latitude`, `Longitude`) y su ingreso medio (`MedInc`). Después, un clasificador supervisado aprende a asignar esos clusters. El análisis y las conclusiones están en el notebook; este README explica qué hay en el repositorio y cómo ejecutarlo.

## Resultados

- K-Means divide el estado en norte, centro y sur, separa por ingreso dentro de cada zona y forma un grupo aparte con los ingresos más altos.
- Las casas de test se asignan con el mismo criterio que las de train: mismo reparto por cluster, mismas medias y misma distancia a los centroides.
- Clasificador elegido: regresión logística con `C=10000`. Acierta el 99,78 % en test (9 errores de 4.128, casi todos sobre una frontera entre clusters). El random forest se queda en el 99,54 %.

## Contenido

| Ruta | Qué es |
| --- | --- |
| `src/explore.ipynb` | Notebook ejecutado con los pasos 1 a 5 del enunciado |
| `src/app.py` | Asigna un cluster a una casa con los dos modelos guardados |
| `data/raw/housing.csv` | Dataset del enunciado, descargado sin cambios del repositorio del proyecto (20.640 filas) |
| `data/processed/housing_{train,test}_clusters.csv` | Train (80 %) y test (20 %) con `Latitude`, `Longitude`, `MedInc` y su `cluster` |
| `models/kmeans_6_clusters.joblib` | K-Means con 6 clusters |
| `models/cluster_classifier_logreg.joblib` | Clasificador supervisado: `StandardScaler` y regresión logística |
| `requirements.txt` | Dependencias del template más `joblib`, todas con versión fija |

El resto (`.devcontainer/`, `src/utils.py`, `.env.example`, `data/interim/`) viene del template y este proyecto no lo usa.

## Ejecución

En Codespaces el entorno instala `requirements.txt` al crearse. En local hace falta Python 3.12 o 3.13:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Para asignar un cluster a una casa (latitud, longitud y `MedInc` en decenas de miles de dólares):

```sh
python src/app.py 37.88 -122.23 8.33
```

Para volver a ejecutar el notebook entero, con sus salidas (tarda menos de un minuto):

```sh
cd src
jupyter nbconvert --to notebook --execute --inplace explore.ipynb
```

Los modelos se guardan con `joblib`. Para cargarlos hay que usar la versión de scikit-learn fijada en `requirements.txt`.

## Decisiones

- **Sin escalar las variables para K-Means.** Las tres tienen una desviación típica casi igual (2,1, 2,0 y 1,9), así que escalarlas no cambia los grupos en la práctica, y los centroides se leen en grados y en dólares. El clasificador sí escala, dentro de su propio pipeline.
- **`cluster` como columna categórica.** Los números de cluster son identificadores, no cantidades.
- **Regresión logística como modelo supervisado.** K-Means separa los grupos con fronteras lineales y la regresión logística también, así que puede reproducirlas casi exactamente. El notebook la compara con árbol de decisión, random forest y KNN mediante validación cruzada.
