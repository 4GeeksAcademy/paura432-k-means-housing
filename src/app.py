"""Asigna un cluster a una casa a partir de su latitud, longitud e ingreso medio.

Uso:
    python src/app.py LATITUDE LONGITUDE MEDINC
    python src/app.py 37.88 -122.23 8.33
"""

import sys
from pathlib import Path

import joblib
import pandas as pd

MODELS = Path(__file__).resolve().parent.parent / "models"
FEATURES = ["Latitude", "Longitude", "MedInc"]


def main(args):
    if len(args) != 3:
        sys.exit(__doc__)
    house = pd.DataFrame([[float(a) for a in args]], columns=FEATURES)
    kmeans = joblib.load(MODELS / "kmeans_6_clusters.joblib")
    classifier = joblib.load(MODELS / "cluster_classifier_logreg.joblib")
    print(f"K-Means: cluster {kmeans.predict(house)[0]}")
    print(f"Clasificador: cluster {classifier.predict(house)[0]}")


if __name__ == "__main__":
    main(sys.argv[1:])
