# Importation des bibliothèques nécessaires
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

# Chargement du jeu de données Iris
iris = load_iris()
X = iris.data
y = iris.target

# Séparation du jeu de données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Création du modèle de forêt aléatoire
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Entraînement du modèle sur l'ensemble d'entraînement
random_forest_model.fit(X_train, y_train)

# Prédiction des probabilités sur l'ensemble de test
y_probabilities = random_forest_model.predict_proba(X_test)

# Prédiction des classes sur l'ensemble de test
y_pred = random_forest_model.predict(X_test)

# Affichage des résultats
for i in range(len(X_test)):
    print(f"Échantillon {i + 1}:")
    print(f"   Probabilités: {y_probabilities[i]}")
    print(f"   Prédiction: {y_pred[i]} (Classe {iris.target_names[y_pred[i]]})")
    print(f"   Réelle: {y_test[i]} (Classe {iris.target_names[y_test[i]]})")


# Calcul de la précision, du recall et de la matrice de confusion
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

# Affichage des résultats
print(f"\nPrécision : {precision}")
print(f"Recall : {recall}")
print(f"\nMatrice de confusion :\n{conf_matrix}")
