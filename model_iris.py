from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report  # Ajout des importations nécessaires

# Chargement des données
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Division des données en ensemble d'entraînement et ensemble de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisation des données
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Création et entraînement du modèle
model = LogisticRegression()
model.fit(X_train, y_train)

# Prédiction des classes et des probabilités
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# Affichage des résultats
for i in range(len(X_test)):
    print(f"Échantillon {i + 1}:")
    print(f"   Probabilités: {y_proba[i]}")
    print(f"   Prédiction: {y_pred[i]} (Classe {iris.target_names[y_pred[i]]})")
    print(f"   Réelle: {y_test[i]} (Classe {iris.target_names[y_test[i]]})")
    print()


from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    # Récupérez les arguments du modèle depuis la requête GET
    sepal_length = float(request.args.get('sepal_length'))
    sepal_width = float(request.args.get('sepal_width'))
    petal_length = float(request.args.get('petal_length'))
    petal_width = float(request.args.get('petal_width'))

    # Normalisez les données d'entrée
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    input_data = scaler.transform(input_data)

    # Faites la prédiction
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]

    # Créez une réponse JSON
    response = {
        'prediction': int(prediction),
        'class_name': iris.target_names[int(prediction)],
        'prediction_proba': prediction_proba.tolist()
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
