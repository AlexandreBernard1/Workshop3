from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report

# Load the data
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predict the classes and probabilities
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# Display the results
for i in range(len(X_test)):
    print(f"Sample {i + 1}:")
    print(f"   Probabilities: {y_proba[i]}")
    print(f"   Prediction: {y_pred[i]} (Class {iris.target_names[y_pred[i]]})")
    print(f"   Actual: {y_test[i]} (Class {iris.target_names[y_test[i]]})")
    print()

# Flask application for prediction
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    # Get the model arguments from the GET request
    sepal_length = float(request.args.get('sepal_length'))
    sepal_width = float(request.args.get('sepal_width'))
    petal_length = float(request.args.get('petal_length'))
    petal_width = float(request.args.get('petal_width'))

    # Normalize the input data
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    input_data = scaler.transform(input_data)

    # Make the prediction
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]

    # Create a JSON response
    response = {
        'prediction': int(prediction),
        'class_name': iris.target_names[int(prediction)],
        'prediction_proba': prediction_proba.tolist()
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)