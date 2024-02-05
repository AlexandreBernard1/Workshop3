from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier
import numpy as np

app = Flask(__name__)


# Load the Iris dataset and train the model
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)


@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Get input parameters from the request
        sepal_length = float(request.args.get('sepal_length'))
        sepal_width = float(request.args.get('sepal_width'))
        petal_length = float(request.args.get('petal_length'))
        petal_width = float(request.args.get('petal_width'))

        # Make a prediction
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        probabilities = model.predict_proba(input_data)[0]
        prediction = model.predict(input_data)[0]
        predicted_class = iris.target_names[prediction]

        # Prepare the response
        response = {
            'prediction': int(prediction),
            'predicted_class': predicted_class,
            'probabilities': probabilities.tolist(),
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
