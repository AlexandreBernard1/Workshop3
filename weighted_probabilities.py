import requests
from sklearn.datasets import load_iris

# Initialize weights for each model
weights = [1.0, 1.0, 1.0]

# Define the three URLs
url_theo = f"https://d78e-89-30-29-68.ngrok-free.app/predict?"
url2_jeremy = f"https://a961-89-30-29-68.ngrok-free.app/predict?"
url3_alexandre = f"https://4db6-45-84-137-18.ngrok-free.app//predict?"

iris = load_iris()

X = iris.data
y = iris.target


# Function to make the request and get probabilities
def get_probabilities(url):
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data.get("prediction_proba", [])
    else:
        print(f"Error: {response.status_code}")
        return []


# Function to update weights based on accuracy
def update_weights(true_class, predicted_classes, weights, learning_rate=0.02):
    for i in range(len(weights)):
        if predicted_classes[i] == true_class:
            weights[i] += learning_rate
        else:
            weights[i] -= learning_rate
        weights[i] = max(0.0, min(1.0, weights[i]))  # Ensure weights are within [0, 1] range
    return weights


for value, target in zip(X, y):
    sepal_length = value[0]
    sepal_width = value[1]
    petal_length = value[2]
    petal_width = value[3]

    url_suite = f"sepal_length={sepal_length}&sepal_width={sepal_width}&petal_length={petal_length}&petal_width={petal_width}"

    # Get probabilities for each URL
    probabilities_theo = get_probabilities(url_theo + url_suite)
    probabilities_jeremy = get_probabilities(url2_jeremy + url_suite)
    probabilities_alexandre = get_probabilities(url3_alexandre + url_suite)

    # Combine probabilities using the weighted average
    weighted_average_probabilities = [
        sum(w * p for w, p in zip(weights, prob)) / sum(weights) for prob in zip(probabilities_theo, probabilities_jeremy, probabilities_alexandre)
    ]

    # Get the predicted class based on the weighted average probabilities
    weighted_consensus_prediction = weighted_average_probabilities.index(max(weighted_average_probabilities))

    # Update weights based on accuracy (assuming the true class is known)
    true_class = target
    predicted_classes = [prob.index(max(prob)) for prob in zip(probabilities_theo, probabilities_jeremy, probabilities_alexandre)]
    weights = update_weights(true_class, predicted_classes, weights)

    # Display the results
    print("Probabilities of Theo:", probabilities_theo)
    print("Probabilities of Jeremy:", probabilities_jeremy)
    print("Probabilities of Alexandre:", probabilities_alexandre)
    print("Weighted Average Probabilities:", weighted_average_probabilities)
    print("True prediction: ", true_class)
    print("Weighted Consensus Prediction:", weighted_consensus_prediction)
    print("Updated Weights:", weights)
    print("---------------------------------------")