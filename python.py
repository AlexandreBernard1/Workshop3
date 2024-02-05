import requests

sepal_length = 5.1
sepal_width = 3.5
petal_length = 1.4
petal_width = 0.2

# Définir les trois URLs
url_theo = f"https://d78e-89-30-29-68.ngrok-free.app/predict?sepal_length={sepal_width}&sepal_width={sepal_width}&petal_length={petal_length}&petal_width={petal_width}"
url2_jeremy = f"https://a961-89-30-29-68.ngrok-free.app/predict?sepal_length={sepal_width}&sepal_width={sepal_width}&petal_length={petal_length}&petal_width={petal_width}"
url3_alexandre = f"https://4db6-45-84-137-18.ngrok-free.app//predict?sepal_length={sepal_width}&sepal_width={sepal_width}&petal_length={petal_length}&petal_width={petal_width}"

# Fonction pour faire la requête et obtenir les probabilités
def get_probabilities(url):
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data.get("prediction_proba", [])
    else:
        print(f"Error: {response.status_code}")
        return []

# Obtenir les probabilités pour chaque URL
probabilities_theo = get_probabilities(url_theo)
probabilities_jeremy = get_probabilities(url2_jeremy)
probabilities_alexandre = get_probabilities(url3_alexandre)

# Faire la moyenne des probabilités
average_probabilities = [
    sum(p) / len(p) for p in zip(probabilities_theo, probabilities_jeremy, probabilities_alexandre)
]

# Afficher les résultats
print("Probabilités de Theo:", probabilities_theo)
print("Probabilités de Jeremy:", probabilities_jeremy)
print("Probabilités d'Alexandre:", probabilities_alexandre)
print("Moyenne des probabilités:", average_probabilities)
