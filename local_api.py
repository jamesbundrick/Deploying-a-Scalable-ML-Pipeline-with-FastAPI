import requests

BASE_URL = "http://127.0.0.1:8000"

# Send a GET request to the root of the API
r = requests.get(BASE_URL)

# Print the status code and the welcome message
print(f"Status Code: {r.status_code}")
print(f"Result: {r.json()['message']}")

data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

# Send a POST request with the data above to the inference endpoint
r = requests.post(f"{BASE_URL}/data/", json=data)

# Print the status code and the model's prediction
print(f"Status Code: {r.status_code}")
print(f"Result: {r.json()['result']}")
