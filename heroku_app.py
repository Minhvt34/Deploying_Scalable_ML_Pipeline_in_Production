import requests

URL = "https://udacity-prj3.herokuapp.com/predict"
data = {
    "age": 40,
    "workclass": "Private",
    "fnlgt": 338409,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Wife",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 80,
    "native-country": "Cuba",
}

# request a response from the API using FastAPI
response = requests.post(URL, json=data, timeout=120)

# print the response
print(f"response.status_code: {response.status_code}")
print(f"Response from {URL} : {response.json()}")
