import requests
from pathlib import Path

# Set the URL of the FastAPI endpoint
url = 'http://127.0.0.1:8000/VectorDB/'

# Set the parameters
params = {'username': 'amin', 'collection_name': 'xc'}


response = requests.post(url, params=params)

# Check if the request was successful
if response.status_code == 200:
    print("Request was successful.")
    print("Response:", response.json())
else:
    print("Request failed.")
    print("Status Code:", response.status_code)
    print("Response:", response.text)
