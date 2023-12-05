import requests

# Your existing parameters
param = {
 # Assuming this should be a non-empty string
    "collection_name": "videos",  
    "mode": "create_collection",
    "vectorDB_type": "Weaviate",
    # Removed "file_path" because the actual file will be sent in the request
}

# File to be sent
file_to_send = {
    'file': open('docs/paper.pdf', 'rb')  # Replace 'path_to_your_file' with the actual file path
}

# Token for authorization header
nils_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhbWluIiwiZXhwIjoxNzAxMjY4MzI0fQ.FeQ8EPN1e4WkZt9ZdJMbCUdvBEZgV4o_dvmTiF-sdLM"
url = 'http://localhost:8083/vector_DB_request/'

# Sending the request
response = requests.post(url, data=param, files=file_to_send, headers={'Authorization': f"Bearer {nils_token}"})

# Close the file after sending
file_to_send['file'].close()

# Print response
print(response.status_code)
print(response.json())
