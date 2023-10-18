# my_fastapi_app_client/client.py

import requests

class llm:
    def __init__(self, base_url, access_token: str = None):
        self.base_url = base_url
        if access_token:
            self.access_token = access_token
        else:
            self.access_token = None

    def authenticate(self, username, password):
        resp = requests.post(f"{self.base_url}/token", data={"username": username, "password": password})
        if resp.status_code == 200:
            self.access_token = resp.json().get("access_token")
        else:
            raise Exception("Authentication failed")

    def query_inference(self, query_data):
        if not self.access_token:
            raise Exception("Not authenticated, please set you access token using authenticate() method")
        headers = {"Authorization": f"Bearer {self.access_token}"}
        resp = requests.post(f"{self.base_url}/inference", json=query_data, headers=headers)
        
        if resp.status_code == 200:
            return resp.json()
        else:
            raise Exception("Inference failed")
