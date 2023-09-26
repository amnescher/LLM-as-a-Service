import requests
import json
BASE_URL = "http://localhost:5000" 
#------------------------------------- Interact with DataBase --------------------------
def add_user(username):
    endpoint = "/add_user/"
    url = BASE_URL + endpoint
    data = {"username": username}  # Provide the necessary data
    response = requests.post(url, json=data)
    return response.json()

def add_conversation(username,content):
    endpoint = "/add_conversation/"
    url = BASE_URL + endpoint
    data = {"username": username, "content": json.dumps(content)}
    response = requests.post(url, json=data)
    return response.json()

def get_all_data():
    endpoint = "/get_all_data/"
    url = BASE_URL + endpoint
    response = requests.get(url)
    return response.json()

def delete_user(username):
    endpoint = "/delete_user/"
    url = BASE_URL + endpoint
    data = {"username": username}
    response = requests.delete(url, json=data)
    return response.json()

def delete_conversation(username,conversation_number):
    endpoint = "/delete_conversation/"
    url = BASE_URL + endpoint
    data = {"username": username,"conversation_number":conversation_number}
    response = requests.delete(url, json=data)
    return response.json()

def check_user_existence(username):
    endpoint = "/check_user_existence/"
    url = BASE_URL + endpoint
    data = {"username": username}
    response = requests.get(url, json=data)
    return response.json()

def retrieve_conversation(username, conversation_number):
    endpoint = "/retrieve_conversation/"
    url = BASE_URL + endpoint
    data = {"username": username,"conversation_number":conversation_number}
    response = requests.post(url, json=data)
    return response.json()

def retrieve_latest_conversation(username):
    endpoint = "/retrieve_latest_conversation/"
    url = BASE_URL + endpoint
    data = {"username": username}
    response = requests.get(url, json=data)
    return response.json()
    
def update_conversation(username, conversation_number, conversation_content):
    endpoint = "/update_conversation/"
    url = BASE_URL + endpoint
    data = {"username": username,"conversation_number":conversation_number,"content": json.dumps(conversation_content)}
    response = requests.post(url, json=data)
    return response.json()