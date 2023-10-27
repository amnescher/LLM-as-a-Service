from fastapi import APIRouter, Depends, HTTPException
from app.models import  InferenceRequest
from app.depencencies.security import get_current_active_user
from app.database import User
import requests
from dotenv import load_dotenv
import os
import yaml

# Load environment variables


config_path = os.environ.get("CONFIG_PATH")
if not config_path:
    raise ValueError("CONFIG_PATH environment variable is not set.")

# Environment and DB setup
with open(config_path, "r") as file:
    config = yaml.safe_load(file)



Ray_service_URL = config.get("Ray_service_URL")
router = APIRouter()

@router.post("/")
async def create_inference(data: InferenceRequest, current_user: User = Depends(get_current_active_user)):
    try:
        data.username = current_user.username
        response = requests.post(f"{Ray_service_URL}/inference", json=data.dict())
        response.raise_for_status()  # Raises an HTTPError if the HTTP request returned an unsuccessful status code
        
        # Extract data from the response
        response_data = response.json()
        return {"username": current_user.username, "data": response_data}

    except requests.HTTPError as e:
        if response.status_code == 400:
            raise HTTPException(status_code=400, detail="Bad request to the other API service.")
        else:
            raise HTTPException(status_code=500, detail=f"Failed to forward request to the other API service. Error: {e}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")
