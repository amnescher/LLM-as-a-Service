from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Depends
from app.models import  VectorDBRequest
from app.depencencies.security import get_current_active_user
from app.database import User
import requests
from dotenv import load_dotenv
import os
import yaml
from typing import Optional
import io
# Load environment variables


import pathlib
#from app.logging_config import setup_logger
import yaml

current_path = pathlib.Path(__file__).parent

config_path = current_path.parent.parent.parent / 'cluster_conf.yaml'
# Environment and DB setup
with open(config_path, "r") as file:
    config = yaml.safe_load(file)



Ray_service_URL = config.get("Ray_service_URL")
router = APIRouter()

@router.post("/")
async def create_inference(data: VectorDBRequest = Depends(), 
                           current_user: User = Depends(get_current_active_user),
                           file: Optional[UploadFile] = File(None)):
    try:
        data.username = current_user.username
        data_dict = data.dict()

        # Check if a file is included in the request
        if file:
            # Preparing the file and data for the multipart/form-data request
            file_content = await file.read()
            files = {'file': (file.filename, io.BytesIO(file_content), file.content_type)}
            data_dict.pop('file', None)  # Ensure no conflicting 'file' key in data

            # Multipart/form-data request
            response = requests.post(f"{Ray_service_URL}/VectorDB", files=files, params=data_dict)
        else:
            # Standard request without file
            response = requests.post(f"{Ray_service_URL}/VectorDB", params=data_dict)

        response.raise_for_status()  # Raises an HTTPError for unsuccessful status codes
        response_data = response.json()
        return {"username": current_user.username, "response": response_data}

    except requests.HTTPError as e:
        if response.status_code == 400:
            raise HTTPException(status_code=400, detail="Bad request to the other API service.")
        else:
            raise HTTPException(status_code=500, detail=f"Failed to forward request to the other API service. Error: {e}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")