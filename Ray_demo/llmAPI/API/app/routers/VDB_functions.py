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
received_files_dir = os.path.join(current_path.parent.parent.parent, 'received_files')
os.makedirs(received_files_dir, exist_ok=True)

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
        

        # Check if a file is included in the request
        if file:
            if file.content_type in ['application/pdf', 'application/zip']:
                # Save the file
                file_path = os.path.join(received_files_dir, f"{current_user.username}_{file.filename}")
                with open(file_path, "wb") as file_object:
                    file_object.write(await file.read())

                # Update the file path in the request data
                data.file_path  = file_path
            else:
                raise HTTPException(status_code=400, detail="Unsupported file type")
        data_dict = data.dict()
        # Send the request to the external service
        response = requests.post(f"{Ray_service_URL}/VectorDB", json=data_dict)
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