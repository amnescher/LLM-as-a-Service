from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Security
from app.models import VectorDBRequest,LoginUser
from app.depencencies.security import get_current_active_user
from app.depencencies.security import get_current_active_user
from app.database import User
import requests
import os
import yaml
import secrets
import zipfile
from typing import Optional
import pathlib
from sqlalchemy.orm import Session
from app.database import get_db
import weaviate
import logging

# Load environment variables and setup
current_path = pathlib.Path(__file__).parent
config_path = current_path.parent.parent.parent / 'cluster_conf.yaml'
received_files_dir = os.path.join(current_path.parent.parent.parent, 'received_files')
os.makedirs(received_files_dir, exist_ok=True)

logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            filename="app.log",  # specify the file name if you want logging to be stored in a file
            filemode="a",  # append to the log file if it exists
        )

logger = logging.getLogger(__name__)
logger.propagate = True

# Load configuration
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

#weaviate_client = weaviate.Client("http://localhost:8080")

Ray_service_URL = config.get("Ray_service_URL")
router = APIRouter()

@router.post("/")
async def query_vectorDB(data: VectorDBRequest = Depends(), 
                         current_user: User = Depends(get_current_active_user),
                         file: Optional[UploadFile] = File(None)):
    try:
        print(f"Received data: {data.dict()}")  # Debug print
        print(f"Received file: {file.filename if file else 'No file'}")
        data.username = current_user.username
        
        # Check if a file is included in the request
        if file:
            # Create a random directory for the file
            random_dir = secrets.token_hex(8)  # Generates a random 16-character string
            file_dir = os.path.join(received_files_dir, random_dir)
            os.makedirs(file_dir, exist_ok=True)

            # Save the file in the random directory
            file_path = os.path.join(file_dir, file.filename)
            with open(file_path, "wb") as file_object:
                file_object.write(await file.read())

            # If the file is a ZIP file, extract it
            if file.content_type == 'application/zip':
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    # Extract only the files not in _MACOSX directory
                    for zip_info in zip_ref.infolist():
                        if not zip_info.filename.startswith('__MACOSX/'):
                            zip_ref.extract(zip_info, file_dir)

            # Update the file path in the request data
            data.file_path = file_dir
        else:
            data.file_path = None

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

@router.post("/add_vdb_class/")
async def add_vdb_class(
    data: VectorDBRequest,
    current_user: User = Depends(get_current_active_user)):

    try:            
            weaviate_client = weaviate.Client("http://localhost:8080")
            prefix = current_user.username
            cls = str(prefix) + "_" + str(data.class_name)
            #class_description = str(description)
            vectorizer = 'text2vec-transformers'
            if cls is not None:
                schema = {'classes': [ 
                    {
                            'class': str(cls),
                            'description': 'normal description',
                            'vectorizer': str(vectorizer),
                            'moduleConfig': {
                                str(vectorizer): {
                                    'vectorizerClassName': False,
                                    }
                            },
                            'properties': [{
                                'dataType': ['text'],
                                'description': 'the text from the documents parsed',
                                'moduleConfig': {
                                    str(vectorizer): {
                                        'skip': False,
                                        'vectorizePropertyName': False,
                                        }
                                },
                                'name': 'page_content',
                            },
                            {
                                'name': 'document_title',
                                'dataType': ['text'],
                            }],      
                            },
                ]}
                weaviate_client.schema.create(schema)
            else:
                return {"error": "No class name provided"}
    except Exception as e:
        return {"error": str(e)}

@router.post("/remove_vdb_class/")  
async def delete_weaviate_class(data: VectorDBRequest,
    current_user: User = Depends(get_current_active_user)):
        try: 
            weaviate_client = weaviate.Client("http://localhost:8080")
            username = current_user.username
            class_name = data.class_name
            full_class_name = str(username) + "_" + str(class_name)
            weaviate_client.schema.delete_class(full_class_name)

        except Exception as e:
            return {"error": str(e)}

@router.post("/get_docs_in_class/")    
async def query_weaviate_document_names(data: VectorDBRequest,
    current_user: User = Depends(get_current_active_user)):
    try:
        weaviate_client = weaviate.Client("http://localhost:8080")
        username = current_user.username
        class_properties = ["document_title"]
        class_name = data.class_name
        #full_class_name = str(username) + "_" + str(class_name)
        full_class_name = 'Admin' + "_" + str(class_name)
        query = weaviate_client.query.get(full_class_name, class_properties)
        print('the query', query)
        query = query.do()

        document_title_set = set()
        documents = query.get('data', {}).get('Get', {}).get(str(full_class_name), [])
        #print('the documents', documents)
        for document in documents:
            document_title = document.get('document_title')
            if document_title is not None:
                document_title_set.add(document_title)
        return list(document_title_set)
    
    except Exception as e:
            return {"error": str(e)}
    
@router.post("/get_classes/")  
async def delete_weaviate_class(data: VectorDBRequest,
    current_user: User = Depends(get_current_active_user)):
    try:
        weaviate_client = weaviate.Client("http://localhost:8080")
        username = current_user.username
        schema = weaviate_client.schema.get()
        classes = schema.get('classes', []) 
        prefix = str(username) + "_"
        prefix = prefix.capitalize()
        filtered_classes = [cls["class"].replace(prefix, "", 1) for cls in classes if cls["class"].startswith(prefix)] #[cls["class"] for cls in classes if cls["class"].startswith(prefix)]
        return filtered_classes
    
    except Exception as e:
            return {"error": str(e)}