from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from pydantic import BaseModel
from typing import Optional
import zipfile
import os
from io import BytesIO
import shutil

# Define the BaseModel class for your input data
class VDBaseInput(BaseModel):
    data_type: Optional[str] = None
    mode: Optional[str] = None
    data_path: Optional[str] = None
    collection: Optional[str] = None
    doc_name: Optional[str] = None
    collection_name: Optional[str] = None
    embedding_name: Optional[str] = None
    pdf_path: Optional[str] = None
    VDB_type: Optional[str] = None

async def process_pdf_file(file_data: bytes):
    # Process the PDF file
    # [Insert your code here to handle the PDF file]
    pass

async def extract_and_process_zip(file_data: bytes):
    try:
        with zipfile.ZipFile(BytesIO(file_data), 'r') as zip_ref:
            # Extract ZIP file
            tmp_dir = 'temp_dir'
            os.makedirs(tmp_dir, exist_ok=True)
            zip_ref.extractall(tmp_dir)

            # Process each file in the ZIP
            for filename in os.listdir(tmp_dir):
                if filename.endswith('.pdf'):
                    file_path = os.path.join(tmp_dir, filename)
                    with open(file_path, 'rb') as pdf_file:
                        process_pdf_file(pdf_file.read())

            # Clean up temporary directory
            shutil.rmtree(tmp_dir)
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid ZIP file")

app = FastAPI()

# Define a constant for maximum file size
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB, adjust as needed

@app.post("/VectorDataBase/")
async def create_upload_file(req: VDBaseInput = Depends(), file: Optional[UploadFile] = File(None)):
    if file:
        if file.size > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File size exceeds limit")
        file.file.seek(0)
        if file.content_type == 'application/pdf':
            process_pdf_file(await file.read())
            return {"filename": file.filename, "message": "PDF file processed"}
        elif file.content_type == 'application/zip':
            extract_and_process_zip(await file.read())
            return {"filename": file.filename, "message": "ZIP file processed"}
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
    else:
        return {"message": "No file uploaded"}
