from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from app.models import Token
from app.services.authentication import authenticate_user, create_access_token
from datetime import timedelta
import os
from sqlalchemy.orm import Session
from dotenv import load_dotenv
from app.logging_config import setup_logger  # new import
from app.database import get_db 

# Set up logger
logger = setup_logger()

load_dotenv("/home/ubuntu/falcon/kubeflow/llm-falcon-chatbot/API/.env")

router = APIRouter()

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")

ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", default="30"))


@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        logger.warning(f"login_for_access_token: Invalid credentials for user {form_data.username}")  # log a warning
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    logger.info(f"login_for_access_token: valid credentials for user {form_data.username}")  
    return {"access_token": access_token, "token_type": "bearer"}
