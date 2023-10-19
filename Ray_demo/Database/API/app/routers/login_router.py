
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from app.models import Token
from app.services.authentication import authenticate_user, create_access_token
from datetime import timedelta
import os
from sqlalchemy.orm import Session
from dotenv import load_dotenv
from app.logging_config import setup_logger  # Logging setup
from app.database import get_db

# Set up logging
logger = setup_logger()

# Load environment variables
load_dotenv(os.getenv("ENV_PATH", "/home/ubuntu/falcon/kubeflow/llm-falcon-chatbot/API/.env"))

router = APIRouter()

# Retrieve configuration from environment variables
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", default="30"))

# FastAPI route to handle login and token issuance
@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """
    Generates a JWT access token for a user upon successful login.
    
    Parameters:
    - form_data: An instance of OAuth2PasswordRequestForm containing the login form data.
    - db: An instance of the database Session.
    
    Returns:
    - A dictionary containing the access token and token type.
    """
    # Authenticate the user based on the form data
    user = authenticate_user(db, form_data.username, form_data.password)
    
    # If authentication fails, log the event and raise an HTTP Exception
    if not user:
        logger.warning(f"login_for_access_token: Invalid credentials for user {form_data.username}")  # Log warning
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Calculate token expiration time
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    # Generate the access token
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    # Log successful login
    logger.info(f"login_for_access_token: valid credentials for user {form_data.username}")  
    
    # Return the access token
    return {"access_token": access_token, "token_type": "bearer"}
