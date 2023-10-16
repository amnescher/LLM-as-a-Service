from datetime import datetime, timedelta
import os
from jose import JWTError, jwt
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker, Session,declarative_base
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql import func
from fastapi import HTTPException, status
from dotenv import load_dotenv
from passlib.context import CryptContext
from app.logging_config import setup_logger
from app.database import get_db,User


# Environment and DB setup
load_dotenv("/home/ubuntu/falcon/kubeflow/llm-falcon-chatbot/API/.env") # Make sure this path is correct
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./test.db")  # Provide a fallback if the env variable is missing
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
DB_SERVICE_URL = os.getenv("DB_SERVICE_URL")  # Make sure this is used somewhere in your application
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
logger = setup_logger()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def authenticate_user(db: Session, username: str, password: str) -> User:
    user = None
    try:
        print("input username")
        user = db.query(User).filter(User.username == username).first()
    except SQLAlchemyError as e:
        logger.error(f"Database error during authentication: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Internal server error") from e
    
    if user is None or not verify_password(password, user.hashed_password):
        return False
    return user


def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt
