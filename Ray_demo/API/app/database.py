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
#from app.logging_config import setup_logger


# Environment and DB setup
load_dotenv("/home/ubuntu/falcon/kubeflow/llm-falcon-chatbot/API/.env") # Make sure this path is correct
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./test.db")  # Provide a fallback if the env variable is missing
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
DB_SERVICE_URL = os.getenv("DB_SERVICE_URL")  # Make sure this is used somewhere in your application


BASE_DIR = "/home/ubuntu/falcon/kubeflow/llm-falcon-chatbot"
# BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Gets the absolute path of the current file's directory
db_path = os.path.join(BASE_DIR, "test.db")

# Check if the database file exists
if not os.path.exists(db_path):
    raise Exception(f"Database file not found at {db_path}")

DATABASE_URL = f"sqlite:///{db_path}"

Base = declarative_base()

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    prompt_token_number = Column(Integer, default=0)  
    gen_token_number = Column(Integer, default=0)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    hashed_password = Column(String)


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    conversation_number = Column(Integer)  # Add conversation number column
    content = Column(String)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    conversation_name = Column(String)


engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()  # SessionLocal should be your database session class
    try:
        yield db
    finally:
        db.close()
