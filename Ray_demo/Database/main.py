from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import func
from fastapi import FastAPI, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from pydantic import BaseModel
from typing import List, Dict
from typing import Optional
from passlib.context import CryptContext
from sqlalchemy import Boolean

from typing import Optional
from passlib.context import CryptContext
from sqlalchemy import Boolean
#from database import session_scope, User, Conversation, pwd_context, SessionLocal
#from database import session_scope

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


from functools import wraps

def role_required(role: str):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Simulate fetching user from the database
            with session_scope() as db:
                user_id = kwargs.get('user_id')  # Assuming user_id is passed to the function
                user = db.query(User).filter(User.id == user_id).first()
                if user and user.role == role:
                    return await func(*args, **kwargs)
                else:
                    raise HTTPException(status_code=403, detail="Permission Denied")
        return wrapper
    return decorator



class Input(BaseModel):
    username: Optional[str]
    password: Optional[str]
    content: Optional[str]
    conversation_number: Optional[int]
    conversation_name: Optional[str]
    user_id: Optional[int]
    prompt_token_number: Optional[int]
    gen_token_number: Optional[int]
    token_limit: Optional[int]



SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# database.py

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, func
from sqlalchemy.orm import declarative_base, sessionmaker
from pydantic import BaseModel
from typing import Optional
from passlib.context import CryptContext
from contextlib import contextmanager
from dotenv import load_dotenv
import os 

load_dotenv(os.getenv("ENV_PATH", "/home/ubuntu/falcon/kubeflow/llm-falcon-chatbot/API/.env"))
DB_DIR = os.getenv("DB_DIR", "/home/ubuntu/falcon/kubeflow/llm-falcon-chatbot")
db_path = os.path.join(DB_DIR, "test.db")
# Initialize password context for hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Define Database URL
DATABASE_URL = f"sqlite:///{db_path}"

# Create SQLAlchemy Base and SessionLocal
Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create the tables
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    prompt_token_number = Column(Integer, default=0)
    gen_token_number = Column(Integer, default=0)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    hashed_password = Column(String)
    disabled = Column(Boolean, default=False)
    token_limit = Column(Integer, default=1000)
    role = Column(String, default="User") 

class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    conversation_number = Column(Integer)
    content = Column(String)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    conversation_name = Column(String)

Base.metadata.create_all(bind=engine)


@contextmanager
def session_scope():
    """Provide a transactional scope around a series of operations."""
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        raise
    finally:
        db.close()

with session_scope() as db:
    admin_username = "admin"
    admin_password = "supersecurepassword"

    existing_admin = db.query(User).filter(User.username == admin_username).first()
    if not existing_admin:
        hashed_password = pwd_context.hash(admin_password)
        new_admin = User(username=admin_username, hashed_password=hashed_password, role="Admin")
        db.add(new_admin)
        db.commit()
        print(f"Admin {admin_username} created.")
    else:
        print(f"Admin {admin_username} already exists.")



oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    with session_scope() as db:
        user = db.query(User).filter(User.username == token_data.username).first()
        if user is None:
            raise credentials_exception
        return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


app = FastAPI()

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    with session_scope() as db:
        user = db.query(User).filter(User.username == form_data.username).first()
        if not user or not verify_password(form_data.password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.username}, expires_delta=access_token_expires
        )
        return {"access_token": access_token, "token_type": "bearer"}

@app.post("/add_user/")
@role_required("Admin")  # Only an Admin can add a user
def add_user(input: Input):
    with session_scope() as db:
        # Check if the username already exists
        existing_user = db.query(User).filter(User.username == input.username).first()
        if existing_user:
            db.close()
            raise HTTPException(
                status_code=400,
                detail="User already exists. Please choose another username.",
            )

        hashed_password = pwd_context.hash(input.password)
        user = User(
            username=input.username,
            hashed_password=hashed_password,
            prompt_token_number=0,
            gen_token_number=0,
        )

        # Set token_limit if provided
        if input.token_limit is not None:
            user.token_limit = input.token_limit

        db.add(user)
        db.commit()
        db.refresh(user)
        db.close()
        return {"user_id": user.id}
