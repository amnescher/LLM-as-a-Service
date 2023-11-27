from pydantic import BaseModel
from typing import Optional
from typing import Any, List

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: str = None

class LoginUser(BaseModel):
    username: str
    disabled: bool = None

class UserInDB(LoginUser):  
    hashed_password: str

class Data(BaseModel): 
    prompt: str

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

class InferenceRequest(BaseModel):
    username: Optional[str]
    prompt: Optional[str]
    memory: Optional[bool]
    conversation_number: Optional[int]
    AI_assistance: Optional[bool]
    collection_name: Optional[str]
    llm_model: Optional[str] = "Llama_70b"

class DataBaseRequest(BaseModel):
    content: Optional[str]
    conversation_number: Optional[int]
    conversation_name: Optional[str]
    user_id: Optional[int]
    prompt_token_number: Optional[int]
    gen_token_number: Optional[int]
    token_limit: Optional[int]

class VectorDBRequest(BaseModel):
    username: str 
    collection_name: Optional[str] 
    mode: str = "add_to_collection"
    vectorDB_type: Optional[str] = "Weaviate"
    file_path: Optional[str] = None