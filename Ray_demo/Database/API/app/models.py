from pydantic import BaseModel
from typing import Optional

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