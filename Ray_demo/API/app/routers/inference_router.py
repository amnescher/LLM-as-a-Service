from fastapi import APIRouter, Depends, HTTPException, status
from app.models import User, Data
from app.depencencies.security import get_current_active_user

router = APIRouter()

@router.post("/")
async def create_inference(data: Data, current_user: User = Depends(get_current_active_user)):
    # If needed, add your logic here to handle the 'data' (e.g., process it, pass it to a model for inference, etc.)
    # Right now, it just returns the received data and the current user's username.
    return {"username": current_user.username, "data": data}
