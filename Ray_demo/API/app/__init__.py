from fastapi import FastAPI
from .routers import login_router, inference_router
from app.logging_config import setup_logger  # new import

logger = setup_logger() 
app = FastAPI()

app.include_router(login_router.router)
app.include_router(inference_router.router, prefix="/inference", tags=["inference"])
