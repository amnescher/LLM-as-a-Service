from fastapi import FastAPI
from .routers import db_functions, login_router, inference
from app.logging_config import setup_logger  # new import
logger = setup_logger() 
app = FastAPI()

app.include_router(login_router.router)
app.include_router(db_functions.router, prefix="/de_request", tags=["db_functions"])
app.include_router(inference.router, prefix="/llm_request", tags=["inference"])