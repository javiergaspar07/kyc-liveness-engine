from contextlib import asynccontextmanager
from fastapi import FastAPI
from api.routes import (
    passive_v1, 
    active_v2, 
    static_authentication,
    dynamic_authentication
)
from core.db import init_db
from ml.models_manager import ml_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP LOGIC ---
    init_db()
    ml_manager.load_models()
    
    # The application yields and runs here (listening for WebSockets)
    yield 
    
    # --- SHUTDOWN LOGIC ---
    ml_manager.unload_models()

app = FastAPI(title="KYC Liveness Engine", version="2.0", lifespan=lifespan)

# Conectamos los módulos externos a la aplicación principal
app.include_router(passive_v1.router)
app.include_router(active_v2.router)
app.include_router(static_authentication.router)
app.include_router(dynamic_authentication.router)