from fastapi import APIRouter

from app.api.v1.endpoints import pipelines

v1_router = APIRouter(prefix="/api/v1")
v1_router.include_router(pipelines.router)
