import logging

from fastapi import FastAPI

from app.api.v1.router import v1_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)

app = FastAPI(
    title="Voice Pipeline API",
    description="Voice pipelines optimized for intent error rate (IER)",
    version="0.1.0",
)

app.include_router(v1_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
