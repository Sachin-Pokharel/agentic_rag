from fastapi import APIRouter
from api.endpoints import agent_rag, upload


api_router = APIRouter()
api_router.include_router(upload.router, tags=['upload'])
api_router.include_router(agent_rag.router, tags=['agent_rag'])