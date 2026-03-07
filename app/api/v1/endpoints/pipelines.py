from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from app.core.exceptions import PipelineNotFoundError, PipelineNotReadyError
from app.dependencies import get_pipeline_service
from app.schemas.invocation import InvokeRequest, InvokeResponse
from app.schemas.pipeline import (
    PipelineCreateRequest,
    PipelineDetailResponse,
    PipelineListResponse,
)
from app.services.pipeline_service import PipelineService

router = APIRouter(prefix="/pipelines", tags=["pipelines"])


@router.get("", response_model=PipelineListResponse)
async def list_pipelines(
    service: PipelineService = Depends(get_pipeline_service),
):
    return await service.list_pipelines()


@router.post("", response_model=PipelineDetailResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_pipeline(
    payload: PipelineCreateRequest,
    background_tasks: BackgroundTasks,
    service: PipelineService = Depends(get_pipeline_service),
):
    return await service.create_pipeline(payload, background_tasks)


@router.get("/{pipeline_id}", response_model=PipelineDetailResponse)
async def get_pipeline(
    pipeline_id: str,
    service: PipelineService = Depends(get_pipeline_service),
):
    try:
        return await service.get_pipeline(pipeline_id)
    except PipelineNotFoundError:
        raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_id} not found")


@router.post("/{pipeline_id}/invoke", response_model=InvokeResponse)
async def invoke_pipeline(
    pipeline_id: str,
    request: InvokeRequest,
    service: PipelineService = Depends(get_pipeline_service),
):
    try:
        return await service.invoke_pipeline(pipeline_id, request)
    except PipelineNotFoundError:
        raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_id} not found")
    except PipelineNotReadyError as e:
        raise HTTPException(
            status_code=409, detail=f"Pipeline is not ready (status={e.status})"
        )
