# app/api/routes/ingest.py
import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
from app.api.dependencies import get_ingestion_service
from app.core.logging import get_logger
import redis.asyncio as redis
from app.core.config import get_settings

router = APIRouter(prefix="/ingest", tags=["ingestion"])
settings = get_settings()

logger = get_logger(__name__)
async def _run_ingestion(file_path: Path, job_id: str):
    """
    Background task: runs the full ingestion pipeline.
    Updates job status in Redis so the client can poll.
    """
    r = redis.from_url(settings.REDIS_URL, decode_responses=True)
    try:
        await r.setex(f"job:{job_id}", 3600, "processing")
        service = get_ingestion_service()
        result = await service.ingest_file(file_path)
        await r.setex(f"job:{job_id}", 3600, f"completed:{result['chunks']} chunks")
        logger.info(f"Job {job_id} completed: {result}")
    except Exception as e:
        await r.setex(f"job:{job_id}", 3600, f"failed:{str(e)}")
        logger.error(f"Job {job_id} failed: {e}")
    finally:
        await r.close()


@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload a document and trigger background ingestion.
    Returns immediately with a job_id for status polling.
    """
    # Validate file type
    allowed = {".pdf", ".txt", ".md"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported file type: {suffix}")

    # Save to disk
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    file_path = upload_dir / f"{uuid.uuid4()}{suffix}"

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Trigger background processing
    job_id = str(uuid.uuid4())
    background_tasks.add_task(_run_ingestion, file_path, job_id)

    return {
        "job_id": job_id,
        "status": "queued",
        "message": "Document uploaded. Poll /ingest/jobs/{job_id} for status.",
    }


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Poll ingestion job status."""
    r = redis.from_url(settings.REDIS_URL, decode_responses=True)
    status = await r.get(f"job:{job_id}")
    await r.close()
    if not status:
        raise HTTPException(404, "Job not found")
    return {"job_id": job_id, "status": status}