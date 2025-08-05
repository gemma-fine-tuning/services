import os
import logging
from fastapi import FastAPI, HTTPException, Query
from google.cloud import run_v2
from google.cloud import storage
from datetime import timedelta, datetime, timezone
from schema import (
    TrainRequest,
    JobSubmitResponse,
    JobStatusResponse,
    JobListResponse,
    JobListEntry,
    DownloadUrlResponse,
)
import json
import uvicorn
import hashlib
from job_manager import JobStateManager, JobMetadata, JobStatus

app = FastAPI(
    title="Gemma Training Service",
    version="1.0.0",
    description="Training backend (Cloud Run GPU) for fine-tuning LLMs with various methods",
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logging.info("✅ Training service ready")

project_id = os.getenv("PROJECT_ID")
if not project_id:
    raise ValueError("PROJECT_ID environment variable must be set for Firestore client")
job_manager = JobStateManager(project_id)

# These will not be configured in the os envvars because they are pretty much fixed to these two values
REGION = os.getenv("REGION", "us-central1")
JOB_NAME = os.getenv("TRAINING_JOB_NAME", "training-job")
GCS_CONFIG_BUCKET = os.getenv("GCS_CONFIG_BUCKET_NAME", "gemma-train-config")
GCS_EXPORT_BUCKET = os.getenv("GCS_EXPORT_BUCKET_NAME", "gemma-export-bucket")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "training"}


def make_job_id(
    processed_dataset_id: str, base_model_id: str, request: TrainRequest
) -> str:
    """
    Creates a job ID for a training job.

    This has the following properties:
    - IDs are only the same if the processed dataset ID, base model ID, and request are the same
    - IDs are deterministic, i.e. same request will always produce the same ID
    - IDs are short because we truncate the hash to 8 characters

    This is desired because the exact same job will overwrite the previous one and avoid confusion and resource waste.
    This also makes tracking in firestore consistent.

    Args:
        processed_dataset_id (str): dataset_name, this might contain spaces we need to replace them first
        base_model_id (str): The ID of the base model
        request (TrainRequest): The request object

    Returns:
        str: The job ID
    """
    base = f"training_{processed_dataset_id.replace(' ', '-')}_{base_model_id.split('/')[-1]}"
    request_str = json.dumps(request.model_dump(), sort_keys=True)
    short_hash = hashlib.sha256(request_str.encode()).hexdigest()[:8]
    return f"{base}_{short_hash}"


@app.get("/jobs", response_model=JobListResponse)
async def list_jobs():
    """List all jobs with job_id and job_name."""
    try:
        jobs = job_manager.list_jobs()
        # Only return jobs that have a job_id
        job_entries = [JobListEntry(**job) for job in jobs if job.get("job_id")]
        return JobListResponse(jobs=job_entries)
    except Exception as e:
        logging.error(f"Failed to list jobs: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to list jobs")


@app.post("/train", response_model=JobSubmitResponse)
async def start_training(request: TrainRequest):
    processed_dataset_id = request.processed_dataset_id
    base_model_id = request.training_config.base_model_id
    job_id = make_job_id(processed_dataset_id, base_model_id, request)

    if (
        request.export_config.destination == "hfhub"
        and not request.export_config.hf_repo_id
    ):
        raise HTTPException(
            status_code=400,
            detail="hf_repo_id is required when export is hfhub",
        )

    # Immediately create job record in Firestore
    job_metadata = JobMetadata(
        job_id=job_id,
        status=JobStatus.QUEUED,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        processed_dataset_id=request.processed_dataset_id,
        base_model_id=request.training_config.base_model_id,
        job_name=request.job_name,
        modality=request.training_config.modality,
    )
    job_manager.ensure_job_document_exists(job_id, job_metadata)

    # Upload config to GCS
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_CONFIG_BUCKET)
        blob = bucket.blob(f"{job_id}.json")
        blob.upload_from_string(
            json.dumps(request.model_dump()), content_type="application/json"
        )
        logging.info(
            f"Uploaded training config to gs://{GCS_CONFIG_BUCKET}/{job_id}.json"
        )
    except Exception as e:
        logging.error(f"Failed to upload config to GCS: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to upload config to GCS: {str(e)}"
        )

    try:
        client = run_v2.JobsClient()
        job_name = f"projects/{project_id}/locations/{REGION}/jobs/{JOB_NAME}"
        run_request = run_v2.RunJobRequest(
            name=job_name,
            overrides=run_v2.RunJobRequest.Overrides(
                container_overrides=[
                    run_v2.RunJobRequest.Overrides.ContainerOverride(
                        env=[run_v2.EnvVar(name="JOB_ID", value=job_id)]
                    )
                ]
            ),
        )
        _ = client.run_job(request=run_request)

        # NOTE: We don't need to wait for the job to complete
        # because the job will be tracked in firestore
        # If needed you can do response = operation.result()

        return JobSubmitResponse(job_id=job_id)
    except Exception as e:
        logging.error(f"Failed to start training job: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to start training job: {str(e)}"
        )


@app.get("/get-download-url", response_model=DownloadUrlResponse)
async def generate_signed_url(
    file_path: str = Query(..., description="Path to the file in GCS bucket"),
):
    """
    Generates a signed URL for a GCS object.
    Expects a 'file_path' query parameter, e.g.,
    /get-download-url?file_path=gguf_models/your_model.gguf
    """
    if not file_path:
        raise HTTPException(status_code=400, detail="Missing 'file_path' parameter")

    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_EXPORT_BUCKET)
        blob = bucket.blob(file_path)

        if not blob.exists():
            raise HTTPException(status_code=404, detail="File not found")

        # Generate a signed URL that is valid for 15 minutes
        expiration_time = datetime.now(timezone.utc) + timedelta(minutes=15)
        signed_url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(minutes=15),
            method="GET",
        )

        return DownloadUrlResponse(
            download_url=signed_url,
            filename=os.path.basename(file_path),
            expires_at=expiration_time.isoformat(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to generate signed URL: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs/{job_id}/download/gguf", response_model=DownloadUrlResponse)
async def download_gguf_file(job_id: str):
    """
    Generate a signed URL for downloading the GGUF file of a specific job.
    This is a convenience endpoint that automatically finds the GGUF file path from job status.
    """
    try:
        # Get job status to find GGUF path
        job_data = job_manager.get_job_status_dict(job_id)
        if not job_data:
            raise HTTPException(status_code=404, detail="Job not found")

        gguf_path = job_data.get("gguf_path")
        if not gguf_path:
            raise HTTPException(
                status_code=404,
                detail="No GGUF file available for this job. Check if include_gguf was enabled during training.",
            )

        # Extract the blob path from the GCS URL
        # gguf_path format: gs://bucket/gguf_models/job_123/model.gguf
        if not gguf_path.startswith("gs://"):
            raise HTTPException(status_code=500, detail="Invalid GGUF path format")

        # Remove gs://bucket/ prefix to get blob path
        path_parts = gguf_path.replace("gs://", "").split("/", 1)
        if len(path_parts) != 2:
            raise HTTPException(status_code=500, detail="Invalid GGUF path format")

        bucket_name, blob_path = path_parts

        # Verify it's the expected bucket
        if bucket_name != GCS_EXPORT_BUCKET:
            raise HTTPException(
                status_code=500, detail="GGUF file in unexpected bucket"
            )

        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_EXPORT_BUCKET)
        blob = bucket.blob(blob_path)

        if not blob.exists():
            raise HTTPException(
                status_code=404, detail="GGUF file not found in storage"
            )

        # Generate a signed URL that is valid for 1 hour
        expiration_time = datetime.now(timezone.utc) + timedelta(hours=1)
        signed_url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(hours=1),
            method="GET",
        )

        return DownloadUrlResponse(
            download_url=signed_url,
            filename=os.path.basename(blob_path),
            expires_at=expiration_time.isoformat(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.error(
            f"Failed to generate GGUF download URL for job {job_id}: {str(e)}"
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    job_data = job_manager.get_job_status_dict(job_id)
    if not job_data:
        logging.error(f"Job not found: {job_id}")
        raise HTTPException(status_code=404, detail="Job not found")
    return job_data


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
