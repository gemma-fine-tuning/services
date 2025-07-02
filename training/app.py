import os
import logging
from fastapi import FastAPI, HTTPException
from google.cloud import firestore
from google.cloud import run_v2
from google.cloud import storage
from schema import TrainRequest, JobSubmitResponse, JobStatusResponse
from job_manager import JobStateManager
import json
import uvicorn

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
db = firestore.Client(project=project_id)
job_manager = JobStateManager(db)

# These will not be configured in the os envvars because they are pretty much fixed to these two values
REGION = os.getenv("REGION", "us-central1")
JOB_NAME = os.getenv("TRAINING_JOB_NAME", "training-job")
GCS_CONFIG_BUCKET = os.getenv("GCS_CONFIG_BUCKET", "gemma-train-config")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "training"}


@app.post("/train", response_model=JobSubmitResponse)
async def start_training(request: TrainRequest):
    processed_dataset_id = request.processed_dataset_id
    base_model_id = request.training_config.base_model_id
    job_id = f"training_{processed_dataset_id}_{base_model_id.split('/')[-1]}"

    if request.export == "hfhub" and not request.hf_repo_id:
        raise HTTPException(
            status_code=400,
            detail="hf_repo_id is required when export is hfhub",
        )

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
