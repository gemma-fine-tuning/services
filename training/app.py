import os
import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks
from google.cloud import firestore
from huggingface_hub import login
from schema import (
    TrainRequest,
    InferenceRequest,
    InferenceResponse,
    SubmitTrainResponse,
    JobStatusResponse,
)
from services import TrainingService, run_inference
from job_manager import JobStateManager, JobTracker

app = FastAPI(
    title="Gemma Training Service",
    version="1.0.0",
    description="Training backend (Cloud Run GPU) for fine-tuning LLMs with various methods",
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",  # Add timestamps to logs
)

# Training metrics will be available via REST API endpoints
logging.info("✅ Training service ready")

# Initialize Firestore client and job state manager
project_id = os.getenv("PROJECT_ID")
# NOTE: Project ID should always be provided else will fail
if not project_id:
    raise ValueError("PROJECT_ID environment variable must be set for Firestore client")
db = firestore.Client(project=project_id)
job_manager = JobStateManager(db)


def login_hf(hf_token: str = None):
    """
    Login to Hugging Face.
    For now we support env variable for dev but in prod we will just raise an error.
    Login is required for pushing and pulling models since Gemma3 is a gated model.
    """
    token = hf_token or os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        login(token=token)
        logging.info("Logged into Hugging Face")
    else:
        logging.warning("HF Token not provided. Hugging Face login skipped.")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "training"}


@app.post("/train", response_model=SubmitTrainResponse)
async def train(request: TrainRequest, background_tasks: BackgroundTasks):
    """Submit training job and return job ID immediately"""
    # Validate request
    if request.export == "hfhub" and not request.hf_repo_id:
        raise HTTPException(
            status_code=400, detail="hf_repo_id is required for Hugging Face export"
        )

    # Compute job ID
    processed_dataset_id = request.processed_dataset_id
    base_model_id = request.training_config.base_model_id
    job_id = f"training_{processed_dataset_id}_{base_model_id.split('/')[-1]}"

    # Setup HF login
    login_hf(request.hf_token)

    # Create job in Firestore with QUEUED status
    job_manager.create_job(job_id, processed_dataset_id, base_model_id)

    # Schedule background training task
    background_tasks.add_task(process_training_job, request.model_dump(), job_id)

    return SubmitTrainResponse(job_id=job_id)


@app.post("/inference/{job_id}", response_model=InferenceResponse)
async def inference(job_id: str, request: InferenceRequest):
    """Run inference using a trained adapter"""
    prompt = request.prompt
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")

    try:
        login_hf(request.hf_token)
        output = run_inference(job_id, prompt, request.storage_type)
        return {"result": output}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Adapter not found")
    except Exception as e:
        logging.error(f"Inference failed with error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/train/{job_id}/status", response_model=JobStatusResponse)
async def train_status(job_id: str):
    """Retrieve status of a training job"""
    # Get job status from JobStateManager
    job_data = job_manager.get_job_status_dict(job_id)
    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatusResponse(
        status=job_data.get("status"),
        wandb_url=job_data.get("wandb_url"),
        adapter_path=job_data.get("adapter_path"),
        base_model_id=job_data.get("base_model_id"),
        error=job_data.get("error"),
    )


def process_training_job(request_data: dict, job_id: str):
    """Background task to execute training with granular status updates"""
    try:
        # Use JobTracker context manager for automatic status transitions
        with JobTracker(job_manager, job_id) as tracker:
            # Execute training with job tracker (WandB and status handled in training service)
            train_req = TrainRequest.model_validate(request_data)
            service = TrainingService.from_provider(train_req.training_config.provider)
            result = service.run_training(train_req, job_tracker=tracker)

            # Mark as completed
            tracker.completed(result.adapter_path)

    except Exception as e:
        logging.error(f"Training job {job_id} failed: {str(e)}", exc_info=True)
        # Exception handling is done by JobTracker context manager


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8081))
    uvicorn.run(app, host="0.0.0.0", port=port)
