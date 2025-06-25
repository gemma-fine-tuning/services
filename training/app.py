import os
import logging
from fastapi import FastAPI, HTTPException
from huggingface_hub import login
from schema import (
    TrainRequest,
    TrainResponse,
    InferenceRequest,
    InferenceResponse,
)
from services import TrainingService, run_inference

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


@app.post("/train", response_model=TrainResponse)
async def train(request: TrainRequest):
    """Start training with given configuration"""
    try:
        if request.export == "hfhub" and not request.hf_repo_id:
            raise HTTPException(
                status_code=400, detail="hf_repo_id is required for Hugging Face export"
            )

        login_hf(request.hf_token)
        training_service = TrainingService.from_provider(
            request.training_config.provider
        )
        return training_service.run_training(request)
    except Exception as e:
        logging.error(f"Training failed with error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


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


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8081))
    uvicorn.run(app, host="0.0.0.0", port=port)
