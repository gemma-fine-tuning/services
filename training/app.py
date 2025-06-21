import os
import logging
from fastapi import FastAPI, HTTPException
from huggingface_hub import login
from training.schema import (
    TrainRequest,
    TrainResponse,
    InferenceRequest,
    InferenceResponse,
)

from services import run_inference, run_training

app = FastAPI(
    title="Gemma Training Service",
    version="1.0.0",
    description="Training backend for fine-tuning LLMs with various methods",
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",  # Add timestamps to logs
)

hf_token = os.environ.get("HUGGINGFACE_TOKEN")
if hf_token:
    login(token=hf_token)
    logging.info("Logged into Hugging Face")
else:
    logging.warning("HUGGINGFACE_TOKEN not provided. Hugging Face login skipped.")

# Training metrics will be available via REST API endpoints
logging.info("✅ Training service ready")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "training"}


@app.post("/train", response_model=TrainResponse)
async def train(request: TrainRequest):
    """Start training with given configuration"""
    try:
        result = run_training(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/inference/{job_id}", response_model=InferenceResponse)
async def inference(job_id: str, payload: InferenceRequest):
    """Run inference using a trained adapter"""
    prompt = payload.prompt
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")
    try:
        output = run_inference(job_id, prompt)
        return {"result": output}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Adapter not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8081))
    uvicorn.run(app, host="0.0.0.0", port=port)
