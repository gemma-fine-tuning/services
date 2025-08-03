import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from google.cloud import firestore
from huggingface_hub import login
from schema import (
    InferenceRequest,
    InferenceResponse,
    BatchInferenceRequest,
    BatchInferenceResponse,
)
from base import run_inference, run_batch_inference
from typing import Optional

app = FastAPI(
    title="Gemma Inference Service",
    version="1.0.0",
    description="Inference service for running inference on trained models",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Initialize Firestore client
project_id = os.getenv("PROJECT_ID")
if not project_id:
    raise ValueError("PROJECT_ID environment variable must be set for Firestore client")
db = firestore.Client(project=project_id)

logging.info("✅ Inference service ready")


def login_hf(hf_token: Optional[str]):
    """
    Login to Hugging Face.
    For now we support env variable for dev but in prod we will just raise an error.
    Login is required for pushing and pulling models since Gemma3 is a gated model.
    """
    token = hf_token
    if token:
        login(token=token)
        logging.info("Logged into Hugging Face")
    else:
        logging.warning("HF Token not provided. Hugging Face login skipped.")


@app.post("/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest):
    """Run inference using a trained adapter"""
    try:
        login_hf(request.hf_token)
        output = await run_in_threadpool(
            run_inference,
            request.adapter_path,
            request.base_model_id,
            request.prompt,
        )
        return {"result": output}
    except FileNotFoundError:
        logging.error(f"Adapter {request.adapter_path} not found")
        raise HTTPException(status_code=404, detail="Adapter not found")
    except Exception as e:
        logging.error(f"Inference failed with error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_inference", response_model=BatchInferenceResponse)
async def batch_inference(request: BatchInferenceRequest):
    """Run batch inference using a trained adapter"""
    messages = request.messages
    if not messages or not isinstance(messages, list) or len(messages) == 0:
        raise HTTPException(status_code=400, detail="messages (list) is required")
    try:
        login_hf(request.hf_token)
        outputs = await run_in_threadpool(
            run_batch_inference,
            request.adapter_path,
            request.base_model_id,
            messages,
        )
        return {"results": outputs}
    except FileNotFoundError:
        logging.error(f"Adapter {request.adapter_path} not found")
        raise HTTPException(status_code=404, detail="Adapter not found")
    except Exception as e:
        logging.error(f"Batch inference failed with error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "inference"}


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Gemma Inference Service", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
