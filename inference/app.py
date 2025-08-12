import os
import logging
import json
import base64
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.concurrency import run_in_threadpool
from google.cloud import firestore
from huggingface_hub import login
from schema import (
    InferenceRequest,
    InferenceResponse,
    BatchInferenceRequest,
    BatchInferenceResponse,
    EvaluationRequest,
    EvaluationResponse,
)
from base import run_inference, run_batch_inference, run_evaluation
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


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "inference"}


# Auth dependency for API Gateway
def get_current_user_id(request: Request) -> str:
    """
    Extract user ID from X-Apigateway-Api-Userinfo header set by API Gateway.
    The gateway requires the JWT to contain iss (issuer), sub (subject), aud (audience), iat (issued at), exp (expiration time) claims
    API Gateway will send the authentication result in the X-Apigateway-Api-Userinfo to the backend API whcih contains the base64url encoded content of the JWT payload.
    In this case, the gateway will override the original Authorization header with this X-Apigateway-Api-Userinfo header.

    Args:
        request: FastAPI Request object containing headers

    Returns:
        str: User ID extracted from JWT claims

    Raises:
        HTTPException: 401 if userinfo header is missing or invalid
    """
    userinfo_header = request.headers.get("X-Apigateway-Api-Userinfo")
    if not userinfo_header:
        raise HTTPException(
            status_code=401,
            detail="Missing authentication userinfo. Ensure requests go through API Gateway.",
        )

    try:
        # Decode base64url encoded JWT payload
        # Add padding if needed for proper base64 decoding
        missing_padding = len(userinfo_header) % 4
        if missing_padding:
            userinfo_header += "=" * (4 - missing_padding)

        decoded_bytes = base64.urlsafe_b64decode(userinfo_header)
        claims = json.loads(decoded_bytes.decode("utf-8"))

        user_id = claims.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=401, detail="User ID not found in authentication claims"
            )
        return user_id
    except (json.JSONDecodeError, base64.binascii.Error, UnicodeDecodeError) as e:
        raise HTTPException(
            status_code=401, detail=f"Invalid authentication userinfo format: {str(e)}"
        )


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
async def inference(
    request: InferenceRequest,
    current_user_id: str = Depends(get_current_user_id),
):
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


@app.post("/inference/batch", response_model=BatchInferenceResponse)
async def batch_inference(
    request: BatchInferenceRequest,
    current_user_id: str = Depends(get_current_user_id),
):
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


@app.post("/evaluation", response_model=EvaluationResponse)
async def evaluation(
    request: EvaluationRequest,
    current_user_id: str = Depends(get_current_user_id),
):
    """Run evaluation of a fine-tuned model on a dataset"""
    try:
        # Verify dataset ownership before evaluating
        try:
            doc = db.collection("processed_datasets").document(request.dataset_id).get()
            if not doc.exists or doc.to_dict().get("user_id") != current_user_id:
                raise HTTPException(status_code=404, detail="Dataset not found")
        except HTTPException:
            raise
        except Exception as e:
            logging.error(f"Failed to verify dataset ownership: {e}")
            raise HTTPException(
                status_code=500, detail="Failed to verify dataset ownership"
            )

        login_hf(request.hf_token)
        result = await run_in_threadpool(
            run_evaluation,
            request.adapter_path,
            request.base_model_id,
            request.dataset_id,
            request.task_type,
            request.metrics,
            request.max_samples,
            request.num_sample_results or 3,
        )
        return {
            "metrics": result["metrics"],
            "samples": result["samples"],
            "num_samples": result["num_samples"],
            "dataset_id": result["dataset_id"],
        }
    except FileNotFoundError as e:
        logging.error(f"Resource not found: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        logging.error(f"Invalid request: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"Evaluation failed with error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
