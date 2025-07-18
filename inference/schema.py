from pydantic import BaseModel
from typing import Any, Dict, List, Literal


class InferenceRequest(BaseModel):
    # HF Token must be provided for Gemma models
    hf_token: str
    # The user should specify this so we don't have to look it up
    storage_type: Literal["gcs", "hfhub"]
    job_id_or_repo_id: str
    prompt: str


class InferenceResponse(BaseModel):
    result: str


class BatchInferenceRequest(BaseModel):
    hf_token: str
    storage_type: Literal["gcs", "hfhub"]
    job_id_or_repo_id: str
    # A list of conversations, where each conversation is a list of messages
    messages: List[List[Dict[str, Any]]]


class BatchInferenceResponse(BaseModel):
    results: list[str]
