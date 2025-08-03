from pydantic import BaseModel
from typing import Any, Dict, List


class InferenceRequest(BaseModel):
    hf_token: str  # HF Token must be provided for Gemma models
    # Path to the adapter (can be local path, GCS path, or HF Hub repo ID)
    adapter_path: str
    # Base model ID to use for tokenizer and model class selection
    base_model_id: str
    # A single text message
    prompt: str


class InferenceResponse(BaseModel):
    result: str


class BatchInferenceRequest(BaseModel):
    hf_token: str
    # Path to the adapter (can be local path, GCS path, or HF Hub repo ID)
    adapter_path: str
    # Base model ID to use for tokenizer and model class selection
    base_model_id: str
    # A list of conversations, where each conversation is a list of messages
    messages: List[List[Dict[str, Any]]]


class BatchInferenceResponse(BaseModel):
    results: list[str]
