from pydantic import BaseModel
from typing import Literal


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
    # English lesson corner: Note that this is "prompts" not "prompt"!
    prompts: list[str]


class BatchInferenceResponse(BaseModel):
    results: list[str]
