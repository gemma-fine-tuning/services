from pydantic import BaseModel
from typing import Literal


class InferenceRequest(BaseModel):
    # HF Token must be provided for Gemma models
    hf_token: str
    # The user should specify this so we don't have to look it up
    storage_type: Literal["gcs", "hfhub"]
    prompt: str


class InferenceResponse(BaseModel):
    result: str
