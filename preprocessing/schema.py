from pydantic import BaseModel
from typing import Optional, Dict, List, Any


class DatasetUploadResponse(BaseModel):
    dataset_id: str
    filename: str
    gcs_path: str
    size_bytes: int


class PreprocessingRequest(BaseModel):
    dataset_source: str  # "upload" or "standard"
    dataset_id: str
    sample_size: Optional[int] = None
    options: Dict[str, Any] = {}


class ProcessingResult(BaseModel):
    processed_dataset_id: str
    original_count: int
    processed_count: int
    train_count: Optional[int] = None
    test_count: Optional[int] = None
    train_gcs_path: Optional[str] = None
    test_gcs_path: Optional[str] = None
    gcs_path: Optional[str] = None
    sample_comparison: Dict[str, Any]


class DatasetInfoResponse(BaseModel):
    dataset_id: str
    gcs_path: str
    size: int
    created: str
    sample: List[Dict[str, Any]]


class PreviewRequest(BaseModel):
    dataset_source: str
    dataset_id: str
    sample_size: int = 5
    options: Dict[str, Any] = {}
