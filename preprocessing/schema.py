from pydantic import BaseModel
from typing import Optional, Dict, List, Any, Literal


class DatasetUploadResponse(BaseModel):
    dataset_id: str
    filename: str
    gcs_path: str
    size_bytes: int


class DatasetAnalysisResponse(BaseModel):
    dataset_id: str
    total_samples: int
    columns: List[str]
    sample_data: List[Dict[str, Any]]
    column_info: Dict[str, Any]
    format_type: str


class FieldMappingConfig(BaseModel):
    type: Literal["column", "template"]
    value: str


class PreprocessingConfig(BaseModel):
    field_mappings: Dict[str, FieldMappingConfig] = {}
    test_size: float = 0.2
    train_test_split: bool = False
    normalize_whitespace: bool = True


class PreprocessingRequest(BaseModel):
    dataset_source: Literal["upload", "huggingface"]
    dataset_id: str
    sample_size: Optional[int] = None
    config: PreprocessingConfig


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
    dataset_type: Literal["raw", "processed"]


class PreviewRequest(BaseModel):
    dataset_source: Literal["upload", "huggingface"]
    dataset_id: str
    sample_size: int = 5
    config: PreprocessingConfig


class PreviewResponse(BaseModel):
    original_samples: List[Dict[str, Any]]
    converted_samples: List[Dict[str, Any]]
    conversion_success: bool
    samples_converted: int
    samples_failed: int


class ValidationResponse(BaseModel):
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    total_samples: int
    valid_samples: int
