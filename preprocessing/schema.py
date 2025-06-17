from pydantic import BaseModel
from typing import Optional, Dict, List, Any


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


class PreprocessingConfig(BaseModel):
    field_mappings: Dict[str, str] = {}
    system_message: str = ""
    include_system: bool = True
    user_template: str = "{content}"
    test_size: float = 0.2
    train_test_split: bool = False
    normalize_whitespace: bool = True
    augmentation_config: Dict[str, Any] = {}


class PreprocessingRequest(BaseModel):
    dataset_source: str  # "upload", "huggingface", or "demo"
    # TODO: Can make this literal?
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
    dataset_type: str  # "raw" or "processed"
    # TODO: Can make this literal?


class PreviewRequest(BaseModel):
    dataset_source: str
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


class DemoDatasetResponse(BaseModel):
    datasets: Dict[str, str]
