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


class BaseSplitConfig(BaseModel):
    type: Literal["hf_split", "manual_split", "no_split"]


class HFSplitConfig(BaseSplitConfig):
    type: Literal["hf_split"] = "hf_split"
    splits: List[str]


class ManualSplitConfig(BaseSplitConfig):
    type: Literal["manual_split"] = "manual_split"
    sample_size: Optional[int] = None
    test_size: Optional[float] = None


class NoSplitConfig(BaseSplitConfig):
    type: Literal["no_split"] = "no_split"
    sample_size: Optional[int] = None


class AugmentationSetupConfig(BaseModel):
    """
    Currently we do not support customising parameters for augmentation methods.
    This is supported in the backend but we do not expose it in the API for now for simplicity.
    """

    augmentation_factor: float = 1.5
    use_eda: Optional[bool] = False
    use_back_translation: Optional[bool] = False
    use_paraphrasing: Optional[bool] = False
    use_synthesis: Optional[bool] = False
    gemini_api_key: Optional[str] = None
    synthesis_ratio: Optional[float] = None
    custom_prompt: Optional[str] = None


class PreprocessingConfig(BaseModel):
    field_mappings: Dict[str, FieldMappingConfig] = {}
    normalize_whitespace: bool = True
    augmentation_config: Optional[AugmentationSetupConfig] = None
    split_config: Optional[HFSplitConfig | ManualSplitConfig | NoSplitConfig] = None


class PreprocessingRequest(BaseModel):
    dataset_source: Literal["upload", "huggingface"]
    dataset_id: str
    config: PreprocessingConfig


class ProcessingResult(BaseModel):
    processed_dataset_id: str
    original_count: int
    processed_count: int
    splits: Dict[
        str, Dict[str, Any]
    ]  # split_name -> {count, gcs_path, processed_count}
    sample_comparison: Dict[str, Any]  # Only one sample from train split


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
