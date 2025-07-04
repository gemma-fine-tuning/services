from pydantic import BaseModel
from typing import Optional, Dict, List, Any, Literal


class DatasetUploadResponse(BaseModel):
    dataset_id: str
    filename: str
    gcs_path: str
    size_bytes: int
    sample: List[Dict[str, Any]]
    columns: List[str]
    num_examples: int


class FieldMappingConfig(BaseModel):
    type: Literal["column", "template"]
    value: str


class BaseSplitConfig(BaseModel):
    type: Literal["hf_split", "manual_split", "no_split"]


class HFSplitConfig(BaseSplitConfig):
    type: Literal["hf_split"] = "hf_split"
    train_split: str
    test_split: str


class ManualSplitConfig(BaseSplitConfig):
    type: Literal["manual_split"] = "manual_split"
    sample_size: Optional[int] = None
    test_size: Optional[float] = None
    split: Optional[str] = None


class NoSplitConfig(BaseSplitConfig):
    type: Literal["no_split"] = "no_split"
    sample_size: Optional[int] = None
    split: Optional[str] = None


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
    dataset_name: str
    dataset_source: Literal["upload", "huggingface"]
    dataset_id: str
    dataset_subset: Optional[str] = "default"
    config: PreprocessingConfig


class ProcessingResult(BaseModel):
    processed_dataset_name: str
    dataset_path: str
    splits: Dict[str, int]  # split_name -> num_rows
    sample_comparison: Dict[str, Any]  # Only one sample from train split


class DatasetInfoResponse(BaseModel):
    dataset_id: str
    gcs_path: str
    size: int
    created: str
    sample: List[Dict[str, Any]]
    dataset_type: Literal["raw", "processed"]


class ValidationResponse(BaseModel):
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    total_samples: int
    valid_samples: int
