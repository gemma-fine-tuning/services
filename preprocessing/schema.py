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
    type: Literal["column", "template", "image"]
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
    dataset_subset: str = "default"
    config: PreprocessingConfig


class DatasetInfoSample(BaseModel):
    dataset_name: str
    dataset_subset: str
    dataset_source: Literal["upload", "huggingface"]
    # Modality of the dataset: 'text' or 'vision'
    modality: Literal["text", "vision"] = "text"
    dataset_id: str
    processed_dataset_id: str
    num_examples: int
    created_at: str
    splits: List[str]


class ProcessingResult(DatasetInfoSample):
    pass


class DatasetsInfoResponse(BaseModel):
    datasets: List[DatasetInfoSample]


class DatasetInfoFull(BaseModel):
    dataset_name: str
    dataset_subset: str
    dataset_source: Literal["upload", "huggingface"]
    # Modality of the dataset: 'text' or 'vision'
    modality: Literal["text", "vision"] = "text"
    dataset_id: str
    processed_dataset_id: str
    created_at: str
    splits: List[Dict[str, Any]]


class DatasetInfoResponse(DatasetInfoFull):
    pass


class DatasetDeleteResponse(BaseModel):
    dataset_name: str
    deleted: bool
    message: str
    deleted_files_count: int
    deleted_resources: Optional[List[str]] = None
