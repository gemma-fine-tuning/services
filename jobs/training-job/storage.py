import os
import json
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from google.cloud import storage
from datasets import Dataset
import logging
from abc import ABC, abstractmethod
import shutil
import io
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CloudStoredModelMetadata:
    job_id: str
    base_model_id: str
    gcs_prefix: str  # GCS folder prefix for adapter artifacts
    use_unsloth: bool = False
    local_dir: Optional[str] = None  # Local path where artifacts are downloaded
    hf_repo_id: Optional[str] = None  # HuggingFace repo ID if applicable

    # these must match the value in export_config
    export_format: Optional[str] = (
        None  # Format of the exported model (e.g., gguf, adapter)
    )
    quantization: Optional[str] = None  # Quantization type if applicable


class CloudStorageService:
    """
    Service for managing artifacts in Google Cloud Storage (GCS).
    This is used for **BOTH** dataset retrieval and model artifact storage.

    Handles uploading/downloading model adapters and processed datasets to/from GCS buckets.
    Provides a unified interface for cloud storage operations across the training pipeline.

    Args:
        data_bucket: GCS bucket name for storing datasets
        export_bucket: GCS bucket name for storing trained model artifacts
    """

    def __init__(self, data_bucket: str, export_bucket: str):
        self.data_bucket = data_bucket
        self.export_bucket = export_bucket
        self.storage_client = storage.Client()

    def upload_model(self, model_dir: str, metadata: CloudStoredModelMetadata) -> str:
        """
        Upload model artifacts and metadata to GCS, return remote URI

        Uses different folder structures based on export format:
        - trained_adapters/{job_id}/ for adapter-only exports
        - merged_models/{job_id}/ for merged model exports
        - gguf_models/{job_id}/ for GGUF format exports

        Args:
            model_dir (str): Local directory containing model files
            metadata (CloudStoredModelMetadata): Metadata object with job details

        Returns:
            str: GCS URI where the model artifacts are stored
        """
        try:
            bucket = self.storage_client.bucket(self.export_bucket)

            # Determine folder prefix based on export format
            format_prefix = self._get_format_prefix(metadata.export_format)
            prefix = f"{format_prefix}/{metadata.job_id}"

            # Upload all files in the model directory
            for root, dirs, files in os.walk(model_dir):
                for fn in files:
                    src = os.path.join(root, fn)
                    rel = os.path.relpath(src, model_dir)
                    blob = bucket.blob(f"{prefix}/{rel}")
                    blob.upload_from_filename(src)

            meta_dict = {
                "job_id": metadata.job_id,
                "base_model_id": metadata.base_model_id,
                "use_unsloth": metadata.use_unsloth,
                "export_format": metadata.export_format,
                "quantization": metadata.quantization,
                "hf_repo_id": metadata.hf_repo_id,
            }
            blob = bucket.blob(f"{prefix}/config.json")
            blob.upload_from_string(
                json.dumps(meta_dict), content_type="application/json"
            )

            # This is the location for this export request
            return f"gs://{self.export_bucket}/{prefix}/"
        except Exception as e:
            logger.error(
                f"Error uploading model artifacts to GCS for job {metadata.job_id}: {e}",
                exc_info=True,
            )
            raise

    def download_model(
        self,
        job_id: str,
        export_format: str = "adapter",
        local_dir: Optional[str] = None,
    ) -> CloudStoredModelMetadata:
        """
        Download model artifacts and metadata from cloud storage into a local dir

        Args:
            job_id (str): Unique identifier for the training job
            export_format (str): Export format to determine correct folder structure
            local_dir (Optional[str]): Local directory to download artifacts to.
                                       If None, uses a temporary directory.

        Returns:
            CloudStoredModelMetadata: Metadata object with job details and local path
        """
        try:
            bucket = self.storage_client.bucket(self.export_bucket)

            # Determine folder prefix based on export format
            format_prefix = self._get_format_prefix(export_format)
            prefix = f"{format_prefix}/{job_id}"

            config_blob = bucket.blob(f"{prefix}/config.json")
            if not config_blob.exists():
                logger.error(f"Adapter config not found in GCS for job {job_id}")
                raise FileNotFoundError("Adapter config not found")
            meta = json.loads(config_blob.download_as_text())

            if not local_dir:
                local_dir = f"/tmp/inference_{job_id}"
            os.makedirs(local_dir, exist_ok=True)

            # Download all files except config
            for blob in bucket.list_blobs(prefix=prefix):
                rel = blob.name[len(prefix) + 1 :]
                if rel == "config.json":
                    continue
                dst = os.path.join(local_dir, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                blob.download_to_filename(dst)

            metadata = CloudStoredModelMetadata(
                job_id=job_id,
                base_model_id=meta.get("base_model_id"),
                gcs_prefix=prefix,
                use_unsloth=meta.get("use_unsloth", False),
                local_dir=local_dir,
                export_format=meta.get("export_format", export_format),
                quantization=meta.get("quantization"),
                hf_repo_id=meta.get("hf_repo_id"),
            )
            return metadata
        except Exception as e:
            logger.error(
                f"Error downloading model artifacts from GCS for job {job_id}: {e}",
                exc_info=True,
            )
            raise

    def download_processed_dataset(
        self, processed_dataset_id: str
    ) -> Tuple[Dataset, Optional[Dataset]]:
        """
        Download processed dataset files from GCS and return as HuggingFace Datasets.

        Retrieves training and optional evaluation datasets from the configured data bucket.
        The datasets are expected to be stored as Parquet files with metadata.json in the
        new preprocessing service format.

        Args:
            processed_dataset_id (str): Identifier for the processed dataset (dataset_name from preprocessing)

        Returns:
            Tuple[Dataset, Optional[Dataset]]: Train and eval datasets

        Raises:
            FileNotFoundError: If the training dataset is not found in GCS
        """
        bucket = self.storage_client.bucket(self.data_bucket)

        # First, download and parse the metadata to understand the dataset structure
        metadata_blob = bucket.blob(
            f"processed_datasets/{processed_dataset_id}/metadata.json"
        )
        if not metadata_blob.exists():
            raise FileNotFoundError(
                f"Dataset metadata not found for {processed_dataset_id}"
            )

        metadata = json.loads(metadata_blob.download_as_text())
        splits = metadata.get("splits", [])

        if not splits:
            raise FileNotFoundError(
                f"No splits found in dataset {processed_dataset_id}"
            )

        # Find train and test splits
        train_split = None
        eval_split = None

        for split_info in splits:
            split_name = split_info.get("split_name", "").lower()
            if split_name in ["train", "training"]:
                train_split = split_info
            elif split_name in ["test", "validation", "eval", "evaluation"]:
                eval_split = split_info

        # If no explicit train split found, use the first split as train
        if not train_split and splits:
            train_split = splits[0]
            logger.warning(
                f"No explicit train split found for {processed_dataset_id}, using {train_split.get('split_name')}"
            )

        if not train_split:
            raise FileNotFoundError(
                f"No train split found in dataset {processed_dataset_id}"
            )

        # Download train dataset
        train_blob = bucket.blob(
            f"processed_datasets/{processed_dataset_id}/{train_split['split_name']}.parquet"
        )
        if not train_blob.exists():
            raise FileNotFoundError(
                f"Train dataset file not found for {processed_dataset_id}"
            )

        # Download as bytes and load as parquet
        train_data_bytes = train_blob.download_as_bytes()

        train_table = pq.read_table(io.BytesIO(train_data_bytes))
        train_dataset = Dataset(train_table)

        # Download eval dataset if exists
        eval_dataset = None
        if eval_split:
            eval_blob = bucket.blob(
                f"processed_datasets/{processed_dataset_id}/{eval_split['split_name']}.parquet"
            )
            if eval_blob.exists():
                eval_data_bytes = eval_blob.download_as_bytes()
                eval_table = pq.read_table(io.BytesIO(eval_data_bytes))
                eval_dataset = Dataset(eval_table)
            else:
                logger.warning(
                    f"Eval dataset file not found for {processed_dataset_id}, using train only"
                )
        else:
            logger.warning(
                f"No eval split found for {processed_dataset_id}, using train only"
            )

        return train_dataset, eval_dataset

    def _get_format_prefix(self, export_format: Optional[str]) -> str:
        """
        Get the folder prefix based on export format.

        Returns:
            str: Folder prefix for storing the model artifacts
        """
        format_mapping = {
            "adapter": "trained_adapters",
            "merged": "merged_models",
            "gguf": "gguf_models",
        }
        return format_mapping.get(
            export_format, "adapters"
        )  # Default to adapters for backwards compatibility


@dataclass
class ModelArtifact:
    """Unified representation of model artifacts regardless of storage backend"""

    base_model_id: str
    job_id: str
    local_path: str
    remote_path: str
    use_unsloth: bool = False
    metadata: Optional[Dict[str, Any]] = None


class ModelStorageStrategy(ABC):
    """
    Abstract base strategy for model storage operations.

    Implements the Strategy pattern to support multiple storage backends (GCS, HuggingFace Hub, etc.).
    Each concrete strategy handles the specifics of saving, loading, and cleaning up model artifacts
    for its respective storage system.

    This abstraction allows the training pipeline to work with different storage backends
    without changing the core training logic.
    """

    @abstractmethod
    def save_model(
        self, local_path: str, metadata: CloudStoredModelMetadata
    ) -> ModelArtifact:
        """
        Upload model artifacts from local path to storage backend.

        Args:
            local_path: Local directory containing saved model files
            metadata: Dictionary containing job_id, base_model_id, and storage-specific config

        Returns:
            ModelArtifact: Artifact reference with paths and metadata
        """
        pass

    @abstractmethod
    def load_model_info(self, artifact_id: str) -> ModelArtifact:
        """
        Load model metadata and prepare for inference.

        Retrieves model information from the storage backend without loading the actual
        model weights. This is used to get artifact metadata for inference setup.

        Args:
            artifact_id: Storage-specific identifier (job_id for GCS, repo_id for HF Hub)

        Returns:
            ModelArtifact: Artifact reference with metadata for model loading
        """
        pass

    @abstractmethod
    def cleanup(self, artifact: ModelArtifact) -> None:
        """Clean up local resources"""
        pass


class GCSStorageStrategy(ModelStorageStrategy):
    """
    Google Cloud Storage implementation of model storage strategy.

    Handles saving and loading model artifacts to/from GCS buckets. This strategy
    is used for persistent storage of trained adapters and supports both Unsloth
    and standard HuggingFace models.
    """

    def __init__(self):
        self.storage_service: CloudStorageService = storage_service

    def save_model(
        self, local_path: str, metadata: CloudStoredModelMetadata
    ) -> ModelArtifact:
        """
        Upload model artifacts from local path to GCS.

        Args:
            local_path: Local directory containing saved model files
            metadata: Must contain job_id, base_model_id, and optional use_unsloth flag

        Returns:
            ModelArtifact: Reference to the uploaded model with GCS paths
        """
        # Upload to GCS (model is already saved locally by utils.py)
        remote_path = self.storage_service.upload_model(local_path, metadata)

        return ModelArtifact(
            base_model_id=metadata.base_model_id,
            job_id=metadata.job_id,
            local_path=local_path,
            remote_path=remote_path,
            use_unsloth=metadata.use_unsloth,
            metadata={
                "export_format": metadata.export_format,
                "quantization": metadata.quantization,
                "gcs_prefix": metadata.gcs_prefix,
            },
        )

    def load_model_info(
        self, job_id: str, export_format: str = "adapter"
    ) -> ModelArtifact:
        """Load model from GCS with format-specific folder structure"""
        meta = self.storage_service.download_model(job_id, export_format)

        return ModelArtifact(
            base_model_id=meta.base_model_id,
            job_id=meta.job_id,
            local_path=meta.local_dir or "",
            remote_path=f"gs://{self.storage_service.export_bucket}/{meta.gcs_prefix}/",
            use_unsloth=meta.use_unsloth,
            metadata={
                "gcs_prefix": meta.gcs_prefix,
                "export_format": meta.export_format,
                "quantization": meta.quantization,
            },
        )

    def cleanup(self, artifact: ModelArtifact) -> None:
        """Clean up local GCS artifacts"""
        if artifact.local_path and os.path.exists(artifact.local_path):
            shutil.rmtree(artifact.local_path, ignore_errors=True)


class HuggingFaceHubStrategy(ModelStorageStrategy):
    """
    HuggingFace Hub storage implementation of model storage strategy.

    Pushes trained models directly to HuggingFace Hub repositories for public or private
    sharing. This strategy is ideal for models that will be shared or need to be accessible
    via the HuggingFace ecosystem.

    Note: Requires proper HuggingFace authentication and repository permissions.
    """

    def __init__(self):
        pass

    def save_model(
        self, local_path: str, metadata: CloudStoredModelMetadata
    ) -> ModelArtifact:
        """
        Upload all files in local directory to HuggingFace Hub repository.
        Much more efficient than loading and re-pushing models.
        """
        hf_repo_id = metadata.hf_repo_id
        logging.info(f"Uploading folder {local_path} to HuggingFace Hub: {hf_repo_id}")

        try:
            api = HfApi()
            api.upload_folder(
                folder_path=local_path,
                repo_id=hf_repo_id,
                repo_type="model",
                private=True,
            )
        except Exception as e:
            logging.error(f"Failed to upload folder to HuggingFace Hub: {e}")
            raise

        return ModelArtifact(
            base_model_id=metadata.base_model_id,
            job_id=metadata.job_id,
            local_path=local_path,
            remote_path=hf_repo_id,
            use_unsloth=metadata.use_unsloth,
            metadata={"hf_repo_id": hf_repo_id},
        )

    def load_model_info(self, repo_id: str) -> ModelArtifact:
        """
        Load model info from HuggingFace Hub.

        Attempts to retrieve model configuration from the HuggingFace repository
        to determine the base model and framework used. Falls back gracefully
        if configuration files are not available.

        Args:
            repo_id: HuggingFace repository identifier (e.g., "user/model-name")

        Returns:
            ModelArtifact: Model metadata for inference setup
        """
        try:
            # Try adapter config first
            adapter_config_path = hf_hub_download(
                repo_id=repo_id, filename="adapter_config.json"
            )
            with open(adapter_config_path, "r") as f:
                adapter_config = json.load(f)
            base_model_id = adapter_config.get("base_model_name_or_path", repo_id)
            use_unsloth = True if base_model_id.startswith("unsloth/") else False
        except Exception:
            logging.warning(
                f"Failed to load adapter config for {repo_id}, falling back to repo_id as model_id"
            )

        return ModelArtifact(
            base_model_id=base_model_id,
            job_id=repo_id,  # Use repo_id as job_id for HF Hub
            local_path="",  # No local path for HF Hub
            remote_path=repo_id,
            use_unsloth=use_unsloth,
            metadata={"hf_repo_id": repo_id},
        )

    def cleanup(self, artifact: ModelArtifact) -> None:
        """
        No local artifacts to clean up for HuggingFace Hub.
        This method is required as abstract method but does nothing
        """
        pass  # No local artifacts to clean up for HF Hub


class StorageStrategyFactory:
    """
    Factory for creating storage strategies.

    Implements the Factory pattern to instantiate the appropriate storage strategy
    based on the export type specified in training requests. This allows the training
    pipeline to work with multiple storage backends without tight coupling.

    Example:
    ```python
    storage_strategy = StorageStrategyFactory.create_strategy("gcs", storage_service=storage_service)
    artifact = storage_strategy.save_model(model, tokenizer, local_path, metadata)
    ```

    Supported storage types:
    - "gcs": Google Cloud Storage (requires storage_service parameter)
    - "hfhub": HuggingFace Hub (no additional parameters required)
    """

    @staticmethod
    def create_strategy(storage_type: str, **kwargs) -> ModelStorageStrategy:
        """
        Create appropriate storage strategy based on type.

        Args:
            storage_type: Type of storage ("gcs" or "hfhub")
            **kwargs: Additional parameters required by specific strategies
                     - For "gcs": storage_service (CloudStorageService instance)
                     - For "hfhub": no additional parameters required

        Returns:
            ModelStorageStrategy: Configured storage strategy instance

        Raises:
            ValueError: If storage_type is not supported
        """
        if storage_type == "gcs":
            return GCSStorageStrategy()
        elif storage_type == "hfhub":
            return HuggingFaceHubStrategy()
        else:
            raise ValueError(f"Unknown storage type: {storage_type}")


# default model storage service instance
data_bucket = os.environ.get("GCS_DATA_BUCKET_NAME", "gemma-dataset-dev")
export_bucket = os.environ.get("GCS_EXPORT_BUCKET_NAME", "gemma-export-dev")
storage_service = CloudStorageService(data_bucket, export_bucket)
