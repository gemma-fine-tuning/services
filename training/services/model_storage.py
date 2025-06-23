import os
import json
from typing import Optional, Dict, Any
from dataclasses import dataclass
from google.cloud import storage
from datasets import Dataset
import logging
from abc import ABC, abstractmethod
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CloudStoredModelMetadata:
    job_id: str
    model_id: str
    gcs_prefix: str  # GCS folder prefix for adapter artifacts
    use_unsloth: bool = False
    local_dir: str = None  # Local path where artifacts are downloaded


class CloudStorageService:
    """
    Service for managing artifacts in Google Cloud Storage (GCS).
    This is used for **BOTH** dataset retrieval and model artifact storage.
    """

    def __init__(self, storage_client, data_bucket: str, export_bucket: str):
        self.storage_client = storage_client
        self.data_bucket = data_bucket
        self.export_bucket = export_bucket

    def upload_model(self, model_dir: str, metadata: CloudStoredModelMetadata) -> str:
        """
        Upload model artifacts and metadata to GCS, return remote URI

        Ensure that `job_id` is unique for each training run because the
        artifacts are stored under `trained_adapters/{job_id}` prefix.

        Args:
            model_dir (str): Local directory containing adapter files
            metadata (CloudStoredModelMetadata): Metadata object with job details

        Returns:
            str: GCS URI where the model artifacts are stored
        """
        bucket = self.storage_client.bucket(self.export_bucket)
        prefix = f"trained_adapters/{metadata.job_id}"

        # upload adapter files
        for root, dirs, files in os.walk(model_dir):
            for fn in files:
                src = os.path.join(root, fn)
                rel = os.path.relpath(src, model_dir)
                blob = bucket.blob(f"{prefix}/{rel}")
                blob.upload_from_filename(src)

        # upload metadata/config.json
        meta_dict = {
            "job_id": metadata.job_id,
            "model_id": metadata.model_id,
            "use_unsloth": metadata.use_unsloth,
        }

        blob = bucket.blob(f"{prefix}/config.json")
        blob.upload_from_string(json.dumps(meta_dict), content_type="application/json")

        return f"gs://{self.export_bucket}/{prefix}/"

    def download_model(
        self, job_id: str, local_dir: Optional[str] = None
    ) -> CloudStoredModelMetadata:
        """
        Download model artifacts and metadata from cloud storage into a local dir

        Args:
            job_id (str): Unique identifier for the training job
            local_dir (Optional[str]): Local directory to download artifacts to.
                                       If None, uses a temporary directory.

        Returns:
            CloudStoredModelMetadata: Metadata object with job details and local path
        """
        bucket = self.storage_client.bucket(self.export_bucket)
        prefix = f"trained_adapters/{job_id}"

        # fetch metadata
        config_blob = bucket.blob(f"{prefix}/config.json")
        if not config_blob.exists():
            raise FileNotFoundError("Adapter config not found")
        meta = json.loads(config_blob.download_as_text())

        # prepare local directory
        if not local_dir:
            local_dir = f"/tmp/inference_{job_id}"
        os.makedirs(local_dir, exist_ok=True)

        # download all artifacts
        for blob in bucket.list_blobs(prefix=prefix):
            rel = blob.name[len(prefix) + 1 :]
            if rel == "config.json":
                continue
            dst = os.path.join(local_dir, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            blob.download_to_filename(dst)

        # build metadata object
        metadata = CloudStoredModelMetadata(
            job_id=job_id,
            model_id=meta.get("model_id"),
            gcs_prefix=prefix,
            use_unsloth=meta.get("use_unsloth", False),
            local_dir=local_dir,
        )

        return metadata

    def download_processed_dataset(self, processed_dataset_id: str):
        """
        Download processed dataset files from GCS and return as HuggingFace Datasets.

        Args:
            processed_dataset_id (str): Identifier for the processed dataset

        Returns:
            Tuple[Dataset, Optional[Dataset]]: Train and eval datasets
        """
        bucket = self.storage_client.bucket(self.data_bucket)

        # download train dataset
        train_blob = bucket.blob(
            f"processed_datasets/{processed_dataset_id}_train.json"
        )
        if not train_blob.exists():
            raise FileNotFoundError("Training dataset not found in GCS")
        train_data = json.loads(train_blob.download_as_text())
        train_dataset = Dataset.from_list(train_data)

        # download eval dataset if exists
        eval_blob = bucket.blob(f"processed_datasets/{processed_dataset_id}_test.json")
        eval_dataset = None
        if eval_blob.exists():
            eval_data = json.loads(eval_blob.download_as_text())
            if eval_data:
                eval_dataset = Dataset.from_list(eval_data)
        else:
            logger.warning(
                f"Eval dataset not found for {processed_dataset_id}, using train only"
            )

        return train_dataset, eval_dataset


@dataclass
class ModelArtifact:
    """Unified representation of model artifacts regardless of storage backend"""

    model_id: str
    job_id: str
    local_path: str
    remote_path: str
    use_unsloth: bool = False
    metadata: Optional[Dict[str, Any]] = None


class ModelStorageStrategy(ABC):
    """Abstract base strategy for model storage operations"""

    @abstractmethod
    def save_model(
        self, model, tokenizer, local_path: str, metadata: Dict[str, Any]
    ) -> ModelArtifact:
        """Save model artifacts and return artifact reference"""
        pass

    @abstractmethod
    def load_model_info(self, artifact_id: str) -> ModelArtifact:
        """Load model metadata and prepare for inference"""
        pass

    @abstractmethod
    def cleanup(self, artifact: ModelArtifact) -> None:
        """Clean up local resources"""
        pass


class GCSStorageStrategy(ModelStorageStrategy):
    """Google Cloud Storage implementation"""

    def __init__(self, storage_service):
        self.storage_service = storage_service

    def save_model(
        self, model, tokenizer, local_path: str, metadata: Dict[str, Any]
    ) -> ModelArtifact:
        """Save model to GCS and return artifact reference"""
        # Use existing storage service logic
        from training.services.model_storage import CloudStoredModelMetadata

        cloud_metadata = CloudStoredModelMetadata(
            job_id=metadata["job_id"],
            model_id=metadata["model_id"],
            gcs_prefix=f"trained_adapters/{metadata['job_id']}",
            use_unsloth=metadata.get("use_unsloth", False),
            local_dir=local_path,
        )

        # Save model locally first
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(local_path)
        else:
            # For trainers
            model.save_model(local_path)
        tokenizer.save_pretrained(local_path)

        # Upload to GCS
        remote_path = self.storage_service.upload_model(local_path, cloud_metadata)

        return ModelArtifact(
            model_id=metadata["model_id"],
            job_id=metadata["job_id"],
            local_path=local_path,
            remote_path=remote_path,
            use_unsloth=metadata.get("use_unsloth", False),
            metadata=metadata,
        )

    def load_model_info(self, job_id: str) -> ModelArtifact:
        """Load model from GCS"""
        meta = self.storage_service.download_model(job_id)

        return ModelArtifact(
            model_id=meta.model_id,
            job_id=meta.job_id,
            local_path=meta.local_dir,
            remote_path=f"gs://{self.storage_service.export_bucket}/{meta.gcs_prefix}/",
            use_unsloth=meta.use_unsloth,
            metadata={"gcs_prefix": meta.gcs_prefix},
        )

    def cleanup(self, artifact: ModelArtifact) -> None:
        """Clean up local GCS artifacts"""
        if artifact.local_path and os.path.exists(artifact.local_path):
            shutil.rmtree(artifact.local_path, ignore_errors=True)


class HuggingFaceHubStrategy(ModelStorageStrategy):
    """HuggingFace Hub storage implementation"""

    def __init__(self):
        pass

    def save_model(
        self, model, tokenizer, local_path: str, metadata: Dict[str, Any]
    ) -> ModelArtifact:
        """Push model to HuggingFace Hub"""
        hf_repo_id = metadata["hf_repo_id"]

        # Save locally first (optional, for consistency)
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(local_path)
        else:
            model.save_model(local_path)
        tokenizer.save_pretrained(local_path)

        # Push to HuggingFace Hub
        logging.info(f"Pushing model to Hugging Face Hub at {hf_repo_id}")

        if hasattr(model, "push_to_hub"):
            # Direct model push (for Unsloth/base models)
            model.push_to_hub(repo_id=hf_repo_id, private=True)
        else:
            # For trainers, get the model first
            model.model.push_to_hub(repo_id=hf_repo_id, private=True)

        tokenizer.push_to_hub(hf_repo_id)

        # Save additional metadata
        training_config = {
            "model_id": metadata["model_id"],
            "use_unsloth": metadata.get("use_unsloth", False),
            "job_id": metadata["job_id"],
        }

        # Upload training config to HF Hub for inference
        config_path = os.path.join(local_path, "adapter_training_config.json")
        with open(config_path, "w") as f:
            json.dump(training_config, f)

        return ModelArtifact(
            model_id=metadata["model_id"],
            job_id=metadata["job_id"],
            local_path=local_path,
            remote_path=hf_repo_id,
            use_unsloth=metadata.get("use_unsloth", False),
            metadata={"hf_repo_id": hf_repo_id},
        )

    def load_model_info(self, repo_id: str) -> ModelArtifact:
        """Load model info from HuggingFace Hub"""
        from huggingface_hub import hf_hub_download

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
            model_id=base_model_id,
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
    Factory for creating storage strategies

    Example:
    ```python
    storage_strategy = StorageStrategyFactory.create_strategy("gcs", storage_service=storage_service)
    artifact = storage_strategy.save_model(model, tokenizer, local_path, metadata)
    ```
    """

    @staticmethod
    def create_strategy(storage_type: str, **kwargs) -> ModelStorageStrategy:
        """Create appropriate storage strategy"""
        if storage_type == "gcs":
            return GCSStorageStrategy(kwargs.get("storage_service"))
        elif storage_type == "hfhub":
            return HuggingFaceHubStrategy()
        else:
            raise ValueError(f"Unknown storage type: {storage_type}")


# default model storage service instance
data_bucket = os.environ.get("GCS_DATA_BUCKET_NAME", "gemma-dataset-dev")
export_bucket = os.environ.get("GCS_EXPORT_BUCKET_NAME", "gemma-export-dev")
storage_service = CloudStorageService(storage.Client(), data_bucket, export_bucket)
