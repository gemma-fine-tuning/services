import os
import json
from typing import Optional
from dataclasses import dataclass
from google.cloud import storage
from datasets import Dataset


@dataclass
class CloudStoredModelMetadata:
    job_id: str
    model_id: str
    gcs_prefix: str  # GCS folder prefix for adapter artifacts
    use_unsloth: bool = False
    local_dir: str = None  # Local path where artifacts are downloaded


class ModelStorageService:
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

        return train_dataset, eval_dataset


# default model storage service instance
data_bucket = os.environ.get("GCS_DATA_BUCKET_NAME", "gemma-dataset-dev")
export_bucket = os.environ.get("GCS_EXPORT_BUCKET_NAME", "gemma-export-dev")
storage_service = ModelStorageService(storage.Client(), data_bucket, export_bucket)
