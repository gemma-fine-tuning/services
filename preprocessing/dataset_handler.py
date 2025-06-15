import json
import uuid
import io
import logging
import pandas as pd
from typing import List, Dict, Optional
from datasets import load_dataset
from werkzeug.utils import secure_filename
from storage_manager import StorageManager
from schema import DatasetUploadResponse, DatasetInfoResponse

logger = logging.getLogger(__name__)


class DatasetHandler:
    """Single class handling all dataset operations - replaces existing upload/download logic"""

    def __init__(self, storage_manager: StorageManager):
        self.storage = storage_manager
        self.supported_formats = {"txt", "csv", "json", "jsonl", "zip"}

    def _allowed_file(self, filename: str) -> bool:
        """Check if file extension is allowed"""
        return (
            "." in filename
            and filename.rsplit(".", 1)[1].lower() in self.supported_formats
        )

    async def upload_dataset(
        self, file_data: bytes, filename: str, metadata: Optional[Dict] = None
    ) -> DatasetUploadResponse:
        """Enhanced version of current upload_dataset function"""
        try:
            # Validate file
            if not filename:
                raise ValueError("No file selected")

            if not self._allowed_file(filename):
                raise ValueError(
                    f"File type not allowed. Allowed: {self.supported_formats}"
                )

            # Generate unique filename
            file_id = str(uuid.uuid4())
            secure_name = secure_filename(filename)
            blob_name = f"raw_datasets/{file_id}_{secure_name}"

            # Upload to storage
            gcs_path = await self.storage.upload_data(file_data, blob_name, metadata)

            return DatasetUploadResponse(
                dataset_id=file_id,
                filename=secure_name,
                gcs_path=gcs_path,
                size_bytes=len(file_data),
            )

        except Exception as e:
            logger.error(f"Error uploading dataset: {str(e)}")
            raise

    async def load_dataset(
        self, dataset_source: str, dataset_id: str, sample_size: Optional[int] = None
    ) -> List[Dict]:
        """
        Load from upload or HuggingFace - combines existing logic

        Here we choose to return List[Dict] rather than Dataset object to allow for
        future flexibility in custom dataset formats. They are converted to Dataset
        in preprocessing step anyways.
        """
        try:
            if dataset_source == "upload":
                return await self._load_uploaded_dataset(dataset_id)
            elif dataset_source == "standard":
                return await self._load_standard_dataset(dataset_id, sample_size)
            else:
                raise ValueError("Invalid dataset_source")

        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise

    async def _load_uploaded_dataset(self, dataset_id: str) -> List[Dict]:
        """Load uploaded dataset from GCS"""
        # Find uploaded file in GCS
        blobs = self.storage.list_datasets(prefix=f"raw_datasets/{dataset_id}_")
        if not blobs:
            raise FileNotFoundError("Uploaded dataset not found")

        blob_name = blobs[0]
        file_content = await self.storage.download_data(blob_name)

        # Parse file based on extension
        filename = blob_name.split("_", 1)[1]

        if filename.endswith(".csv"):
            df = pd.read_csv(io.StringIO(file_content))
            return df.to_dict("records")
        elif filename.endswith(".json"):
            return json.loads(file_content)
        elif filename.endswith(".jsonl"):
            return [json.loads(line) for line in file_content.strip().split("\n")]
        else:
            raise ValueError("Unsupported file format")

    async def _load_standard_dataset(
        self, dataset_name: str, sample_size: Optional[int] = None
    ) -> List[Dict]:
        """Load a standard dataset from Hugging Face"""
        try:
            # Load dataset from Hugging Face
            dataset = load_dataset(dataset_name, split="train")

            # Shuffle and sample if needed
            if sample_size and len(dataset) > sample_size:
                dataset = dataset.shuffle().select(range(sample_size))

            return list(dataset)

        except Exception as e:
            logger.error(f"Error loading standard dataset: {str(e)}")
            raise

    def validate_dataset(self, dataset: List[Dict], format_type: str):
        """Quality checks and validation - new feature (placeholder)"""
        pass

    async def get_dataset_info(self, dataset_id: str) -> DatasetInfoResponse:
        """Enhanced version of current get_dataset_info"""
        try:
            blob_name = f"processed_datasets/{dataset_id}.json"

            # Check if dataset exists
            if not self.storage.blob_exists(blob_name):
                raise FileNotFoundError("Dataset not found")

            # Get dataset content and metadata
            dataset_content = json.loads(await self.storage.download_data(blob_name))
            metadata = self.storage.get_metadata(blob_name)

            return DatasetInfoResponse(
                dataset_id=dataset_id,
                gcs_path=f"gs://{self.storage.bucket_name}/{blob_name}",
                size=metadata.get("size", 0),
                created=metadata.get("created", ""),
                sample=dataset_content[:3]
                if len(dataset_content) > 3
                else dataset_content,
            )

        except Exception as e:
            logger.error(f"Error getting dataset info: {str(e)}")
            raise

    def detect_schema(self, sample_data: List[Dict]):
        """Auto-detect dataset structure - new feature (placeholder)"""
        pass
