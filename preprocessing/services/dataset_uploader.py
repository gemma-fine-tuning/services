import uuid
import logging
from typing import Optional, Dict
from werkzeug.utils import secure_filename
from storage.base import StorageInterface
from schema import DatasetUploadResponse

logger = logging.getLogger(__name__)


class DatasetUploader:
    """
    A class that handles dataset file uploads and storage operations.

    This class provides functionality to upload dataset files to storage, with support
    for various file formats and metadata handling. It ensures secure file handling
    and generates unique identifiers for uploaded datasets.

    The uploader supports the following file formats:
    - Text files (.txt)
    - CSV files (.csv)
    - JSON files (.json)
    - JSONL files (.jsonl)
    - Excel files (.xlsx, .xls)
    - Parquet files (.parquet)

    Attributes:
        storage (StorageInterface): An interface for storage operations
        supported_formats (Set[str]): Set of supported file extensions

    Example:
        >>> storage = StorageInterface()
        >>> uploader = DatasetUploader(storage)
        >>> response = await uploader.upload_dataset(file_data, "dataset.csv")
    """

    def __init__(self, storage: StorageInterface):
        """
        Initialize the DatasetUploader with a storage interface.

        Args:
            storage (StorageInterface): An interface for storage operations that provides
                methods for storing and retrieving files.
        """
        self.storage = storage
        self.supported_formats = {
            "txt",
            "csv",
            "json",
            "jsonl",
            "xlsx",
            "xls",
            "parquet",
        }

    def _is_allowed_file(self, filename: str) -> bool:
        """
        Check if the file extension is allowed for upload.

        This method verifies if the file's extension is in the list of supported formats.
        The check is case-insensitive.

        Args:
            filename (str): The name of the file to check

        Returns:
            bool: True if the file extension is supported, False otherwise

        Example:
            >>> is_allowed = uploader._is_allowed_file("dataset.csv")
            >>> print(is_allowed)
            True
        """
        return (
            "." in filename
            and filename.rsplit(".", 1)[1].lower() in self.supported_formats
        )

    async def upload_dataset(
        self, file_data: bytes, filename: str, metadata: Optional[Dict] = None
    ) -> DatasetUploadResponse:
        """
        Upload a dataset file to storage.

        This method handles the complete upload process:
        1. Validates the file format
        2. Generates a unique identifier
        3. Secures the filename
        4. Adds metadata
        5. Uploads to storage
        6. Returns upload response

        Args:
            file_data (bytes): The raw file data to upload
            filename (str): The original name of the file
            metadata (Optional[Dict]): Additional metadata to store with the file.
                Common metadata includes:
                - original_filename: Original name of the file
                - file_id: Unique identifier for the file
                - upload_type: Type of upload (e.g., 'user_upload')
                - custom metadata fields

        Returns:
            DatasetUploadResponse: An object containing:
                - dataset_id (str): Unique identifier for the uploaded dataset
                - filename (str): Secure version of the filename
                - gcs_path (str): Path where the file is stored
                - size_bytes (int): Size of the uploaded file in bytes

        Raises:
            ValueError: If no file is selected or if the file format is not supported
            Exception: For other errors during the upload process

        Example:
            >>> with open("dataset.csv", "rb") as f:
            ...     file_data = f.read()
            >>> response = await uploader.upload_dataset(
            ...     file_data,
            ...     "dataset.csv",
            ...     metadata={"description": "My dataset"}
            ... )
            >>> print(response.dataset_id)
            '123e4567-e89b-12d3-a456-426614174000'
        """
        try:
            if not filename:
                raise ValueError("No file selected")

            if not self._is_allowed_file(filename):
                raise ValueError(
                    f"File type not allowed. Supported formats: {self.supported_formats}"
                )

            file_id = str(uuid.uuid4())
            secure_name = secure_filename(filename)
            blob_name = f"raw_datasets/{file_id}_{secure_name}"

            upload_metadata = {
                "original_filename": filename,
                "file_id": file_id,
                "upload_type": "user_upload",
                **(metadata or {}),
            }

            storage_path = await self.storage.upload_data(
                file_data, blob_name, upload_metadata
            )

            return DatasetUploadResponse(
                dataset_id=file_id,
                filename=secure_name,
                gcs_path=storage_path,
                size_bytes=len(file_data),
            )

        except Exception as e:
            logger.error(f"Error uploading dataset: {str(e)}")
            raise
