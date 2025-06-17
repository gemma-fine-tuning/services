import uuid
import logging
from typing import List, Dict, Any, Optional
from datasets import Dataset
from storage.base import StorageInterface
from .dataset_uploader import DatasetUploader
from .dataset_loader import DatasetLoader
from .dataset_analyzer import DatasetAnalyzer
from .format_converter import FormatConverter
from schema import (
    DatasetUploadResponse,
    DatasetAnalysisResponse,
    ProcessingResult,
    DatasetInfoResponse,
    PreprocessingConfig,
    PreviewResponse,
    ValidationResponse,
)

logger = logging.getLogger(__name__)


class DatasetService:
    """
    A class that orchestrates all dataset operations and provides a unified interface
    for dataset management, processing, and analysis.

    This service combines functionality from multiple components:
    - DatasetUploader: Handles file uploads
    - DatasetLoader: Loads datasets from various sources
    - DatasetAnalyzer: Analyzes dataset structure and content
    - FormatConverter: Converts datasets to ChatML format

    The service provides high-level operations for:
    - Uploading datasets
    - Analyzing dataset structure and content
    - Previewing dataset processing
    - Processing datasets to ChatML format
    - Managing processed datasets
    - Validating dataset formats

    Attributes:
        storage (StorageInterface): Interface for storage operations
        uploader (DatasetUploader): Handles dataset uploads
        loader (DatasetLoader): Handles dataset loading
        analyzer (DatasetAnalyzer): Handles dataset analysis
        converter (FormatConverter): Handles format conversion

    Example:
        >>> storage = StorageInterface()
        >>> service = DatasetService(storage)
        >>> response = await service.upload_dataset(file_data, "dataset.csv")
    """

    def __init__(self, storage: StorageInterface):
        """
        Initialize the DatasetService with required components.

        Args:
            storage (StorageInterface): An interface for storage operations that provides
                methods for storing and retrieving files.
        """
        self.storage = storage
        self.uploader = DatasetUploader(storage)
        self.loader = DatasetLoader(storage)
        self.analyzer = DatasetAnalyzer()
        self.converter = FormatConverter()

    async def upload_dataset(
        self, file_data: bytes, filename: str, metadata: Optional[Dict] = None
    ) -> DatasetUploadResponse:
        """
        Upload a dataset file to storage.

        This method handles the complete upload process, including:
        - File validation
        - Secure filename generation
        - Metadata addition
        - Storage upload
        - Response generation

        Args:
            file_data (bytes): The raw file data to upload
            filename (str): The original name of the file
            metadata (Optional[Dict]): Additional metadata to store with the file.
                Common metadata includes:
                - description: Dataset description
                - source: Data source information
                - tags: Dataset tags
                - custom fields

        Returns:
            DatasetUploadResponse: An object containing:
                - dataset_id (str): Unique identifier for the uploaded dataset
                - filename (str): Secure version of the filename
                - gcs_path (str): Path where the file is stored
                - size_bytes (int): Size of the uploaded file in bytes

        Raises:
            ValueError: If the file is invalid or format is not supported
            Exception: For other errors during upload

        Example:
            >>> with open("dataset.csv", "rb") as f:
            ...     file_data = f.read()
            >>> response = await service.upload_dataset(
            ...     file_data,
            ...     "dataset.csv",
            ...     metadata={"description": "My dataset"}
            ... )
        """
        return await self.uploader.upload_dataset(file_data, filename, metadata)

    async def analyze_dataset(
        self, dataset_source: str, dataset_id: str, sample_size: Optional[int] = None
    ) -> DatasetAnalysisResponse:
        """
        Load and analyze a dataset to provide comprehensive metadata.

        This method performs a complete analysis of the dataset, including:
        - Loading the dataset from the specified source
        - Analyzing dataset structure
        - Computing statistics
        - Detecting format type
        - Providing sample data

        Args:
            dataset_source (str): The source of the dataset. Must be one of:
                - 'upload': For user-uploaded datasets
                - 'huggingface': For datasets from Hugging Face
            dataset_id (str): The identifier for the dataset:
                - For 'upload': The file ID of the uploaded dataset
                - For 'huggingface': The Hugging Face dataset name
            sample_size (Optional[int]): The number of samples to analyze.
                If None, analyzes the entire dataset.

        Returns:
            DatasetAnalysisResponse: An object containing:
                - dataset_id (str): The analyzed dataset's ID
                - total_samples (int): Total number of samples
                - columns (List[str]): List of column names
                - sample_data (List[Dict]): Sample data
                - column_info (Dict): Column statistics
                - format_type (str): Detected format type

        Raises:
            Exception: If there's an error during analysis

        Example:
            >>> analysis = await service.analyze_dataset(
            ...     "upload",
            ...     "my_dataset_id",
            ...     sample_size=1000
            ... )
        """
        try:
            dataset = await self.loader.load_dataset(
                dataset_source, dataset_id, sample_size
            )

            analysis = self.analyzer.analyze_dataset(dataset)

            return DatasetAnalysisResponse(dataset_id=dataset_id, **analysis)

        except Exception as e:
            logger.error(f"Error analyzing dataset: {str(e)}")
            raise

    async def preview_processing(
        self,
        dataset_source: str,
        dataset_id: str,
        config: PreprocessingConfig,
        num_samples: int = 3,
    ) -> PreviewResponse:
        """
        Preview how the dataset would look after processing.

        This method provides a preview of the dataset processing by:
        - Loading a sample of the dataset
        - Applying the processing configuration
        - Showing both original and processed samples
        - Reporting conversion statistics

        Args:
            dataset_source (str): The source of the dataset
            dataset_id (str): The identifier for the dataset
            config (PreprocessingConfig): Configuration for processing, including:
                - field_mappings: Maps input fields to ChatML roles with type and value:
                    - type: "column" or "template"
                    - value: column name or template string with {column} references
                - include_system: Whether to include system message
            num_samples (int): Number of samples to include in preview

        Returns:
            PreviewResponse: An object containing:
                - original_samples (List[Dict]): Original sample data
                - converted_samples (List[Dict]): Processed sample data
                - conversion_success (bool): Whether conversion was successful
                - samples_converted (int): Number of successfully converted samples
                - samples_failed (int): Number of failed conversions

        Raises:
            Exception: If there's an error during preview

        Example:
            >>> config = PreprocessingConfig(
            ...     field_mappings={
            ...         "user_field": {"type": "column", "value": "question"},
            ...         "assistant_field": {"type": "template", "value": "Answer: {answer}"}
            ...     }
            ... )
            >>> preview = await service.preview_processing(
            ...     "upload",
            ...     "my_dataset_id",
            ...     config,
            ...     num_samples=3
            ... )
        """
        try:
            dataset = await self.loader.load_dataset(
                dataset_source, dataset_id, num_samples
            )

            config_dict = config.dict()

            preview = self.converter.preview_conversion(
                dataset, config_dict, num_samples
            )

            return PreviewResponse(**preview)

        except Exception as e:
            logger.error(f"Error previewing processing: {str(e)}")
            raise

    async def process_dataset(
        self,
        dataset_source: str,
        dataset_id: str,
        config: PreprocessingConfig,
        sample_size: Optional[int] = None,
    ) -> ProcessingResult:
        """
        Process a dataset with the given configuration.

        This method performs the complete dataset processing workflow:
        1. Loads the dataset
        2. Converts to ChatML format
        3. Validates the converted dataset
        4. Handles train/test split if requested
        5. Saves the processed dataset

        Args:
            dataset_source (str): The source of the dataset
            dataset_id (str): The identifier for the dataset
            config (PreprocessingConfig): Configuration for processing, including:
                - field_mappings: Maps input fields to ChatML roles with type and value:
                    - type: "column" or "template"
                    - value: column name or template string with {column} references
                - include_system: Whether to include system message
                - train_test_split: Whether to split into train/test sets
                - test_size: Size of test set (if splitting)
            sample_size (Optional[int]): Number of samples to process.
                If None, processes the entire dataset.

        Returns:
            ProcessingResult: An object containing:
                - processed_dataset_id (str): ID of the processed dataset
                - original_count (int): Number of original samples
                - processed_count (int): Number of processed samples
                - train_count (int): Number of training samples (if split)
                - test_count (int): Number of test samples (if split)
                - train_gcs_path (str): Path to training data (if split)
                - test_gcs_path (str): Path to test data (if split)
                - gcs_path (str): Path to processed data (if not split)
                - sample_comparison (Dict): Original and processed samples

        Raises:
            ValueError: If dataset is empty or conversion fails
            Exception: For other errors during processing

        Example:
            >>> config = PreprocessingConfig(
            ...     field_mappings={
            ...         "system_field": {"type": "template", "value": "You are a helpful assistant."},
            ...         "user_field": {"type": "column", "value": "question"},
            ...         "assistant_field": {"type": "template", "value": "Answer: {answer}"}
            ...     },
            ...     train_test_split=True,
            ...     test_size=0.2
            ... )
            >>> result = await service.process_dataset(
            ...     "upload",
            ...     "my_dataset_id",
            ...     config
            ... )
        """
        try:
            dataset = await self.loader.load_dataset(
                dataset_source, dataset_id, sample_size
            )

            if not dataset:
                raise ValueError("Dataset is empty or could not be loaded")

            config_dict = config.dict()

            processed_dataset = self.converter.convert_to_chatml(dataset, config_dict)

            if not processed_dataset:
                raise ValueError("No samples could be converted to ChatML format")

            validation = self.converter.validate_chatml_format(processed_dataset)
            if not validation["is_valid"]:
                logger.warning(f"Dataset validation warnings: {validation['warnings']}")
                if validation["errors"]:
                    raise ValueError(
                        f"Dataset validation failed: {validation['errors'][:3]}"
                    )

            processed_id = str(uuid.uuid4())

            if config.train_test_split:
                return await self._save_with_split(
                    processed_dataset, processed_id, config.test_size, dataset
                )
            else:
                return await self._save_full_dataset(
                    processed_dataset, processed_id, dataset
                )

        except Exception as e:
            logger.error(f"Error processing dataset: {str(e)}")
            raise

    async def _save_with_split(
        self,
        processed_dataset: List[Dict],
        processed_id: str,
        test_size: float,
        original_dataset: List[Dict],
    ) -> ProcessingResult:
        """
        Save a processed dataset with train/test split.

        This method:
        1. Splits the dataset into train and test sets
        2. Saves both sets to storage
        3. Returns processing results with split information

        Args:
            processed_dataset (List[Dict]): The processed dataset to split
            processed_id (str): Unique identifier for the processed dataset
            test_size (float): Proportion of data to use for testing
            original_dataset (List[Dict]): The original dataset for comparison

        Returns:
            ProcessingResult: An object containing split information and paths

        Example:
            >>> result = await service._save_with_split(
            ...     processed_dataset,
            ...     "processed_123",
            ...     0.2,
            ...     original_dataset
            ... )
        """
        hf_dataset = Dataset.from_list(processed_dataset)
        split = hf_dataset.train_test_split(test_size=test_size, shuffle=True, seed=42)

        train_dataset = split["train"]
        test_dataset = split["test"]

        train_blob_name = f"processed_datasets/{processed_id}_train.json"
        test_blob_name = f"processed_datasets/{processed_id}_test.json"

        train_gcs_path = await self.storage.upload_data(
            train_dataset.to_list(), train_blob_name
        )
        test_gcs_path = await self.storage.upload_data(
            test_dataset.to_list(), test_blob_name
        )

        return ProcessingResult(
            processed_dataset_id=processed_id,
            original_count=len(original_dataset),
            processed_count=len(processed_dataset),
            train_count=len(train_dataset),
            test_count=len(test_dataset),
            train_gcs_path=train_gcs_path,
            test_gcs_path=test_gcs_path,
            sample_comparison={
                "original": original_dataset[0] if original_dataset else None,
                "processed": train_dataset[0] if len(train_dataset) > 0 else None,
            },
        )

    async def _save_full_dataset(
        self,
        processed_dataset: List[Dict],
        processed_id: str,
        original_dataset: List[Dict],
    ) -> ProcessingResult:
        """
        Save a processed dataset without splitting.

        This method:
        1. Saves the entire processed dataset to storage
        2. Returns processing results with dataset information

        Args:
            processed_dataset (List[Dict]): The processed dataset to save
            processed_id (str): Unique identifier for the processed dataset
            original_dataset (List[Dict]): The original dataset for comparison

        Returns:
            ProcessingResult: An object containing dataset information and path

        Example:
            >>> result = await service._save_full_dataset(
            ...     processed_dataset,
            ...     "processed_123",
            ...     original_dataset
            ... )
        """
        processed_blob_name = f"processed_datasets/{processed_id}.json"
        gcs_path = await self.storage.upload_data(
            processed_dataset, processed_blob_name
        )

        return ProcessingResult(
            processed_dataset_id=processed_id,
            original_count=len(original_dataset),
            processed_count=len(processed_dataset),
            gcs_path=gcs_path,
            sample_comparison={
                "original": original_dataset[0] if original_dataset else None,
                "processed": processed_dataset[0] if processed_dataset else None,
            },
        )

    async def get_dataset_info(
        self, dataset_id: str, dataset_type: Optional[str] = None
    ) -> DatasetInfoResponse:
        """
        Get information about a dataset (either raw or processed).

        This method retrieves metadata and sample data for a dataset,
        including:
        - Dataset size
        - Creation time
        - Storage path
        - Sample data
        - Dataset type (raw or processed)

        Args:
            dataset_id (str): The identifier of the dataset
            dataset_type (Optional[str]): The type of dataset to look for ("raw" or "processed").
                If None, will check both types.

        Returns:
            DatasetInfoResponse: An object containing:
                - dataset_id (str): The dataset's ID
                - gcs_path (str): Path where the dataset is stored
                - size (int): Size of the dataset in bytes
                - created (str): Creation timestamp
                - sample (List[Dict]): Sample data from the dataset
                - dataset_type (str): Type of dataset ("raw" or "processed")

        Raises:
            FileNotFoundError: If the dataset is not found
            Exception: For other errors during retrieval

        Example:
            >>> info = await service.get_dataset_info("123", "raw")
            >>> print(info.dataset_type)
            'raw'
        """
        try:
            # FIXME: NOT ABLE TO FIND PROCESSED DATASETS
            if dataset_type == "processed":
                processed_path = f"processed_datasets/{dataset_id}.json"
                if self.storage.file_exists(processed_path):
                    possible_paths = [processed_path]
                else:
                    possible_paths = []
            elif dataset_type == "raw":
                raw_files = self.storage.list_files(
                    prefix=f"raw_datasets/{dataset_id}_"
                )
                possible_paths = raw_files if raw_files else []
            else:
                processed_path = f"processed_datasets/{dataset_id}.json"
                raw_files = self.storage.list_files(
                    prefix=f"raw_datasets/{dataset_id}_"
                )
                possible_paths = (
                    [processed_path] if self.storage.file_exists(processed_path) else []
                ) + (raw_files if raw_files else [])

            blob_name = None
            for path in possible_paths:
                if self.storage.file_exists(path):
                    blob_name = path
                    break

            if not blob_name:
                raise FileNotFoundError("Dataset not found")

            dataset_type = "processed" if "processed_datasets" in blob_name else "raw"

            dataset_content = await self.storage.download_data(blob_name)
            import json

            if blob_name.endswith(".json"):
                dataset_list = json.loads(dataset_content)
            else:
                dataset_list = await self.loader.load_dataset("upload", dataset_id)

            metadata = self.storage.get_metadata(blob_name)

            return DatasetInfoResponse(
                dataset_id=dataset_id,
                gcs_path=f"gs://{getattr(self.storage, 'bucket_name', 'local')}/{blob_name}",
                size=metadata.get("size", 0),
                created=metadata.get("created", ""),
                sample=dataset_list[:3] if len(dataset_list) > 3 else dataset_list,
                dataset_type=dataset_type,
            )

        except Exception as e:
            logger.error(f"Error getting dataset info: {str(e)}")
            raise

    def validate_dataset(self, dataset: List[Dict]) -> ValidationResponse:
        """
        Validate a dataset in ChatML format.

        This method performs comprehensive validation of the ChatML format,
        checking:
        - Overall structure
        - Message format
        - Required fields
        - Valid roles
        - Presence of user and assistant messages

        Args:
            dataset (List[Dict]): The dataset to validate

        Returns:
            ValidationResponse: An object containing:
                - is_valid (bool): Whether the dataset is valid
                - errors (List[str]): List of validation errors
                - warnings (List[str]): List of validation warnings
                - total_samples (int): Total number of samples
                - valid_samples (int): Number of valid samples

        Example:
            >>> validation = service.validate_dataset(dataset)
            >>> print(validation.is_valid)
            True
        """
        try:
            validation = self.converter.validate_chatml_format(dataset)
            return ValidationResponse(**validation)
        except Exception as e:
            logger.error(f"Error validating dataset: {str(e)}")
            raise

    def get_column_statistics(self, dataset: List[Dict], column: str) -> Dict[str, Any]:
        """
        Get detailed statistics for a specific column.

        This method provides comprehensive statistics for a given column, including:
        - Total number of values
        - Number of unique values
        - Sample values
        - For text fields: average, minimum, and maximum lengths

        Args:
            dataset (List[Dict]): The dataset to analyze
            column (str): The name of the column to analyze

        Returns:
            Dict[str, Any]: A dictionary containing:
                - total_values (int): Total number of values
                - unique_values (int): Number of unique values
                - sample_values (List): First 10 values
                - For text fields:
                    - avg_length (float): Average length of values
                    - min_length (int): Minimum length
                    - max_length (int): Maximum length
                - error (str): Error message if no valid values found

        Example:
            >>> stats = service.get_column_statistics(dataset, 'question')
            >>> print(stats['avg_length'])
            45.6
        """
        return self.analyzer.get_column_statistics(dataset, column)
