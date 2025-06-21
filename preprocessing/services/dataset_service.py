import uuid
import logging
from typing import List, Dict, Any, Optional
from storage.base import StorageInterface
from .dataset_uploader import DatasetUploader
from .dataset_loader import DatasetLoader
from .dataset_analyzer import DatasetAnalyzer
from .format_converter import FormatConverter
from augmentation import run_augment_pipeline
from schema import (
    DatasetUploadResponse,
    DatasetAnalysisResponse,
    ProcessingResult,
    DatasetInfoResponse,
    PreprocessingConfig,
    PreviewResponse,
    ValidationResponse,
)
from datasets import Dataset

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
    ) -> ProcessingResult:
        """
        Process a dataset to ChatML format.

        This method performs the complete dataset processing pipeline:
        1. Loads the dataset from the specified source
        2. Converts it to ChatML format
        3. Applies data augmentation if configured
        4. Validates the converted dataset
        5. Saves the processed dataset to storage
        6. Returns processing results

        Args:
            dataset_source (str): The source of the dataset
            dataset_id (str): The identifier for the dataset
            config (PreprocessingConfig): Configuration for processing, including:
                - field_mappings: Maps input fields to ChatML roles
                - system_message: Default system message
                - include_system: Whether to include system message
                - user_template: Template for formatting user content
                - train_test_split: Whether to split into train/test sets
                - test_size: Size of test set (if splitting)
                - augmentation_config: Configuration for data augmentation, including:
                    - enabled: Whether to apply augmentation
                    - use_eda: Whether to use Easy Data Augmentation
                    - use_back_translation: Whether to use back translation
                    - use_paraphrasing: Whether to use paraphrasing
                    - use_synthesis: Whether to use AI synthesis
                    - gemini_api_key: API key for Gemini (if using synthesis)
                    - augmentation_factor: Factor to increase dataset size
            sample_size (Optional[int]): Number of samples to process.
                If None, processes the entire dataset.

        Returns:
            ProcessingResult: An object containing processing results and metadata

        Raises:
            Exception: If there's an error during processing

        Example:
            >>> config = PreprocessingConfig(
            ...     field_mappings={
            ...         "system_field": {"type": "template", "value": "You are a helpful assistant."},
            ...         "user_field": {"type": "column", "value": "question"},
            ...         "assistant_field": {"type": "template", "value": "Answer: {answer}"}
            ...     },
            ...     system_message="You are a helpful assistant.",
            ...     train_test_split=True,
            ...     test_size=0.2,
            ...     augmentation_config={
            ...         "enabled": True,
            ...         "use_eda": True,
            ...         "use_synthesis": True,
            ...         "gemini_api_key": "your_api_key",
            ...         "augmentation_factor": 1.5
            ...     }
            ... )
            >>> result = await service.process_dataset(
            ...     "upload",
            ...     "my_dataset_id",
            ...     config,
            ...     split_config=HFSplitConfig(type="hf_split", splits=["train", "test"]),
            ... )
            >>> result = await service.process_dataset("upload", "my_dataset", config)
        """
        try:
            # Load dataset with splits
            dataset = await self.loader.load_dataset(dataset_source, dataset_id, config)

            if not dataset:
                raise ValueError("Dataset is empty or could not be loaded")

            config_dict = config.dict()

            processed_dataset = self.converter.convert_to_chatml(dataset, config.dict)

            # Apply data augmentation if enabled
            augmentation_config = config_dict.get("augmentation_config", {})
            if augmentation_config.get("enabled", False):
                dataset = self._apply_augmentation(dataset, augmentation_config)

            if not processed_dataset:
                raise ValueError("No samples could be converted to ChatML format")
            # Convert to ChatML format
            processed_dataset = self.converter.convert_to_chatml(dataset, config.dict())

            # Validate the processed dataset
            validation = self.converter.validate_chatml_format(processed_dataset)
            if not validation["is_valid"]:
                logger.warning(f"Dataset validation warnings: {validation['warnings']}")
                if validation["errors"]:
                    raise ValueError(
                        f"Dataset validation failed: {validation['errors'][:3]}"
                    )

            processed_id = str(uuid.uuid4())

            # Save all splits
            return await self._save_dataset(processed_dataset, processed_id, dataset)

        except Exception as e:
            logger.error(f"Error processing dataset: {str(e)}")
            raise

    async def _save_dataset(
        self,
        processed_dataset: Dict[str, List[Dict]],
        processed_id: str,
        original_dataset: Dict[str, List[Dict]],
    ) -> ProcessingResult:
        """
        Save a processed dataset with all its splits.

        This method:
        1. Saves each split of the processed dataset to storage with appropriate suffixes
        2. Returns processing results with information about all splits

        Args:
            processed_dataset (Dict[str, List[Dict]]): The processed dataset with splits
            processed_id (str): Unique identifier for the processed dataset
            original_dataset (Dict[str, List[Dict]]): The original dataset for comparison

        Returns:
            ProcessingResult: An object containing split information and paths

        Example:
            >>> result = await service._save_dataset(
            ...     {"train": [...], "test": [...]},
            ...     "processed_123",
            ...     {"train": [...], "test": [...]}
            ... )
        """
        splits_info = {}
        total_processed_count = 0
        total_original_count = 0

        # Save each split
        for split_name, split_data in processed_dataset.items():
            total_processed_count += len(split_data)

            # Create blob name with split suffix
            blob_name = f"processed_datasets/{processed_id}_{split_name}.json"

            # Upload split data
            gcs_path = await self.storage.upload_data(split_data, blob_name)

            # Store split information
            splits_info[split_name] = {
                "count": len(split_data),
                "gcs_path": gcs_path,
                "processed_count": len(split_data),
            }

        # Calculate total original count
        for split_data in original_dataset.values():
            total_original_count += len(split_data)

        # Get sample comparison from train split only
        train_split = processed_dataset.get("train", [])
        original_train = original_dataset.get("train", [])

        sample_comparison = {
            "original": original_train[0] if original_train else None,
            "processed": train_split[0] if train_split else None,
        }

        return ProcessingResult(
            processed_dataset_id=processed_id,
            original_count=total_original_count,
            processed_count=total_processed_count,
            splits=splits_info,
            sample_comparison=sample_comparison,
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
            if dataset_type == "processed":
                # Look for processed datasets with split suffixes
                processed_files = self.storage.list_files(
                    prefix=f"processed_datasets/{dataset_id}_"
                )
                if processed_files:
                    possible_paths = processed_files
                else:
                    # Fallback to old format without suffix
                    processed_path = f"processed_datasets/{dataset_id}.json"
                    possible_paths = (
                        [processed_path]
                        if self.storage.file_exists(processed_path)
                        else []
                    )
            elif dataset_type == "raw":
                raw_files = self.storage.list_files(
                    prefix=f"raw_datasets/{dataset_id}_"
                )
                possible_paths = raw_files if raw_files else []
            else:
                # Check both processed and raw datasets
                processed_files = self.storage.list_files(
                    prefix=f"processed_datasets/{dataset_id}_"
                )
                raw_files = self.storage.list_files(
                    prefix=f"raw_datasets/{dataset_id}_"
                )

                # Also check old processed format
                old_processed_path = f"processed_datasets/{dataset_id}.json"
                old_processed_exists = self.storage.file_exists(old_processed_path)

                possible_paths = (
                    (processed_files if processed_files else [])
                    + ([old_processed_path] if old_processed_exists else [])
                    + (raw_files if raw_files else [])
                )

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

    def validate_dataset(self, dataset: Dict[str, List[Dict]]) -> ValidationResponse:
        """
        Validate a dataset with splits in ChatML format.

        This method performs comprehensive validation of the ChatML format for each split,
        checking:
        - Overall structure
        - Message format
        - Required fields
        - Valid roles
        - Presence of user and assistant messages

        Args:
            dataset (Dict[str, List[Dict]]): The dataset with splits to validate

        Returns:
            ValidationResponse: An object containing:
                - is_valid (bool): Whether the entire dataset is valid
                - errors (List[str]): List of validation errors across all splits
                - warnings (List[str]): List of validation warnings across all splits
                - total_samples (int): Total number of samples across all splits
                - valid_samples (int): Number of valid samples across all splits

        Example:
            >>> validation = service.validate_dataset({"train": [...], "test": [...]})
            >>> print(validation.is_valid)
            True
        """
        try:
            validation = self.converter.validate_chatml_format(dataset)
            return ValidationResponse(
                is_valid=validation["is_valid"],
                errors=validation["errors"],
                warnings=validation["warnings"],
                total_samples=validation["total_samples"],
                valid_samples=validation["valid_samples"],
            )
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

    def _apply_augmentation(
        self, dataset: List[Dict], augmentation_config: Dict
    ) -> List[Dict]:
        """
        Apply data augmentation to the dataset using the augmentation pipeline.

        This method converts the dataset to HuggingFace format, applies augmentation,
        and returns the augmented dataset as a list of dictionaries.

        Args:
            dataset (List[Dict]): The dataset to augment
            augmentation_config (Dict): Configuration for the augmentation pipeline

        Returns:
            List[Dict]: The augmented dataset

        Example augmentation_config:
        {
            "enabled": True,
            "use_eda": True,
            "use_back_translation": True,
            "use_paraphrasing": False,
            "use_synthesis": True,
            "gemini_api_key": "your_gemini_api_key",
            "augmentation_factor": 1.5,
            "eda_alpha_sr": 0.1,
            "paraphrase_model": "humarin/chatgpt_paraphraser_on_T5_base"
        }
        """
        try:
            # Convert to HuggingFace Dataset for augmentation
            hf_dataset = Dataset.from_list(dataset)

            # Apply augmentation pipeline
            augmented_list, result = run_augment_pipeline(
                hf_dataset.to_list(), augmentation_config
            )

            # Log results
            if result.errors:
                for error in result.errors:
                    logger.warning(f"Augmentation error: {error}")

            logger.info(f"Augmentation methods used: {result.methods_used}")
            logger.info(f"Synthesis used: {result.synthesis_used}")
            logger.info(
                f"Dataset augmented from {len(dataset)} to {len(augmented_list)} samples"
            )

            return augmented_list

        except Exception as e:
            logger.error(f"Augmentation failed: {e}")
            logger.warning("Continuing without augmentation")
            return dataset
