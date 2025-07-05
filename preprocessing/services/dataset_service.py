import logging
from typing import List, Dict, Optional
from storage.base import StorageInterface
from .dataset_handler import DatasetHandler
from .dataset_loader import DatasetLoader
from .format_converter import FormatConverter
from augmentation import run_augment_pipeline
from schema import (
    DatasetUploadResponse,
    ProcessingResult,
    PreprocessingConfig,
    DatasetsInfoResponse,
    DatasetInfoSample,
    DatasetInfoResponse,
)
from datasets import Dataset
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class DatasetService:
    """
    A class that orchestrates all dataset operations and provides a unified interface
    for dataset management and processing.

    This service combines functionality from multiple components:
    - DatasetHandler: Handles file uploads
    - DatasetLoader: Loads datasets from various sources
    - FormatConverter: Converts datasets to ChatML format

    The service provides high-level operations for:
    - Uploading datasets
    - Processing datasets to ChatML format
    - Managing processed datasets

    Attributes:
        storage (StorageInterface): Interface for storage operations
        uploader (DatasetHandler): Handles dataset uploads
        loader (DatasetLoader): Handles dataset loading
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
        self.handler = DatasetHandler(storage)
        self.loader = DatasetLoader(storage)
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
        return await self.handler.upload_dataset(file_data, filename, metadata)

    async def process_dataset(
        self,
        dataset_name: str,
        dataset_source: str,
        dataset_id: str,
        config: PreprocessingConfig,
        dataset_subset: Optional[str] = "default",
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
            dataset_name (str): The name of the dataset, used for the processed dataset name
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
            ...     "my_dataset",
            ...     "upload",
            ...     "my_dataset_id",
            ...     config,
            ...     split_config=HFSplitConfig(type="hf_split", splits=["train", "test"]),
            ... )
        """
        try:
            if await self.handler.does_dataset_exist(dataset_name):
                raise ValueError(
                    f"Dataset {dataset_name} already exists. Please use a different name."
                )

            # Load dataset with splits
            dataset = await self.loader.load_dataset(
                dataset_source, dataset_id, config, dataset_subset
            )

            if not dataset:
                raise ValueError("Dataset is empty or could not be loaded")

            config_dict = config.dict()

            processed_dataset = self.converter.convert_to_chatml(dataset, config_dict)

            if not processed_dataset:
                raise ValueError("No samples could be converted to ChatML format")

            # Apply data augmentation if the user created a config specification
            augmentation_config = config_dict.get("augmentation_config", {})
            if augmentation_config:
                processed_dataset = self._apply_augmentation(
                    processed_dataset, augmentation_config
                )

            # Save all splits
            await self.handler.upload_processed_dataset(
                processed_dataset,
                dataset_name,
                dataset_id,
                dataset_subset,
                config,
                dataset_source,
            )

            return ProcessingResult(
                dataset_name=dataset_name,
                dataset_subset=dataset_subset,
                dataset_source=dataset_source,
                dataset_id=dataset_id,
                num_examples=len(processed_dataset["train"]),
                created_at=datetime.now().isoformat(),
                splits=list(processed_dataset.keys()),
            )

        except Exception as e:
            logger.error(f"Error processing dataset: {str(e)}")
            raise

    async def get_datasets_info(self) -> DatasetsInfoResponse:
        """
        Get information about all the processed datasets.

        This method:
        1. Lists all directories in the processed_datasets/ folder
        2. For each dataset directory, reads the metadata.json file
        3. Extracts the required information and returns it as a DatasetsInfoResponse

        Returns:
            DatasetsInfoResponse: An object containing a list of DatasetInfoSample objects
                with information about each processed dataset

        Raises:
            Exception: If there's an error reading the datasets information
        """
        try:
            all_files = self.storage.list_files(prefix="processed_datasets/")

            dataset_names = set()

            for file_path in all_files:
                if file_path.startswith("processed_datasets/"):
                    parts = file_path.split("/")
                    if len(parts) >= 3:
                        dataset_names.add(parts[1])

            datasets_info = []

            for dataset_name in dataset_names:
                try:
                    metadata_path = f"processed_datasets/{dataset_name}/metadata.json"

                    if not self.storage.file_exists(metadata_path):
                        logger.warning(
                            f"Metadata file not found for dataset: {dataset_name}"
                        )
                        continue

                    metadata_content = await self.storage.download_data(metadata_path)
                    metadata = json.loads(metadata_content)

                    total_examples = 0
                    for split in metadata.get("splits", []):
                        total_examples += split.get("num_rows", 0)

                    dataset_info = DatasetInfoSample(
                        dataset_name=metadata.get("dataset_name"),
                        dataset_subset=metadata.get("dataset_subset"),
                        dataset_source=metadata.get("dataset_source"),
                        dataset_id=metadata.get("dataset_id"),
                        num_examples=total_examples,
                        created_at=metadata.get("upload_date"),
                        splits=[
                            split.get("split_name")
                            for split in metadata.get("splits", [])
                        ],
                    )

                    datasets_info.append(dataset_info)

                except Exception as e:
                    logger.error(
                        f"Error reading metadata for dataset {dataset_name}: {str(e)}"
                    )
                    continue

            return DatasetsInfoResponse(datasets=datasets_info)

        except Exception as e:
            logger.error(f"Error getting datasets info: {str(e)}")
            raise

    async def get_dataset_info(self, dataset_name: str) -> DatasetInfoResponse:
        """
        Get information about a dataset including samples from each split.
        """
        try:
            import io
            import pyarrow.parquet as pq

            metadata_path = f"processed_datasets/{dataset_name}/metadata.json"
            metadata_content = await self.storage.download_data(metadata_path)
            metadata = json.loads(metadata_content)

            # Get splits information with samples
            splits_with_samples = []
            for split_info in metadata.get("splits", []):
                split_name = split_info.get("split_name")
                split_path = f"processed_datasets/{dataset_name}/{split_name}.parquet"

                # Get samples from the split
                samples = []
                try:
                    if self.storage.file_exists(split_path):
                        split_data = await self.storage.download_binary_data(split_path)
                        table = pq.read_table(io.BytesIO(split_data))
                        samples = table.slice(0, 5).to_pylist()
                except Exception as e:
                    logger.warning(
                        f"Could not read samples from split {split_name}: {str(e)}"
                    )

                splits_with_samples.append(
                    {
                        "split_name": split_name,
                        "num_rows": split_info.get("num_rows", 0),
                        "path": split_info.get("path", ""),
                        "samples": samples,
                    }
                )

            return DatasetInfoResponse(
                dataset_name=metadata.get("dataset_name"),
                dataset_subset=metadata.get("dataset_subset"),
                dataset_source=metadata.get("dataset_source"),
                dataset_id=metadata.get("dataset_id"),
                created_at=metadata.get("upload_date"),
                splits=splits_with_samples,
            )

        except Exception as e:
            logger.error(f"Error getting dataset info: {str(e)}")
            raise

    def _augment_split(
        self, split_data: List[Dict], augmentation_config: Dict
    ) -> List[Dict]:
        """
        Apply augmentation to a single split or list of data.

        Args:
            split_data (List[Dict]): The data to augment
            augmentation_config (Dict): Configuration for the augmentation pipeline

        Returns:
            List[Dict]: The augmented data
        """
        try:
            if not split_data:
                return split_data

            # Convert to HuggingFace Dataset for augmentation
            hf_dataset = Dataset.from_list(split_data)

            # Apply augmentation pipeline
            augmented_list, result = run_augment_pipeline(
                hf_dataset.to_list(), augmentation_config
            )

            # Log results
            if result.errors:
                for error in result.errors:
                    logger.warning(f"Augmentation error: {error}")

            return augmented_list

        except Exception as e:
            logger.error(f"Split augmentation failed: {e}")
            return split_data
