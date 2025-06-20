import json
import io
import logging
import pandas as pd
from typing import List, Dict
from datasets import load_dataset
from storage.base import StorageInterface
from schema import (
    PreprocessingConfig,
    ManualSplitConfig,
    NoSplitConfig,
    HFSplitConfig,
)

logger = logging.getLogger(__name__)


class DatasetLoader:
    """
    A class that handles loading datasets from uploaded files and Hugging Face datasets.

    This class provides functionality to load datasets from different sources and formats,
    supporting various file types like CSV, JSON, JSONL, Excel, and Parquet files.

    Attributes:
        storage (StorageInterface): An interface for storage operations, used to access
            uploaded files and manage dataset storage.

    Example:
        >>> storage = StorageInterface()
        >>> loader = DatasetLoader(storage)
        >>> dataset = await loader.load_dataset("upload", "my_dataset_id")
    """

    def __init__(self, storage: StorageInterface):
        """
        Initialize the DatasetLoader with a storage interface.

        Args:
            storage (StorageInterface): An interface for storage operations that provides
                methods for accessing and managing stored datasets.
        """
        self.storage = storage

    async def load_dataset(
        self, dataset_source: str, dataset_id: str, config: PreprocessingConfig
    ) -> List[Dict]:
        """
        Load a dataset from the specified source.

        This method supports loading datasets from three different sources:
        1. 'upload': Loads a user-uploaded dataset from storage
        2. 'huggingface': Loads a dataset from Hugging Face's dataset hub

        Args:
            dataset_source (str): The source of the dataset. Must be one of:
                - 'upload': For user-uploaded datasets
                - 'huggingface': For datasets from Hugging Face
            dataset_id (str): The identifier for the dataset:
                - For 'upload': The file ID of the uploaded dataset
                - For 'huggingface': The Hugging Face dataset name
            config (PreprocessingConfig): Configuration for processing, including:


        Returns:
            List[Dict]: A list of dictionaries containing the dataset samples.

        Raises:
            ValueError: If an invalid dataset_source is provided.
            FileNotFoundError: If the uploaded dataset is not found.
            Exception: For other errors during dataset loading.

        Example:
            >>> dataset = await loader.load_dataset("huggingface", "squad", sample_size=100)
        """
        if dataset_source == "upload" and isinstance(
            config.split_config, HFSplitConfig
        ):
            raise ValueError(
                "HuggingFace split configuration (hf_split) cannot be used with uploaded datasets. Use 'manual_split' or 'no_split' instead."
            )

        try:
            if dataset_source == "upload":
                return await self._load_uploaded_dataset(dataset_id, config)
            elif dataset_source == "huggingface":
                return await self._load_huggingface_dataset(dataset_id, config)
            else:
                raise ValueError(f"Invalid dataset_source: {dataset_source}")

        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise

    async def _load_uploaded_dataset(
        self, dataset_id: str, config: PreprocessingConfig
    ) -> Dict[str, List[Dict]]:
        """
        Load a dataset that was previously uploaded by the user.

        This method searches for the uploaded file in storage and parses it based on its
        file extension. Supports various file formats including CSV, JSON, JSONL, Excel,
        and Parquet files. The data is then processed according to the split configuration:
        - NoSplitConfig: Returns all data as train split with optional sampling
        - ManualSplitConfig: Samples the data and splits into train/test sets

        Args:
            dataset_id (str): The unique identifier of the uploaded dataset.
            config (PreprocessingConfig): Configuration for processing, including split_config.

        Returns:
            Dict[str, List[Dict]]: A dictionary with split names as keys and dataset samples as values.

        Raises:
            FileNotFoundError: If no file is found with the given dataset_id.
            ValueError: If the file format cannot be parsed.

        Example:
            >>> dataset = await loader._load_uploaded_dataset("123e4567-e89b-12d3-a456-426614174000", config)
        """
        print(self.storage.list_files())
        files = self.storage.list_files(prefix=f"raw_datasets/{dataset_id}_")
        if not files:
            raise FileNotFoundError("Uploaded dataset not found")

        file_path = files[0]
        file_content = await self.storage.download_data(file_path)

        filename = file_path.split("_", 1)[1]
        data = self._parse_file_content(file_content, filename)

        # No split configuration - return all data as train split
        if isinstance(config.split_config, NoSplitConfig):
            sample_size = config.split_config.sample_size
            if sample_size and len(data) > sample_size:
                # Shuffle and sample the data
                import random

                random.shuffle(data)
                data = data[:sample_size]
            return {"train": data}

        # Manual split configuration - sample and split into train/test
        elif isinstance(config.split_config, ManualSplitConfig):
            sample_size = config.split_config.sample_size
            test_size = config.split_config.test_size

            if sample_size and len(data) > sample_size:
                # Shuffle and sample the data
                import random

                random.shuffle(data)
                data = data[:sample_size]

            # If no test_size provided, return all data as train split
            if test_size is None:
                return {"train": data}

            # Split into train/test
            split_index = int(len(data) * (1 - test_size))
            train_data = data[:split_index]
            test_data = data[split_index:]

            return {"train": train_data, "test": test_data}

        # No split config provided - return all data as train split
        else:
            return {"train": data}

    async def _load_huggingface_dataset(
        self, dataset_name: str, config: PreprocessingConfig
    ) -> Dict[str, List[Dict]]:
        """
        Load a dataset from Hugging Face's dataset hub.

        This method downloads and loads a dataset from Hugging Face based on the split_config:
        - NoSplitConfig: Loads train split with sampling
        - ManualSplitConfig: Loads train split, samples it, then splits into train/test
        - HFSplitConfig: Loads all specified splits from Hugging Face

        Args:
            dataset_name (str): The name of the dataset on Hugging Face (e.g., 'squad', 'glue').
            config (PreprocessingConfig): Configuration for processing, including split_config.

        Returns:
            Dict[str, List[Dict]]: A dictionary with split names as keys and dataset samples as values.

        Raises:
            Exception: If there's an error loading the dataset from Hugging Face.

        Example:
            >>> dataset = await loader._load_huggingface_dataset("squad", config)
        """
        try:
            # No split configuration - just get train split
            if isinstance(config.split_config, NoSplitConfig):
                dataset = load_dataset(dataset_name, split="train")
                sample_size = config.split_config.sample_size
                if sample_size and len(dataset) > sample_size:
                    dataset = dataset.shuffle().select(range(sample_size))
                return {"train": list(dataset)}

            # Manual split configuration - get train split, sample it, then split into train/test
            elif isinstance(config.split_config, ManualSplitConfig):
                dataset = load_dataset(dataset_name, split="train")
                sample_size = config.split_config.sample_size
                test_size = config.split_config.test_size

                if sample_size and len(dataset) > sample_size:
                    dataset = dataset.shuffle().select(range(sample_size))

                # Convert to list
                dataset_list = list(dataset)

                # If no test_size provided, return all data as train split
                if test_size is None:
                    return {"train": dataset_list}

                # Split into train/test
                split_index = int(len(dataset_list) * (1 - test_size))
                train_data = dataset_list[:split_index]
                test_data = dataset_list[split_index:]

                return {"train": train_data, "test": test_data}

            # HF split configuration - get all specified splits
            elif isinstance(config.split_config, HFSplitConfig):
                splits = config.split_config.splits
                if not splits:
                    # Default to train if no splits specified
                    dataset = load_dataset(dataset_name, split="train")
                    return {"train": list(dataset)}

                dataset = {}
                for split in splits:
                    dataset[split] = list(load_dataset(dataset_name, split=split))
                return dataset

            # No split config provided - default to train split
            else:
                dataset = load_dataset(dataset_name, split="train")
                return {"train": list(dataset)}

        except Exception as e:
            logger.error(f"Error loading Hugging Face dataset: {str(e)}")
            raise

    def _parse_file_content(self, content: str, filename: str) -> List[Dict]:
        """
        Parse file content based on its extension.

        This method supports parsing various file formats:
        - CSV: Comma-separated values
        - JSON: JavaScript Object Notation
        - JSONL: JSON Lines format
        - Excel: .xlsx and .xls files
        - Parquet: Apache Parquet format
        - Text: Plain text files

        Args:
            content (str): The raw content of the file.
            filename (str): The name of the file, used to determine the file format.

        Returns:
            List[Dict]: A list of dictionaries containing the parsed data.

        Raises:
            ValueError: If the file format cannot be parsed.

        Example:
            >>> data = loader._parse_file_content(file_content, "data.csv")
        """
        try:
            if filename.endswith(".csv"):
                df = pd.read_csv(io.StringIO(content))
                return df.to_dict("records")
            elif filename.endswith(".json"):
                return json.loads(content)
            elif filename.endswith(".jsonl"):
                return [
                    json.loads(line)
                    for line in content.strip().split("\n")
                    if line.strip()
                ]
            elif filename.endswith((".xlsx", ".xls")):
                df = pd.read_excel(io.BytesIO(content.encode()))
                return df.to_dict("records")
            elif filename.endswith(".parquet"):
                df = pd.read_parquet(io.BytesIO(content.encode()))
                return df.to_dict("records")
            else:
                return [{"text": content}]
        except Exception as e:
            logger.error(f"Error parsing file content: {str(e)}")
            raise ValueError(f"Unable to parse file format: {filename}")
