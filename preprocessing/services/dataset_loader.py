import json
import io
import logging
import pandas as pd
from typing import List, Dict, Optional
from datasets import load_dataset
from storage.base import StorageInterface

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
        self, dataset_source: str, dataset_id: str, sample_size: Optional[int] = None
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
            sample_size (Optional[int]): The number of samples to load. If None, loads the entire dataset.
                Only applicable for Hugging Face datasets.

        Returns:
            List[Dict]: A list of dictionaries containing the dataset samples.

        Raises:
            ValueError: If an invalid dataset_source is provided.
            FileNotFoundError: If the uploaded dataset is not found.
            Exception: For other errors during dataset loading.

        Example:
            >>> dataset = await loader.load_dataset("huggingface", "squad", sample_size=100)
        """
        try:
            if dataset_source == "upload":
                return await self._load_uploaded_dataset(dataset_id)
            elif dataset_source == "huggingface":
                return await self._load_huggingface_dataset(dataset_id, sample_size)
            else:
                raise ValueError(f"Invalid dataset_source: {dataset_source}")

        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise

    async def _load_uploaded_dataset(self, dataset_id: str) -> List[Dict]:
        """
        Load a dataset that was previously uploaded by the user.

        This method searches for the uploaded file in storage and parses it based on its
        file extension. Supports various file formats including CSV, JSON, JSONL, Excel,
        and Parquet files.

        Args:
            dataset_id (str): The unique identifier of the uploaded dataset.

        Returns:
            List[Dict]: A list of dictionaries containing the dataset samples.

        Raises:
            FileNotFoundError: If no file is found with the given dataset_id.
            ValueError: If the file format cannot be parsed.

        Example:
            >>> dataset = await loader._load_uploaded_dataset("123e4567-e89b-12d3-a456-426614174000")
        """
        print(self.storage.list_files())
        files = self.storage.list_files(prefix=f"raw_datasets/{dataset_id}_")
        if not files:
            raise FileNotFoundError("Uploaded dataset not found")

        file_path = files[0]
        file_content = await self.storage.download_data(file_path)

        filename = file_path.split("_", 1)[1]
        return self._parse_file_content(file_content, filename)

    # TODO: HAVEN'T TESTED THIS YET
    async def _load_huggingface_dataset(
        self, dataset_name: str, sample_size: Optional[int] = None
    ) -> List[Dict]:
        """
        Load a dataset from Hugging Face's dataset hub.

        This method downloads and loads a dataset from Hugging Face, with optional
        sampling functionality. The dataset is loaded in the 'train' split by default.

        Args:
            dataset_name (str): The name of the dataset on Hugging Face (e.g., 'squad', 'glue').
            sample_size (Optional[int]): The number of samples to load. If None, loads the entire dataset.

        Returns:
            List[Dict]: A list of dictionaries containing the dataset samples.

        Raises:
            Exception: If there's an error loading the dataset from Hugging Face.

        Example:
            >>> dataset = await loader._load_huggingface_dataset("squad", sample_size=1000)
        """
        try:
            dataset = load_dataset(dataset_name, split="train")

            if sample_size and len(dataset) > sample_size:
                dataset = dataset.shuffle().select(range(sample_size))

            return list(dataset)

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
