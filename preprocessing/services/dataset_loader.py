import logging
from datasets import load_dataset, DatasetDict
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
    ) -> DatasetDict:
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
            DatasetDict: A dictionary with split names as keys and dataset samples as values.

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
    ) -> DatasetDict:
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
            DatasetDict: A dictionary with split names as keys and dataset samples as values.

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
        filename = file_path.split("_", 1)[1]

        # find the type of file
        file_type = filename.split(".")[-1]
        if (
            file_type != "csv"
            and file_type != "json"
            and file_type != "jsonl"
            and file_type != "parquet"
            and file_type != "xlsx"
            and file_type != "xls"
            and file_type != "txt"
        ):
            raise ValueError("Invalid file type")

        # No split configuration - return all data as train split
        if isinstance(config.split_config, NoSplitConfig):
            sample_size = config.split_config.sample_size
            dataset = load_dataset(file_type, data_files=file_path)

            if sample_size and dataset["train"].num_rows > sample_size:
                shuffled_dataset = (
                    dataset["train"].shuffle(seed=42).select(range(sample_size))
                )
                dataset = DatasetDict({"train": shuffled_dataset})

            return dataset

        # Manual split configuration - sample and split into train/test
        elif isinstance(config.split_config, ManualSplitConfig):
            sample_size = config.split_config.sample_size
            test_size = config.split_config.test_size

            dataset = load_dataset(file_type, data_files=file_path)
            train_dataset = dataset["train"]

            # if sample_size and test_size are provided, sample and split into train/test
            if sample_size and train_dataset.num_rows > sample_size and test_size:
                train_size = int(sample_size * (1 - test_size))
                shuffled_dataset = train_dataset.shuffle(seed=42)
                train_split = shuffled_dataset.select(range(train_size))
                test_split = shuffled_dataset.select(range(train_size, sample_size))
                return DatasetDict({"train": train_split, "test": test_split})

            # if sample_size is provided, sample the data
            elif sample_size and train_dataset.num_rows > sample_size:
                sampled_dataset = train_dataset.shuffle(seed=42).select(
                    range(sample_size)
                )
                return DatasetDict({"train": sampled_dataset})

            # if test_size is provided, split into train/test
            elif test_size:
                train_size = int(train_dataset.num_rows * (1 - test_size))
                shuffled_dataset = train_dataset.shuffle(seed=42)
                train_split = shuffled_dataset.select(range(train_size))
                test_split = shuffled_dataset.select(
                    range(train_size, train_dataset.num_rows)
                )
                return DatasetDict({"train": train_split, "test": test_split})

            # if no sampling or splitting is specified, return all data as train
            else:
                return dataset

    async def _load_huggingface_dataset(
        self, dataset_name: str, config: PreprocessingConfig
    ) -> DatasetDict:
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
            DatasetDict: A dictionary with split names as keys and dataset samples as values.

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
                if sample_size and dataset.num_rows > sample_size:
                    dataset = dataset.shuffle(seed=42).select(range(sample_size))
                return DatasetDict({"train": dataset})

            # Manual split configuration - get train split, sample it, then split into train/test
            elif isinstance(config.split_config, ManualSplitConfig):
                dataset = load_dataset(dataset_name, split="train")
                sample_size = config.split_config.sample_size
                test_size = config.split_config.test_size

                # if sample_size and test_size are provided, sample and split into train/test
                if sample_size and dataset.num_rows > sample_size and test_size:
                    train_size = int(sample_size * (1 - test_size))
                    shuffled_dataset = dataset.shuffle(seed=42)
                    train_split = shuffled_dataset.select(range(train_size))
                    test_split = shuffled_dataset.select(range(train_size, sample_size))
                    return DatasetDict({"train": train_split, "test": test_split})

                # if sample_size is provided, sample the data
                elif sample_size and dataset.num_rows > sample_size:
                    sampled_dataset = dataset.shuffle(seed=42).select(
                        range(sample_size)
                    )
                    return DatasetDict({"train": sampled_dataset})

                # if test_size is provided, split into train/test
                elif test_size:
                    train_size = int(dataset.num_rows * (1 - test_size))
                    shuffled_dataset = dataset.shuffle(seed=42)
                    train_split = shuffled_dataset.select(range(train_size))
                    test_split = shuffled_dataset.select(
                        range(train_size, dataset.num_rows)
                    )
                    return DatasetDict({"train": train_split, "test": test_split})

                # if no sampling or splitting is specified, return all data as train
                else:
                    return DatasetDict({"train": dataset})

            # HF split configuration - get all specified splits
            elif isinstance(config.split_config, HFSplitConfig):
                splits = config.split_config.splits

                # if no splits are provided, return all splits
                if not splits:
                    dataset = load_dataset(dataset_name)
                    return dataset
                else:
                    dataset = load_dataset(dataset_name, split=splits)
                    return dataset

            else:
                raise ValueError("Invalid split configuration")

        except Exception as e:
            logger.error(f"Error loading Hugging Face dataset: {str(e)}")
            raise
