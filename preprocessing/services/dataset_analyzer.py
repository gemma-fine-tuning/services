import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class DatasetAnalyzer:
    """
    A class that provides comprehensive analysis and metadata extraction for datasets.

    This class offers functionality to analyze datasets in various formats and extract
    useful metadata such as column statistics, format detection, and ChatML mapping
    suggestions. It's particularly useful for understanding dataset structure and
    preparing data for model training.

    The analyzer can handle different dataset formats including:
    - Question-Answer pairs
    - Instruction-following examples
    - Text classification datasets
    - Conversational datasets
    - Custom formats

    Example:
        >>> analyzer = DatasetAnalyzer()
        >>> analysis = analyzer.analyze_dataset(my_dataset)
    """

    def analyze_dataset(self, dataset: List[Dict]) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of a dataset and return detailed metadata.

        This method analyzes the dataset structure, content, and format to provide
        useful insights for data processing and model training. It includes:
        - Basic statistics (total samples, columns)
        - Column analysis (data types, null values, sample values)
        - Format detection
        - Sample data preview

        Args:
            dataset (List[Dict]): The dataset to analyze, where each item is a dictionary
                representing a sample.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - total_samples (int): Total number of samples in the dataset
                - columns (List[str]): List of all column names
                - sample_data (List[Dict]): First 3 samples from the dataset
                - column_info (Dict): Detailed information about each column
                - format_type (str): Detected format type of the dataset

        Example:
            >>> analysis = analyzer.analyze_dataset(my_dataset)
            >>> print(analysis['total_samples'])
            1000
            >>> print(analysis['format_type'])
            'question_answer'
        """
        if not dataset:
            return {
                "total_samples": 0,
                "columns": [],
                "sample_data": [],
                "column_info": {},
                "format_type": "unknown",
            }

        try:
            total_samples = len(dataset)

            all_columns = set()
            for sample in dataset:
                if isinstance(sample, dict):
                    all_columns.update(sample.keys())

            columns = sorted(list(all_columns))

            column_info = self._analyze_columns(dataset, columns)

            format_type = self._detect_format_type(dataset, columns)

            sample_data = dataset[:3]

            return {
                "total_samples": total_samples,
                "columns": columns,
                "sample_data": sample_data,
                "column_info": column_info,
                "format_type": format_type,
            }

        except Exception as e:
            logger.error(f"Error analyzing dataset: {str(e)}")
            raise

    def _analyze_columns(
        self, dataset: List[Dict], columns: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze each column in the dataset and return detailed statistics.

        This method provides comprehensive analysis for each column including:
        - Total count of samples
        - Number of non-null values
        - Number of null values
        - Null percentage
        - Data types present
        - Sample values

        Args:
            dataset (List[Dict]): The dataset to analyze
            columns (List[str]): List of column names to analyze

        Returns:
            Dict[str, Any]: A dictionary mapping column names to their analysis results.
                Each column's analysis includes:
                - total_count (int): Total number of samples
                - non_null_count (int): Number of non-null values
                - null_count (int): Number of null values
                - null_percentage (float): Percentage of null values
                - data_types (List[str]): List of data types present
                - sample_values (List): First 5 non-null values

        Example:
            >>> column_info = analyzer._analyze_columns(dataset, ['question', 'answer'])
            >>> print(column_info['question']['null_percentage'])
            0.0
        """
        column_info = {}

        for column in columns:
            values = []
            null_count = 0

            for sample in dataset:
                if isinstance(sample, dict):
                    value = sample.get(column)
                    if value is None or value == "":
                        null_count += 1
                    else:
                        values.append(value)

            total_count = len(dataset)
            non_null_count = len(values)

            data_types = set()
            for value in values[:100]:  # Sample first 100 values
                data_types.add(type(value).__name__)

            sample_values = values[:5] if values else []

            column_info[column] = {
                "total_count": total_count,
                "non_null_count": non_null_count,
                "null_count": null_count,
                "null_percentage": (null_count / total_count) * 100
                if total_count > 0
                else 0,
                "data_types": list(data_types),
                "sample_values": sample_values,
            }

        return column_info

    def _detect_format_type(self, dataset: List[Dict], columns: List[str]) -> str:
        """
        Detect the format type of the dataset based on its structure and column names.

        This method analyzes the dataset structure to identify common format patterns:
        - ChatML: Messages with role/content structure
        - Question-Answer: Pairs of questions and answers
        - Instruction: Instruction-following examples
        - Classification: Text classification datasets
        - Conversation: Conversational datasets
        - Custom: Other formats

        Args:
            dataset (List[Dict]): The dataset to analyze
            columns (List[str]): List of column names in the dataset

        Returns:
            str: The detected format type, one of:
                - 'chatml': ChatML format
                - 'question_answer': Question-answer pairs
                - 'instruction': Instruction-following examples
                - 'classification': Text classification
                - 'conversation': Conversational format
                - 'custom': Other formats
                - 'unknown': If format cannot be determined

        Example:
            >>> format_type = analyzer._detect_format_type(dataset, ['question', 'answer'])
            >>> print(format_type)
            'question_answer'
        """
        if not dataset or not columns:
            return "unknown"

        if self._is_chatml_format(dataset):
            return "chatml"

        column_set = set(col.lower() for col in columns)

        # TODO: MIGHT JUST REMOVE THIS, FRONTEND DOES NOT NEED THIS
        qa_patterns = [
            {"question", "answer"},
            {"q", "a"},
            {"query", "response"},
            {"input", "output"},
            {"prompt", "completion"},
        ]

        for pattern in qa_patterns:
            if pattern.issubset(column_set):
                return "question_answer"

        instruction_patterns = [
            {"instruction", "output"},
            {"instruction", "response"},
            {"task", "solution"},
        ]

        for pattern in instruction_patterns:
            if pattern.issubset(column_set):
                return "instruction"

        if {"text", "label"}.issubset(column_set) or {"sentence", "label"}.issubset(
            column_set
        ):
            return "classification"

        if "conversation" in column_set or "dialogue" in column_set:
            return "conversation"

        return "custom"

    def _is_chatml_format(self, dataset: List[Dict]) -> bool:
        """
        Check if the dataset is already in ChatML format.

        This method verifies if the dataset follows the ChatML format structure:
        - Each sample has a 'messages' field
        - Messages are a list of dictionaries
        - Each message has 'role' and 'content' fields
        - Roles are one of: 'system', 'user', 'assistant', 'tool'

        Args:
            dataset (List[Dict]): The dataset to check

        Returns:
            bool: True if the dataset is in ChatML format, False otherwise

        Example:
            >>> is_chatml = analyzer._is_chatml_format(dataset)
            >>> print(is_chatml)
            True
        """
        if not dataset:
            return False

        for sample in dataset[:5]:  # Check first 5 samples
            if not isinstance(sample, dict):
                continue

            if "messages" in sample:
                messages = sample["messages"]
                if isinstance(messages, list) and len(messages) > 0:
                    for msg in messages:
                        if isinstance(msg, dict) and "role" in msg and "content" in msg:
                            if msg["role"] in ["system", "user", "assistant", "tool"]:
                                return True

        return False

    def get_column_statistics(self, dataset: List[Dict], column: str) -> Dict[str, Any]:
        """
        Get detailed statistics for a specific column in the dataset.

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
            >>> stats = analyzer.get_column_statistics(dataset, 'question')
            >>> print(stats['avg_length'])
            45.6
        """
        if not dataset:
            return {}

        values = []
        for sample in dataset:
            if isinstance(sample, dict) and column in sample:
                value = sample[column]
                if value is not None and value != "":
                    values.append(value)

        if not values:
            return {"error": "No valid values found for column"}

        stats = {
            "total_values": len(values),
            "unique_values": len(set(str(v) for v in values)),
            "sample_values": values[:10],
        }

        if all(isinstance(v, str) for v in values[:10]):
            lengths = [len(str(v)) for v in values]
            stats.update(
                {
                    "avg_length": sum(lengths) / len(lengths),
                    "min_length": min(lengths),
                    "max_length": max(lengths),
                }
            )

        return stats
