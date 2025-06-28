import logging
import re
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class FormatConverter:
    """
    A class that handles conversion of datasets to ChatML format.

    This class provides functionality to convert various dataset formats into the
    ChatML format, which is a standardized format for conversational AI training data.
    It supports conversion from multiple input formats and includes validation and
    preview capabilities.

    The ChatML format follows this structure:
    ```json
    {
        "messages": [
            {"role": "system", "content": "System message"},
            {"role": "user", "content": "User message"},
            {"role": "assistant", "content": "Assistant response"}
        ]
    }
    ```

    Attributes:
        whitespace_pattern (Pattern): Regular expression pattern for normalizing whitespace

    Example:
        >>> converter = FormatConverter()
        >>> chatml_data = converter.convert_to_chatml(dataset, config)
    """

    def __init__(self):
        """
        Initialize the FormatConverter with a whitespace normalization pattern.
        """
        self.whitespace_pattern = re.compile(r"\s+")

    def convert_to_chatml(
        self, dataset: Dict[str, List[Dict]], config: Dict[str, Any]
    ) -> Dict[str, List[Dict]]:
        """
        Convert a dataset with splits to ChatML format.

        This method converts each example in each split of the dataset to the ChatML format based on
        the provided configuration. It handles field mapping, template formatting, and
        ensures the output follows the ChatML structure for all splits.

        Args:
            dataset (Dict[str, List[Dict]]): The input dataset with splits to convert, where keys are split names
                (e.g., 'train', 'test', 'validation') and values are lists of examples
            config (Dict[str, Any]): Configuration for the conversion, including:
                - field_mappings (Dict): Maps input fields to ChatML roles

        Returns:
            Dict[str, List[Dict]]: Dictionary with split names as keys and lists of examples in ChatML format as values,
                where each example has:
                - messages (List[Dict]): List of message objects with role and content

        Raises:
            Exception: If there's an error during conversion

        Example:
            >>> config = {
            ...     "field_mappings": {
            ...         "user_field": {"type": "template", "value": "User: {question}"},
            ...         "assistant_field": {"type": "column", "value": "answer"}
            ...         "system_field": {"type": "template", "value": "You are a helpful assistant."}
            ...     }
            ... }
            >>> dataset = {
            ...     "train": [{"question": "What is ML?", "answer": "Machine Learning"}],
            ...     "test": [{"question": "What is AI?", "answer": "Artificial Intelligence"}]
            ... }
            >>> chatml_data = converter.convert_to_chatml(dataset, config)
            >>> # Returns: {
            ... #     "train": [{"messages": [{"role": "user", "content": "User: What is ML?"}, {"role": "assistant", "content": "Machine Learning"}]}],
            ... #     "test": [{"messages": [{"role": "user", "content": "User: What is AI?"}, {"role": "assistant", "content": "Artificial Intelligence"}]}]
            ... # }
        """
        try:
            if not dataset:
                return {}

            converted_dataset = {}

            for split_name, split_data in dataset.items():
                if not split_data:
                    converted_dataset[split_name] = []
                    continue

                # Check if this split is already in ChatML format
                if self._is_chatml_format(split_data):
                    logger.info(f"Split '{split_name}' is already in ChatML format")
                    converted_dataset[split_name] = split_data
                    continue

                converted_split = []

                for example in split_data:
                    converted_example = self._convert_single_example(example, config)
                    if converted_example:
                        converted_split.append(converted_example)

                converted_dataset[split_name] = converted_split
                logger.info(
                    f"Converted {len(converted_split)} examples in split '{split_name}' to ChatML format"
                )

            logger.info(
                f"Converted dataset with {len(converted_dataset)} splits to ChatML format"
            )
            return converted_dataset

        except Exception as e:
            logger.error(f"Error converting to ChatML format: {str(e)}")
            raise

    def _convert_single_example(self, example: Dict, config: Dict[str, Any]) -> Dict:
        """
        Convert a single example to ChatML format.

        This method converts one example from the input format to ChatML format,
        applying field mappings, templates, and ensuring proper message structure.

        Args:
            example (Dict): The input example to convert
            config (Dict[str, Any]): Configuration for the conversion, including:
                - field_mappings (Dict): Maps input fields to ChatML roles with type and value:
                    - type: "column" or "template"
                    - value: column name or template string with {column} references

        Returns:
            Dict: The converted example in ChatML format, or None if conversion fails
        """
        try:
            field_mappings = config.get("field_mappings", {})
            messages = []

            if "system_field" in field_mappings:
                system_config = field_mappings["system_field"]
                if system_config["type"] == "column":
                    if system_config["value"] in example:
                        system_message = str(example[system_config["value"]])
                        if system_message:
                            messages.append(
                                {"role": "system", "content": system_message}
                            )
                else:  # template
                    try:
                        template_vars = {
                            key: str(value) for key, value in example.items()
                        }
                        system_message = system_config["value"].format(**template_vars)
                        if system_message:
                            messages.append(
                                {"role": "system", "content": system_message}
                            )
                    except (KeyError, ValueError) as e:
                        logger.warning(
                            f"System message template formatting failed: {e}"
                        )

            user_content = self._extract_content(example, field_mappings, "user")
            if user_content:
                messages.append({"role": "user", "content": user_content})

            assistant_content = self._extract_content(
                example, field_mappings, "assistant"
            )
            if assistant_content:
                messages.append({"role": "assistant", "content": assistant_content})

            if (
                len(messages) >= 2
                and any(msg["role"] == "user" for msg in messages)
                and any(msg["role"] == "assistant" for msg in messages)
            ):
                return {"messages": messages}

            return None

        except Exception as e:
            logger.warning(f"Failed to convert example: {e}")
            return None

    def _extract_content(
        self,
        example: Dict,
        field_mappings: Dict,
        role: str,
    ) -> str:
        """
        Extract and format content for a specific role from the example.

        This method extracts content from the input example based on field mappings
        and applies template formatting if specified. It also normalizes whitespace
        in the content.

        Args:
            example (Dict): The input example
            field_mappings (Dict): Maps roles to field mapping configs
            role (str): The role to extract content for ('user' or 'assistant')

        Returns:
            str: The extracted and formatted content

        Example:
            >>> example = {"question": "What is ML?"}
            >>> field_mappings = {
            ...     "user_field": {"type": "column", "value": "question"}
            ... }
            >>> content = converter._extract_content(example, field_mappings, "user")
        """
        field_config = field_mappings.get(f"{role}_field")
        if not field_config:
            return ""

        if field_config["type"] == "column":
            if field_config["value"] not in example:
                return ""
            content = example[field_config["value"]]
        else:  # template
            try:
                template_vars = {key: str(value) for key, value in example.items()}
                content = field_config["value"].format(**template_vars)
            except (KeyError, ValueError) as e:
                logger.warning(f"Template formatting failed: {e}, using raw template")
                content = field_config["value"]

        if isinstance(content, str):
            content = self.whitespace_pattern.sub(" ", content).strip()

        return str(content)

    def _is_chatml_format(self, dataset: List[Dict]) -> bool:
        """
        Check if the dataset is already in ChatML format.

        This method verifies if the dataset follows the ChatML structure by checking
        the first few samples for proper message format and role/content fields.

        Args:
            dataset (List[Dict]): The dataset to check

        Returns:
            bool: True if the dataset is in ChatML format, False otherwise

        Example:
            >>> is_chatml = converter._is_chatml_format(dataset)
            >>> print(is_chatml)
            True
        """
        if not dataset:
            return False

        for sample in dataset[:3]:
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

    def validate_chatml_format(self, dataset: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """
        Validate a dataset with splits in ChatML format and return validation results.

        This method performs comprehensive validation of the ChatML format for each split, checking:
        - Overall structure
        - Message format
        - Required fields
        - Valid roles
        - Presence of user and assistant messages

        Args:
            dataset (Dict[str, List[Dict]]): The dataset with splits to validate, where keys are split names
                (e.g., 'train', 'test', 'validation') and values are lists of examples

        Returns:
            Dict[str, Any]: Validation results containing:
                - is_valid (bool): Whether the entire dataset is valid
                - errors (List[str]): List of validation errors across all splits
                - warnings (List[str]): List of validation warnings across all splits
                - total_samples (int): Total number of samples across all splits
                - valid_samples (int): Number of valid samples across all splits
                - split_results (Dict[str, Dict]): Detailed validation results for each split

        Example:
            >>> dataset = {
            ...     "train": [{"messages": [{"role": "user", "content": "Hello"}]}],
            ...     "test": [{"messages": [{"role": "assistant", "content": "Hi"}]}]
            ... }
            >>> validation = converter.validate_chatml_format(dataset)
            >>> print(validation['is_valid'])
            True
        """
        overall_results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "total_samples": 0,
            "valid_samples": 0,
            "split_results": {},
        }

        if not dataset:
            overall_results["errors"].append("Dataset is empty")
            overall_results["is_valid"] = False
            return overall_results

        for split_name, split_data in dataset.items():
            split_results = self._validate_single_split(split_data, split_name)
            overall_results["split_results"][split_name] = split_results

            # Aggregate results
            overall_results["total_samples"] += split_results["total_samples"]
            overall_results["valid_samples"] += split_results["valid_samples"]
            overall_results["errors"].extend(split_results["errors"])
            overall_results["warnings"].extend(split_results["warnings"])

        if overall_results["errors"]:
            overall_results["is_valid"] = False

        return overall_results

    def _validate_single_split(
        self, split_data: List[Dict], split_name: str
    ) -> Dict[str, Any]:
        """
        Validate a single split of the dataset in ChatML format.

        Args:
            split_data (List[Dict]): The split data to validate
            split_name (str): The name of the split being validated

        Returns:
            Dict[str, Any]: Validation results for this split
        """
        split_results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "total_samples": len(split_data),
            "valid_samples": 0,
        }

        if not split_data:
            split_results["warnings"].append(f"Split '{split_name}' is empty")
            return split_results

        for i, sample in enumerate(split_data):
            sample_errors = []

            if not isinstance(sample, dict):
                sample_errors.append(
                    f"Split '{split_name}', Sample {i}: Not a dictionary"
                )
                continue

            if "messages" not in sample:
                sample_errors.append(
                    f"Split '{split_name}', Sample {i}: Missing 'messages' field"
                )
                continue

            messages = sample["messages"]

            if not isinstance(messages, list):
                sample_errors.append(
                    f"Split '{split_name}', Sample {i}: 'messages' is not a list"
                )
                continue

            if len(messages) == 0:
                sample_errors.append(
                    f"Split '{split_name}', Sample {i}: 'messages' list is empty"
                )
                continue

            has_user = False
            has_assistant = False

            for j, message in enumerate(messages):
                if not isinstance(message, dict):
                    sample_errors.append(
                        f"Split '{split_name}', Sample {i}, Message {j}: Not a dictionary"
                    )
                    continue

                if "role" not in message:
                    sample_errors.append(
                        f"Split '{split_name}', Sample {i}, Message {j}: Missing 'role' field"
                    )
                    continue

                if "content" not in message:
                    sample_errors.append(
                        f"Split '{split_name}', Sample {i}, Message {j}: Missing 'content' field"
                    )
                    continue

                role = message["role"]
                if role not in ["system", "user", "assistant", "tool"]:
                    sample_errors.append(
                        f"Split '{split_name}', Sample {i}, Message {j}: Invalid role '{role}'"
                    )
                    continue

                if role == "user":
                    has_user = True
                elif role == "assistant":
                    has_assistant = True

            if not has_user:
                split_results["warnings"].append(
                    f"Split '{split_name}', Sample {i}: No user message found"
                )
            if not has_assistant:
                split_results["warnings"].append(
                    f"Split '{split_name}', Sample {i}: No assistant message found"
                )

            if not sample_errors:
                split_results["valid_samples"] += 1
            else:
                split_results["errors"].extend(sample_errors)

        if split_results["errors"]:
            split_results["is_valid"] = False

        return split_results

    def preview_conversion(
        self, dataset: List[Dict], config: Dict[str, Any], num_samples: int = 3
    ) -> Dict[str, Any]:
        """
        Preview how the dataset would look after conversion to ChatML format.

        This method provides a preview of the conversion process by converting a small
        sample of the dataset and returning both original and converted examples.

        Args:
            dataset (List[Dict]): The dataset to preview
            config (Dict[str, Any]): Configuration for the conversion
            num_samples (int): Number of samples to include in the preview

        Returns:
            Dict[str, Any]: Preview results containing:
                - original_samples (List[Dict]): Original sample data
                - converted_samples (List[Dict]): Converted sample data
                - conversion_success (bool): Whether conversion was successful
                - samples_converted (int): Number of successfully converted samples
                - samples_failed (int): Number of failed conversions

        Example:
            >>> preview = converter.preview_conversion(dataset, config, num_samples=2)
            >>> print(preview['conversion_success'])
            True
        """
        if not dataset:
            return {"error": "Dataset is empty"}

        # Take a small sample for preview
        sample_data = dataset[:num_samples]

        # Convert the sample
        converted_sample = self.convert_to_chatml(sample_data, config)

        return {
            "original_samples": sample_data,
            "converted_samples": converted_sample,
            "conversion_success": len(converted_sample) > 0,
            "samples_converted": len(converted_sample),
            "samples_failed": len(sample_data) - len(converted_sample),
        }
