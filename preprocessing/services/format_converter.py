import logging
import re
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# TODO: THERE SHOULD BE A WAY THAT PERSON CAN SELECT A COLUMN FOR SYSTEM MESSAGE, RATHER THAN JUST GIVING A STRING


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
        self, dataset: List[Dict], config: Dict[str, Any]
    ) -> List[Dict]:
        """
        Convert a dataset to ChatML format.

        This method converts each example in the dataset to the ChatML format based on
        the provided configuration. It handles field mapping, template formatting, and
        ensures the output follows the ChatML structure.

        Args:
            dataset (List[Dict]): The input dataset to convert
            config (Dict[str, Any]): Configuration for the conversion, including:
                - field_mappings (Dict): Maps input fields to ChatML roles
                - system_message (str): Default system message
                - include_system (bool): Whether to include system message
                - user_template (str): Template for formatting user content

        Returns:
            List[Dict]: List of examples in ChatML format, where each example has:
                - messages (List[Dict]): List of message objects with role and content

        Raises:
            Exception: If there's an error during conversion

        Example:
            >>> config = {
            ...     "field_mappings": {
            ...         "user_field": "question",
            ...         "assistant_field": "answer"
            ...     },
            ...     "system_message": "You are a helpful assistant."
            ... }
            >>> chatml_data = converter.convert_to_chatml(dataset, config)
        """
        try:
            if not dataset:
                return []

            if self._is_chatml_format(dataset):
                logger.info("Dataset is already in ChatML format")
                return dataset

            converted_dataset = []

            for example in dataset:
                converted_example = self._convert_single_example(example, config)
                if converted_example:
                    converted_dataset.append(converted_example)

            logger.info(f"Converted {len(converted_dataset)} examples to ChatML format")
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
            config (Dict[str, Any]): Configuration for the conversion

        Returns:
            Dict: The converted example in ChatML format, or None if conversion fails

        Example:
            >>> example = {"question": "What is ML?", "answer": "Machine Learning"}
            >>> config = {
            ...     "field_mappings": {"user_field": "question", "assistant_field": "answer"},
            ...     "system_message": "You are a helpful assistant."
            ... }
            >>> converted = converter._convert_single_example(example, config)
        """
        try:
            field_mappings = config.get("field_mappings", {})
            system_message = config.get("system_message", "")
            include_system = config.get("include_system", True)
            user_template = config.get("user_template", "{content}")

            messages = []

            if include_system and system_message:
                messages.append({"role": "system", "content": system_message})

            user_content = self._extract_content(
                example, field_mappings, "user", user_template
            )
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
            logger.warning(f"Failed to convert example: {str(e)}")
            return None

    def _extract_content(
        self,
        example: Dict,
        field_mappings: Dict,
        role: str,
        template: str = "{content}",
    ) -> str:
        """
        Extract and format content for a specific role from the example.

        This method extracts content from the input example based on field mappings
        and applies template formatting if specified. It also normalizes whitespace
        in the content.

        Args:
            example (Dict): The input example
            field_mappings (Dict): Maps roles to field names
            role (str): The role to extract content for ('user' or 'assistant')
            template (str): Template string for formatting content

        Returns:
            str: The extracted and formatted content

        Example:
            >>> example = {"question": "What is ML?"}
            >>> field_mappings = {"user_field": "question"}
            >>> content = converter._extract_content(example, field_mappings, "user")
        """
        field_name = field_mappings.get(f"{role}_field")

        if not field_name or field_name not in example:
            return ""

        content = example[field_name]

        if role == "user" and "{" in template:
            try:
                template_vars = {key: str(value) for key, value in example.items()}
                template_vars["content"] = str(content)
                content = template.format(**template_vars)
            except (KeyError, ValueError) as e:
                logger.warning(f"Template formatting failed: {e}, using raw content")
                content = str(content)

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

    def validate_chatml_format(self, dataset: List[Dict]) -> Dict[str, Any]:
        """
        Validate a dataset in ChatML format and return validation results.

        This method performs comprehensive validation of the ChatML format, checking:
        - Overall structure
        - Message format
        - Required fields
        - Valid roles
        - Presence of user and assistant messages

        Args:
            dataset (List[Dict]): The dataset to validate

        Returns:
            Dict[str, Any]: Validation results containing:
                - is_valid (bool): Whether the dataset is valid
                - errors (List[str]): List of validation errors
                - warnings (List[str]): List of validation warnings
                - total_samples (int): Total number of samples
                - valid_samples (int): Number of valid samples

        Example:
            >>> validation = converter.validate_chatml_format(dataset)
            >>> print(validation['is_valid'])
            True
        """
        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "total_samples": len(dataset),
            "valid_samples": 0,
        }

        if not dataset:
            validation_results["errors"].append("Dataset is empty")
            validation_results["is_valid"] = False
            return validation_results

        for i, sample in enumerate(dataset):
            sample_errors = []

            if not isinstance(sample, dict):
                sample_errors.append(f"Sample {i}: Not a dictionary")
                continue

            if "messages" not in sample:
                sample_errors.append(f"Sample {i}: Missing 'messages' field")
                continue

            messages = sample["messages"]

            if not isinstance(messages, list):
                sample_errors.append(f"Sample {i}: 'messages' is not a list")
                continue

            if len(messages) == 0:
                sample_errors.append(f"Sample {i}: 'messages' list is empty")
                continue

            has_user = False
            has_assistant = False

            for j, message in enumerate(messages):
                if not isinstance(message, dict):
                    sample_errors.append(f"Sample {i}, Message {j}: Not a dictionary")
                    continue

                if "role" not in message:
                    sample_errors.append(
                        f"Sample {i}, Message {j}: Missing 'role' field"
                    )
                    continue

                if "content" not in message:
                    sample_errors.append(
                        f"Sample {i}, Message {j}: Missing 'content' field"
                    )
                    continue

                role = message["role"]
                if role not in ["system", "user", "assistant", "tool"]:
                    sample_errors.append(
                        f"Sample {i}, Message {j}: Invalid role '{role}'"
                    )
                    continue

                if role == "user":
                    has_user = True
                elif role == "assistant":
                    has_assistant = True

            if not has_user:
                validation_results["warnings"].append(
                    f"Sample {i}: No user message found"
                )
            if not has_assistant:
                validation_results["warnings"].append(
                    f"Sample {i}: No assistant message found"
                )

            if not sample_errors:
                validation_results["valid_samples"] += 1
            else:
                validation_results["errors"].extend(sample_errors)

        if validation_results["errors"]:
            validation_results["is_valid"] = False

        return validation_results

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
