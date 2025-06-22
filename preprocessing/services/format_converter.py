import logging
import re
from typing import List, Dict, Any
from datasets import DatasetDict

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
        self, dataset: DatasetDict, config: Dict[str, Any]
    ) -> DatasetDict:
        """
        Convert a dataset with splits to ChatML format.

        This method converts each example in each split of the dataset to the ChatML format based on
        the provided configuration. It handles field mapping, template formatting, and
        ensures the output follows the ChatML structure for all splits.

        Args:
            dataset (DatasetDict): The input dataset with splits to convert, where keys are split names
                (e.g., 'train', 'test', 'validation') and values are lists of examples
            config (Dict[str, Any]): Configuration for the conversion, including:
                - field_mappings (Dict): Maps input fields to ChatML roles

        Returns:
            DatasetDict: Dictionary with split names as keys and lists of examples in ChatML format as values,
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
            >>> dataset = DatasetDict({
            ...     "train": Dataset.from_list([{"question": "What is ML?", "answer": "Machine Learning"}]),
            ...     "test": Dataset.from_list([{"question": "What is AI?", "answer": "Artificial Intelligence"}])
            ... })
            >>> chatml_data = converter.convert_to_chatml(dataset, config)
            >>> # Returns: {
            ... #     "train": Dataset.from_list([{"messages": [{"role": "user", "content": "User: What is ML?"}, {"role": "assistant", "content": "Machine Learning"}]}]),
            ... #     "test": Dataset.from_list([{"messages": [{"role": "user", "content": "User: What is AI?"}, {"role": "assistant", "content": "Artificial Intelligence"}]}]),
            ... # }
        """
        try:
            if not dataset:
                raise ValueError("Dataset is empty")

            transformed_dataset = dataset.map(
                self._convert_single_example,
                config,
                batched=True,
                batch_size=8,
                remove_columns=dataset.column_names,
            )
            # Filter out failed conversions (empty dictionaries)
            transformed_dataset = transformed_dataset.filter(
                lambda x: "messages" in x and x["messages"]
            )

            return transformed_dataset

        except Exception as e:
            logger.error(f"Error converting to ChatML format: {str(e)}")
            raise

    def _convert_single_example(self, example: Dict, config: Dict[str, Any]) -> Dict:
        """
        Convert a single example to ChatML format.

        This method converts one example from the input format to ChatML format,
        applying field mappings, templates, and ensuring proper message structure.
        Designed to work with dataset.map() functionality.

        Args:
            example (Dict): The input example to convert
            config (Dict[str, Any]): Configuration for the conversion, including:
                - field_mappings (Dict): Maps input fields to ChatML roles with type and value:
                    - type: "column" or "template"
                    - value: column name or template string with {column} references

        Returns:
            Dict: The converted example in ChatML format with messages field, or empty dict if conversion fails
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

            # Check if we have valid user and assistant messages
            has_user = any(msg["role"] == "user" for msg in messages)
            has_assistant = any(msg["role"] == "assistant" for msg in messages)

            if has_user and has_assistant:
                return {"messages": messages}
            else:
                # Return empty dict for failed conversions to work with map()
                return {}

        except Exception as e:
            logger.warning(f"Failed to convert example: {e}")
            # Return empty dict for failed conversions to work with map()
            return {}

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
