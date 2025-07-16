import logging
import re
from typing import Dict, Any, List, Optional
from datasets import DatasetDict
from PIL import Image
import base64

logger = logging.getLogger(__name__)


class FormatConverter:
    """
    A class that handles conversion of datasets to ChatML format.

    This class provides functionality to convert various dataset formats into the
    ChatML format, which is a standardized format for conversational AI training data.
    It supports conversion from multiple input formats and includes validation.

    For vision datasets, it supports multimodal ChatML format with image content.

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

    For vision datasets, content can be multimodal:
    ```json
    {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image", "image": PIL_Image_object}
                ]
            },
            {"role": "assistant", "content": [{"type": "text", "text": "Response"}]}
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

        Automatically detects image fields in field_mappings and creates multimodal ChatML format.

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
        """
        try:
            if not dataset:
                raise ValueError("Dataset is empty")

            # Check if we have any image fields in the configuration
            has_image_fields = self._has_image_fields(config.get("field_mappings", {}))

            if has_image_fields:
                conversion_method = self._convert_vision_example
            else:
                conversion_method = self._convert_single_example

            transformed_dataset = dataset.map(
                conversion_method,
                fn_kwargs={"config": config},
                batched=False,
                remove_columns=dataset[next(iter(dataset))].column_names,
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
            field_mappings (Dict): Maps field names to field mapping configs
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
        # Find field mapping for this role
        field_config = field_mappings.get(f"{role}_field")
        if not field_config:
            return ""

        # Skip image fields in text extraction
        if field_config.get("type") == "image":
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

    def _extract_vision_content(
        self,
        example: Dict,
        config: Dict,
        role: str,
    ) -> List[Dict[str, Any]]:
        """
        Extract and format multimodal content for a specific role from the example.

        This method extracts both text and image content from the input example
        and formats it for vision ChatML format. Images are only added to user messages.

        Args:
            example (Dict): The input example
            config (Dict): Configuration including field_mappings
            role (str): The role to extract content for ('user' or 'assistant')

        Returns:
            List[Dict[str, Any]]: List of content items with type and data

        Example:
            >>> example = {"question": "What is in this image?", "image": PIL_Image}
            >>> config = {
            ...     "field_mappings": {
            ...         "user_field": {"type": "column", "value": "question"},
            ...         "image_field": {"type": "image", "value": "image"}
            ...     }
            ... }
            >>> content = converter._extract_vision_content(example, config, "user")
            >>> [
            ...     {"type": "text", "text": "What is in this image?"},
            ...     {"type": "image", "image": PIL_Image_object}
            ... ]
        """
        content_items = []
        field_mappings = config.get("field_mappings", {})

        # Extract text content
        text_content = self._extract_content(example, field_mappings, role)
        if text_content:
            content_items.append({"type": "text", "text": text_content})

        # Extract image content - ONLY for user messages
        if role == "user":
            # Find all image field mappings
            for field_name, field_config in field_mappings.items():
                if field_config.get("type") == "image":
                    image_column = field_config.get("value")
                    if image_column and image_column in example:
                        image_data = example[image_column]
                        if image_data is not None:
                            processed_image = self._process_image_field(image_data)
                            if processed_image:
                                content_items.append(
                                    {"type": "image", "image": processed_image}
                                )

        return content_items

    def _process_image_field(self, image_data: Any) -> Optional[Any]:
        """
        Process image data into PIL format.

        Args:
            image_data: Image data in various formats

        Returns:
            Optional[Any]: Processed image or None if processing fails
        """
        try:
            # Already a PIL Image
            if Image and isinstance(image_data, Image.Image):
                return image_data.convert("RGB")

            # Dict with bytes (HuggingFace dataset format)
            if isinstance(image_data, dict) and "bytes" in image_data:
                import io

                image_bytes = image_data["bytes"]
                if Image:
                    image = Image.open(io.BytesIO(image_bytes))
                    return image.convert("RGB")

            # Base64 encoded string
            if isinstance(image_data, str):
                if image_data.startswith("data:image/"):
                    # Data URL format
                    header, data = image_data.split(",", 1)
                    image_bytes = base64.b64decode(data)
                else:
                    # Regular base64
                    image_bytes = base64.b64decode(image_data)

                image = Image.open(io.BytesIO(image_bytes))
                return image.convert("RGB")

            # File path (if it's a string path)
            if isinstance(image_data, str) and not self._is_base64_image(image_data):
                try:
                    image = Image.open(image_data)
                    return image.convert("RGB")
                except Exception:
                    pass

        except Exception as e:
            logger.error(f"Error processing image field: {str(e)}")

        return None

    def _convert_vision_example(self, example: Dict, config: Dict[str, Any]) -> Dict:
        """
        Convert a single example to vision ChatML format.

        This method converts one example from the input format to vision ChatML format,
        supporting multimodal content with both text and images.

        Args:
            example (Dict): The input example to convert
            config (Dict[str, Any]): Configuration for the conversion

        Returns:
            Dict: The converted example in vision ChatML format with messages field
        """
        try:
            field_mappings = config.get("field_mappings", {})
            messages = []

            # Handle system message (text only)
            if "system_field" in field_mappings:
                system_config = field_mappings["system_field"]
                if system_config["type"] == "column":
                    if system_config["value"] in example:
                        system_message = str(example[system_config["value"]])
                        if system_message:
                            messages.append(
                                {
                                    "role": "system",
                                    "content": [
                                        {"type": "text", "text": system_message}
                                    ],
                                }
                            )
                else:  # template
                    try:
                        template_vars = {
                            key: str(value) for key, value in example.items()
                        }
                        system_message = system_config["value"].format(**template_vars)
                        if system_message:
                            messages.append(
                                {
                                    "role": "system",
                                    "content": [
                                        {"type": "text", "text": system_message}
                                    ],
                                }
                            )
                    except (KeyError, ValueError) as e:
                        logger.warning(
                            f"System message template formatting failed: {e}"
                        )

            # Handle user message (potentially multimodal)
            user_content = self._extract_vision_content(example, config, "user")
            if user_content:
                messages.append({"role": "user", "content": user_content})

            # Handle assistant message (always text only)
            assistant_content = self._extract_content(example, config, "assistant")
            if assistant_content:
                messages.append({"role": "assistant", "content": assistant_content})

            # Check if we have valid user and assistant messages
            has_user = any(msg["role"] == "user" for msg in messages)
            has_assistant = any(msg["role"] == "assistant" for msg in messages)

            if has_user and has_assistant:
                return {"messages": messages}
            else:
                return {}

        except Exception as e:
            logger.warning(f"Failed to convert vision example: {e}")
            return {}

    def _is_base64_image(self, value: str) -> bool:
        """
        Check if a string is a base64 encoded image.

        Args:
            value: String to check

        Returns:
            bool: True if string appears to be base64 encoded image
        """
        try:
            # Check for data URL format
            if value.startswith("data:image/"):
                return True

            # Try to decode as base64
            if len(value) > 100:  # Reasonable minimum for image data
                decoded = base64.b64decode(value)
                return len(decoded) > 100

        except Exception:
            pass

        return False

    def _has_image_fields(self, field_mappings: Dict[str, Dict[str, Any]]) -> bool:
        """
        Check if field mappings contain any image fields.

        Args:
            field_mappings: Dictionary of field mappings

        Returns:
            bool: True if any field has type="image"
        """
        return any(
            field_config.get("type") == "image"
            for field_config in field_mappings.values()
        )
