import logging
import re
from typing import Dict, Any, List, Optional
from datasets import DatasetDict
import base64
import io
from PIL import Image

logger = logging.getLogger(__name__)


class FormatConverter:
    """
    A class that handles conversion of datasets to ChatML format.

    This class provides functionality to convert various dataset formats into the
    ChatML format, which is a standardized format for conversational AI training data.
    It supports conversion from multiple input formats and includes validation.

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

            # Validate field mappings refer to real columns
            field_mappings = config.get("field_mappings", {})
            first_split = dataset[next(iter(dataset))]
            available_columns = set(first_split.column_names)

            for field_key, field_config in field_mappings.items():
                if field_config.get("type") in ("column", "image"):
                    column_name = field_config.get("value")
                    if column_name not in available_columns:
                        raise ValueError(
                            f"Field mapping '{field_key}' refers to column '{column_name}' "
                            f"which does not exist in dataset. Available columns: {sorted(available_columns)}"
                        )

            # Check if we have any image fields in the configuration
            has_image_fields = self._has_image_fields(field_mappings)
            logger.info(f"Has image fields: {has_image_fields}")

            transformed_dataset = dataset.map(
                self._convert_single_example,
                fn_kwargs={
                    "field_mappings": field_mappings,
                    "is_multimodal": has_image_fields,
                },
                batched=False,
                remove_columns=dataset[next(iter(dataset))].column_names,
            )
            # Filter out failed conversions (empty dictionaries)
            transformed_dataset = transformed_dataset.filter(
                lambda x: "messages" in x and x["messages"]
            )

            logger.info(f"Converted dataset splits: {list(transformed_dataset.keys())}")
            for split_name, split_data in transformed_dataset.items():
                logger.info(
                    f"Converted split {split_name} has {len(split_data)} examples"
                )

            return transformed_dataset

        except Exception as e:
            logger.error(f"Error converting to ChatML format: {str(e)}")
            raise

    def _convert_single_example(
        self,
        example: Dict,
        field_mappings: Dict[str, Dict[str, Any]],
        is_multimodal: bool = False,
    ) -> Dict:
        """
        Convert a single example to ChatML format.

        This method converts one example from the input format to ChatML format,
        applying field mappings, templates, and ensuring proper message structure.
        Designed to work with dataset.map() functionality.

        Args:
            example (Dict): The input example to convert
            field_mappings (Dict): Maps input fields to ChatML roles with type and value:
                - type: "column" or "template"
                - value: column name or template string with {column} references

        Returns:
            Dict: The converted example in ChatML format with messages field, or empty dict if conversion fails
        """
        # Simplified single-example conversion using helper methods
        try:
            messages: List[Dict[str, Any]] = []

            # System message
            sys_msg = self._create_system_message(example, field_mappings)
            if sys_msg:
                messages.append(sys_msg)

            # User message (text only)
            user_msg = self._create_user_message(
                example, field_mappings, is_multimodal=is_multimodal
            )
            if user_msg:
                messages.append(user_msg)

            # Assistant message
            assistant_msg = self._create_assistant_message(example, field_mappings)
            if assistant_msg:
                messages.append(assistant_msg)

            # Validate
            if self._validate_messages(messages):
                return {"messages": messages}
            return {}
        except Exception as e:
            logger.warning(f"Failed to convert single example: {e}")
            return {}

    def _extract_text_content(
        self,
        example: Dict,
        field_mappings: Dict,
        role: str,
    ) -> str:
        """
        Extract and format text content for a specific role from the example.

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
            >>> content = converter._extract_text_content(example, field_mappings, "user")
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

    def _extract_multimodal_content(
        self,
        example: Dict,
        field_mappings: Dict,
        role: str,
    ) -> List[Dict[str, Any]]:
        """
        Extract and format multimodal content (text + images) for a specific role.

        This method extracts both text and image content from the input example
        and formats it for vision ChatML format. Images are only added to user messages.

        Args:
            example (Dict): The input example
            field_mappings (Dict): Maps roles to field mapping configs
            role (str): The role to extract content for ('user' or 'assistant')

        Returns:
            List[Dict[str, Any]]: List of content items with type and data

        Example:
            >>> example = {"question": "What is in this image?", "image": PIL_Image}
            >>> field_mappings = {
            ...     "user_field": {"type": "column", "value": "question"},
            ...     "image_field": {"type": "image", "value": "image"}
            ... }
            >>> content = converter._extract_multimodal_content(example, field_mappings, "user")
            >>> [
            ...     {"type": "text", "text": "What is in this image?"},
            ...     {"type": "image", "image": PIL_Image_object}
            ... ]

        NOTE: After serialisation (parquet) the `PIL_Image_object` becomes:

            "image": {
                "bytes": "base64_encoded_image_data",
                "path": null
            }

        When bytes cannot be used (e.g. API response) we convert to base64 string.
        Otherwise, we keep the PIL Image object (e.g. for training).
        """
        content_items = []

        # Extract text content using the existing method
        text_content = self._extract_text_content(example, field_mappings, role)
        if text_content:
            content_items.append({"type": "text", "text": text_content})

        # Extract image content - ONLY for user messages
        if role == "user":
            # Find all image field mappings
            for _, field_config in field_mappings.items():
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
                            else:
                                logger.warning(
                                    f"Failed to process image field '{image_column}' for {role}"
                                )

        logger.debug(f"Final content items for {role}: {len(content_items)} items")
        return content_items

    def _process_image_field(self, image_data: Any) -> Optional[Any]:
        """
        Process image data and keep as PIL Image for efficient storage.

        This approach is optimized for the overall system:
        1. PIL Images are efficiently stored in parquet format by datasets library
        2. Training services can directly use PIL Images from parquet
        3. Only convert to base64 when needed for API responses or inference

        Args:
            image_data: Image data in various formats

        Returns:
            Optional[PIL.Image]: PIL Image object or None if processing fails
        """
        try:
            # Already a PIL Image - return as-is
            if Image and isinstance(image_data, Image.Image):
                logger.info("Image is already a PIL Image, converting to RGB")
                return image_data.convert("RGB")

            # Dict with bytes (HuggingFace dataset format)
            elif isinstance(image_data, dict) and "bytes" in image_data:
                logger.info("Image is in HuggingFace format with bytes field")
                image_bytes = image_data["bytes"]
                if Image:
                    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    return pil_image

            # Base64 encoded string
            elif isinstance(image_data, str):
                if image_data.startswith("data:image/"):
                    # Data URL format
                    logger.info("Image is a data URL")
                    header, data = image_data.split(",", 1)
                    image_bytes = base64.b64decode(data)
                    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    return pil_image
                elif self._is_base64_image(image_data):
                    # Regular base64
                    logger.info("Image is regular base64")
                    image_bytes = base64.b64decode(image_data)
                    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    return pil_image
                else:
                    # Might be a file path
                    try:
                        pil_image = Image.open(image_data).convert("RGB")
                        return pil_image
                    except Exception as path_error:
                        logger.warning(f"Failed to open image from path: {path_error}")
                        return None

            # Raw bytes
            elif isinstance(image_data, (bytes, bytearray)):
                pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")
                return pil_image

            else:
                logger.error(f"Unsupported image data type: {type(image_data)}")
                return None

        except Exception as e:
            logger.error(f"Error processing image field: {str(e)}")
            logger.error(f"Image data type: {type(image_data)}")
            if hasattr(image_data, "__len__"):
                logger.error(f"Image data length: {len(image_data)}")
            return None

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

    def _create_system_message(
        self, example: Dict, field_mappings: Dict
    ) -> Optional[Dict]:
        """
        Create a system message from the example data.

        Args:
            example (Dict): The input example
            field_mappings (Dict): Field mappings configuration

        Returns:
            Optional[Dict]: System message dict or None if no system field
        """
        if "system_field" not in field_mappings:
            return None

        system_config = field_mappings["system_field"]
        system_message = ""

        if system_config["type"] == "column":
            if system_config["value"] in example:
                system_message = str(example[system_config["value"]])
        else:  # template
            try:
                template_vars = {key: str(value) for key, value in example.items()}
                system_message = system_config["value"].format(**template_vars)
            except (KeyError, ValueError) as e:
                logger.warning(f"System message template formatting failed: {e}")
                return None

        if system_message:
            return {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            }
        return None

    def _create_user_message(
        self, example: Dict, field_mappings: Dict, is_multimodal: bool = False
    ) -> Optional[Dict]:
        """
        Create a user message from the example data.

        Args:
            example (Dict): The input example
            field_mappings (Dict): Field mappings configuration
            is_multimodal (bool): Whether to include image content

        Returns:
            Optional[Dict]: User message dict or None if no content
        """
        user_content = (
            self._extract_multimodal_content(example, field_mappings, "user")
            if is_multimodal
            else [
                {
                    "type": "text",
                    "text": self._extract_text_content(example, field_mappings, "user"),
                }
            ]
        )

        # Filter out empty text content
        if is_multimodal:
            user_content = [
                item for item in user_content if item.get("text") or item.get("image")
            ]
        else:
            user_content = [item for item in user_content if item.get("text")]

        if user_content:
            return {"role": "user", "content": user_content}

        return None

    def _create_assistant_message(
        self, example: Dict, field_mappings: Dict
    ) -> Optional[Dict]:
        """
        Create an assistant message from the example data.

        Args:
            example (Dict): The input example
            field_mappings (Dict): Field mappings configuration

        Returns:
            Optional[Dict]: Assistant message dict or None if no content
        """
        assistant_content = self._extract_text_content(
            example, field_mappings, "assistant"
        )
        if assistant_content:
            return {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_content}],
            }
        return None

    def _validate_messages(self, messages: List[Dict]) -> bool:
        """
        Validate that we have the required messages for a valid conversation.

        Args:
            messages (List[Dict]): List of message dictionaries

        Returns:
            bool: True if messages are valid for training
        """
        has_user = any(msg["role"] == "user" for msg in messages)
        has_assistant = any(msg["role"] == "assistant" for msg in messages)
        return has_user and has_assistant
