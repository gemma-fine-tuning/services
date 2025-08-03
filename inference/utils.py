import base64
import io
from typing import List, Tuple
from PIL import Image
import logging

logger = logging.getLogger(__name__)


def infer_storage_type_from_path(adapter_path: str) -> str:
    """
    Infer storage type from adapter path.

    Args:
        adapter_path: Path to adapter (local, GCS, or HF Hub)

    Returns:
        str: Either "local", "gcs", or "hfhub"
    """
    if adapter_path.startswith("gs://"):
        return "gcs"
    elif (
        "/" in adapter_path
        and not adapter_path.startswith("/")
        and not adapter_path.startswith("./")
    ):
        # Heuristic: if it contains "/" but doesn't start with "/" or "./" it's likely a HF Hub repo
        return "hfhub"
    else:
        # Local path (absolute or relative)
        return "local"


def infer_modality_from_messages(messages: List) -> str:
    """
    Infer the modality (text or vision) from the messages.
    This is a simple heuristic based on the content type.

    Args:
        messages: List of message conversations

    Returns:
        str: Either "text" or "vision"
    """
    for msg in messages:
        if isinstance(msg, list):
            for item in msg:
                if isinstance(item.get("content"), list):
                    for content in item["content"]:
                        if content.get("type") == "image":
                            return "vision"
    return "text"


def prepare_vision_inputs(
    processor, messages: List
) -> Tuple[List[List[Image.Image]], List[str]]:
    """
    Prepare vision inputs for batch processing.

    Expected structure of messages:
    [
        [
            {"role": "user", "content":
                [
                    {"type": "image", "image": "<base64_image>"},
                    {"type": "text", "text": "Describe the image."}
                ]
            },
            {"role": "assistant", "content": "Describe the image."}
        ],
        ...
    ]

    This does two things:
    1. Extracts images from the messages and decodes them from base64 to PIL so you can just pass them to the processor
    2. Formats the text prompts for the processor (this prevents the need to do this in the main inference function)

    Args:
        processor: The model processor for handling vision inputs
        messages: List of message conversations

    Returns:
        Tuple containing:
            - images: [[Image objects], [Image objects], ...]
            - texts: ["Formatted text prompt", "", ...]
            - len(images) == len(texts) == batch_size
    """
    # These two lists will hold the batch inputs
    images = []
    texts = []

    for msgs in messages:
        # Extract all image fields from this conversation
        raw_images = []
        for msg in msgs:
            if isinstance(msg.get("content"), list):
                for item in msg["content"]:
                    if item.get("type") == "image":
                        raw_images.append(item["image"])
        if not raw_images:
            raise ValueError("Image content not found in vision prompt")

        # Decode each image to PIL and collect
        pil_images: List[Image.Image] = []
        for image_content in raw_images:
            # Handle data URI header (e.g., "data:image/png;base64,...")
            if "," in image_content:
                image_content = image_content.split(",", 1)[1]
            # Pad base64
            padding = len(image_content) % 4
            if padding:
                image_content += "=" * (4 - padding)
            pil = Image.open(io.BytesIO(base64.b64decode(image_content)))
            pil_images.append(pil)

        # Add the list of PIL images for this conversation to the "batch"
        images.append(pil_images)

        # Prepare text content for this conversation and add to "batch"
        text = processor.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
        ).strip()
        texts.append(text)

    return images, texts


def get_model_device_config():
    """
    Get appropriate device configuration for model loading.

    Returns:
        dict: Model kwargs for device and dtype configuration
    """
    import torch

    return {
        "torch_dtype": torch.float16
        if torch.cuda.get_device_capability()[0] < 8
        else torch.bfloat16,
        "device_map": "auto",
    }


def get_stop_tokens(tokenizer):
    """
    Get appropriate stop tokens for generation.

    Args:
        tokenizer: Model tokenizer

    Returns:
        list: List of stop token IDs
    """
    stop_tokens = [tokenizer.eos_token_id]

    # Add Gemma-specific stop token if available
    try:
        gemma_stop = tokenizer.convert_tokens_to_ids("<end_of_turn>")
        if gemma_stop is not None:
            stop_tokens.append(gemma_stop)
    except (KeyError, ValueError):
        pass  # Token doesn't exist, skip

    return stop_tokens
