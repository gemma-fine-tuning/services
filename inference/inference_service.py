import logging
import torch
import time
from storage import StorageStrategyFactory
from typing import List, Tuple
from PIL import Image
import base64
import io


class InferenceService:
    """
    Service for running inference with trained adapters from multiple storage backends.

    Supports:
    - Google Cloud Storage (GCS) - original backend
    - Hugging Face Hub - for models uploaded via HF Hub
    - Both standard Transformers and Unsloth models

    The service automatically detects the adapter type (Unsloth vs standard) and loads
    the appropriate inference pipeline. For HF Hub, it attempts to read configuration
    files to determine the base model and training framework used.
    """

    def run_inference(
        self, job_id_or_repo_id: str, prompt: str, storage_type: str = "gcs"
    ) -> str:
        """
        Fetch adapter artifacts, run generation, and return output text.
        Handles model loading, prompt formatting, and output postprocessing.

        Args:
            job_id_or_repo_id (str): Job ID for GCS or HF Hub repository ID
            prompt (str): Input text to generate a response for
            storage_type (str): Either "gcs" or "hfhub" to specify storage backend

        Returns:
            str: Generated text response from the model

        Raises:
            FileNotFoundError: If adapter artifacts or config are missing
            ValueError: If base model ID is not found in adapter config
        """
        return self.run_batch_inference(
            job_id_or_repo_id, [[{"role": "user", "content": prompt}]], storage_type
        )[0]

    def run_batch_inference(
        self, job_id_or_repo_id: str, messages: List, storage_type: str = "gcs"
    ) -> List[str]:
        """
        Fetch adapter artifacts, run generation for a batch of messages, and return output texts.
        Handles model loading, prompt formatting, and output postprocessing for a batch.

        Args:
            job_id_or_repo_id (str): Job ID for GCS or HF Hub repository ID
            messages (list[str]): List of input texts to generate responses for
            storage_type (str): Either "gcs" or "hfhub" to specify storage backend

        Returns:
            list[str]: List of generated text responses from the model

        NOTE: Streaming with TextStreamer has been removed because we believe it will never be used
        for this use case and batch inference is much more suitable. However, if needed just add back
        by setting a TextStreamer object in model.generate() call
        """
        logging.info(f"Starting batch inference for {job_id_or_repo_id}...")
        start_time = time.time()

        strategy = StorageStrategyFactory.create_strategy(storage_type)
        artifact = strategy.load_model_info(job_id_or_repo_id)

        base_model_id = artifact.base_model_id
        adapter_path = (
            artifact.local_path if storage_type == "gcs" else artifact.remote_path
        )
        use_unsloth = artifact.use_unsloth

        if not base_model_id:
            raise ValueError("Base model ID not found in adapter config")

        # TODO: This is not compatible with hf hub need changes from training side!
        # is_vision_model = artifact.metadata.get("modality", "text") == "vision"
        is_vision_model = self._infer_modality_from_messages(messages) == "vision"

        try:
            if use_unsloth:
                if is_vision_model:
                    outputs = self._run_batch_inference_unsloth_vision(
                        base_model_id, adapter_path, messages
                    )
                else:
                    outputs = self._run_batch_inference_unsloth_text(
                        base_model_id, adapter_path, messages
                    )
            else:
                if is_vision_model:
                    outputs = self._run_batch_inference_transformers_vision(
                        base_model_id, adapter_path, messages
                    )
                else:
                    outputs = self._run_batch_inference_transformers_text(
                        base_model_id, adapter_path, messages
                    )
        except Exception as e:
            logging.error(f"Batch inference failed with error: {str(e)}", exc_info=True)
            raise e
        finally:
            strategy.cleanup(artifact)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logging.info(
            f"Batch inference for {job_id_or_repo_id} completed in {time.time() - start_time:.2f} seconds."
        )
        return outputs

    def _prepare_vision_inputs(
        self, processor, messages: List
    ) -> Tuple[List[List[Image.Image]], List[str]]:
        """
        Expected structure of messages:
        [
            [
                {"role": "user", "content":
                    [
                        {"type": "image", "image": "<base64_image>"},
                        {"type": "text, "text": "Describe the image."}
                    ]
                },
                {"role": "assistant", "content": "Describe the image."}
            ],
            ...
        ]

        This does two things:
        1. Extracts images from the messages and decodes them from base64 to PIL so you can just pass them to the processor
        2. Formats the text prompts for the processor (this prevents the need to do this in the main inference function)

        Expected output:
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

    def _run_batch_inference_transformers_text(
        self,
        base_model_id: str,
        adapter_path: str,
        messages: List,
    ) -> List[str]:
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoTokenizer,
        )
        from transformers.utils.quantization_config import BitsAndBytesConfig

        # TODO: Do not set quantization config by default let the user select?
        # However doing this at inference time should work for models trained with and without quantization
        model_kwargs = {
            "torch_dtype": torch.float16
            if torch.cuda.get_device_capability()[0] < 8
            else torch.bfloat16,
            "device_map": "auto",
            "quantization_config": BitsAndBytesConfig(load_in_4bit=True),
        }

        if base_model_id in ["google/gemma-3-1b-it", "google/gemma-3-1b-pt"]:
            model = AutoModelForCausalLM.from_pretrained(
                adapter_path,  # We can use adapter_path directly because it is either local or hf repo, no need base_model_id
                **model_kwargs,
            )
        else:
            # AutoModelForImageTextToText is still used for text-only models!
            # The only determining factor is whether the base_model_id is a vision model or not
            model = AutoModelForImageTextToText.from_pretrained(
                adapter_path,
                **model_kwargs,
            )
        # NOTE: We no longer need to explicitly add adapter / peft, since the adapter_config.json should specify that
        # model = PeftModel.from_pretrained(model, adapter_path)
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)

        chat_messages = [
            tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
            )
            for message in messages
        ]
        tokenizer.pad_token = tokenizer.eos_token
        batch_inputs = tokenizer(chat_messages, return_tensors="pt", padding=True).to(
            model.device
        )
        stop_tokens = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<end_of_turn>"),
        ]
        # Get the length so we can slice the output
        input_length = batch_inputs.input_ids.shape[1]
        generated_ids = model.generate(
            **batch_inputs,
            max_new_tokens=512,
            do_sample=False,
            top_k=50,
            eos_token_id=stop_tokens,
        )
        # You must slice the generated_ids instead of slicing the decoded output
        # This, together with skipping special tokens, remove all the "model" and "user" tokens
        decoded = tokenizer.batch_decode(
            generated_ids[:, input_length:], skip_special_tokens=True
        )
        return decoded

    def _run_batch_inference_transformers_vision(
        self,
        base_model_id: str,
        adapter_path: str,
        messages: List,
    ) -> List[str]:
        from transformers import (
            AutoModelForImageTextToText,
            AutoProcessor,
        )
        from transformers.utils.quantization_config import BitsAndBytesConfig

        model_kwargs = {
            "torch_dtype": torch.float16
            if torch.cuda.get_device_capability()[0] < 8
            else torch.bfloat16,
            "device_map": "auto",
            "quantization_config": BitsAndBytesConfig(load_in_4bit=True),
        }

        model = AutoModelForImageTextToText.from_pretrained(
            adapter_path,
            **model_kwargs,
        )
        processor = AutoProcessor.from_pretrained(base_model_id)

        images, texts = self._prepare_vision_inputs(processor, messages)

        inputs = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=128)
        decoded = processor.batch_decode(
            generated_ids[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        return decoded

    def _run_batch_inference_unsloth_text(
        self,
        base_model_id: str,
        adapter_path: str,
        messages: List,
    ) -> List[str]:
        from unsloth import FastModel
        from unsloth.chat_templates import get_chat_template

        model, tokenizer = FastModel.from_pretrained(
            model_name=adapter_path,
            max_seq_length=2048,
            load_in_4bit=True,
        )
        FastModel.for_inference(model)
        tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")
        tokenizer.pad_token = tokenizer.eos_token

        chat_messages = [
            tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in messages
        ]
        batch_inputs = tokenizer(chat_messages, return_tensors="pt", padding=True).to(
            "cuda"
        )
        generated_ids = model.generate(
            **batch_inputs,
            max_new_tokens=256,
            temperature=1.0,
            top_p=0.95,
            top_k=64,
        )
        input_length = batch_inputs.input_ids.shape[1]
        decoded = tokenizer.batch_decode(
            generated_ids[:, input_length:], skip_special_tokens=True
        )
        return decoded

    def _run_batch_inference_unsloth_vision(
        self,
        base_model_id: str,
        adapter_path: str,
        messages: List,
    ) -> List[str]:
        from unsloth import FastVisionModel
        from unsloth.chat_templates import get_chat_template

        model, processor = FastVisionModel.from_pretrained(
            model_name=adapter_path,
            load_in_4bit=True,
        )
        FastVisionModel.for_inference(model)
        processor = get_chat_template(processor, chat_template="gemma-3")

        images, texts = self._prepare_vision_inputs(processor, messages)

        inputs = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
        ).to("cuda")

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            use_cache=True,
            temperature=1.0,
            top_p=0.95,
            top_k=64,
        )
        decoded = processor.batch_decode(
            generated_ids[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        return decoded

    def _infer_modality_from_messages(self, messages: List) -> str:
        """
        Infer the modality (text or vision) from the messages.
        This is a simple heuristic based on the content type.
        """
        for msg in messages:
            if isinstance(msg, list):
                for item in msg:
                    if isinstance(item["content"], list):
                        for content in item["content"]:
                            if content.get("type") == "image":
                                return "vision"
        return "text"


# default service instance
inference_service = InferenceService()


def run_inference(
    job_id_or_repo_id: str, prompt: str, storage_type: str = "gcs"
) -> str:
    """
    Convenience function for running inference with different storage backends.
    **This is just a wrapper around the batch inference function!**

    Args:
        job_id_or_repo_id (str): Job ID for GCS or HF Hub repository ID
        prompt (str): Input text to generate a response for
        storage_type (str): Either "gcs" or "hfhub" to specify storage backend

    Returns:
        str: Generated text response from the model

    """
    return inference_service.run_inference(job_id_or_repo_id, prompt, storage_type)


def run_batch_inference(
    job_id_or_repo_id: str, messages: List, storage_type: str = "gcs"
) -> List[str]:
    """
    Convenience function for running batch inference with different storage backends.
    Handles model loading and output generation for both GCS and HF Hub.

    Args:
        job_id_or_repo_id (str): Job ID for GCS or HF Hub repository ID
            - If storage_type is "gcs", this is the job ID
            - If storage_type is "hfhub", this is the Hugging Face repository ID
            - The frontend would determine this automatically
        messages (list): List of input texts or formatted messages to generate responses for
        storage_type (str): Either "gcs" or "hfhub" to specify storage backend

    Returns:
        list[str]: List of generated text responses from the model
    """
    return inference_service.run_batch_inference(
        job_id_or_repo_id, messages, storage_type
    )
