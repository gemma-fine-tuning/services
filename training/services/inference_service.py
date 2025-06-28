import os
import logging
import torch
from .model_storage import storage_service, StorageStrategyFactory


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

    def __init__(self):
        self.export_bucket = os.environ.get(
            "GCS_EXPORT_BUCKET_NAME", "gemma-export-dev"
        )

    def run_inference(
        self, job_id_or_repo: str, prompt: str, storage_type: str = "gcs"
    ) -> str:
        """
        Fetch adapter artifacts, run generation, and return output text.
        Handles model loading, prompt formatting, and output postprocessing.

        Args:
            job_id_or_repo (str): Job ID for GCS or HF Hub repository ID
            prompt (str): Input text to generate a response for
            storage_type (str): Either "gcs" or "hfhub" to specify storage backend

        Returns:
            str: Generated text response from the model

        Raises:
            FileNotFoundError: If adapter artifacts or config are missing
            ValueError: If base model ID is not found in adapter config
        """
        # Use storage strategy to load model info
        strategy = StorageStrategyFactory.create_strategy(
            storage_type, storage_service=storage_service
        )
        artifact = strategy.load_model_info(job_id_or_repo)

        base_model_id = artifact.base_model_id
        adapter_path = (
            artifact.local_path if storage_type == "gcs" else artifact.remote_path
        )
        use_unsloth = artifact.use_unsloth

        if not base_model_id:
            raise ValueError("Base model ID not found in adapter config")

        try:
            if use_unsloth:
                output = self._run_inference_unsloth(
                    base_model_id, adapter_path, prompt, storage_type
                )
            else:
                output = self._run_inference_transformers(
                    base_model_id, adapter_path, prompt, storage_type
                )
        finally:
            # Cleanup using strategy
            strategy.cleanup(artifact)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logging.info(f"Inference completed for {job_id_or_repo}")
        return output

    def _run_inference_transformers(
        self,
        base_model_id: str,
        adapter_path: str,
        prompt: str,
        stream: bool = False,
    ) -> str:
        """
        Run inference using Transformers with trained adapter.
        Loads the base model and adapter, applies chat template, and generates output.

        Args:
            base_model_id (str): Base model ID or repository ID
            adapter_path (str): Path to the adapter artifacts
            prompt (str): Input text to generate a response for
            stream (bool): Whether to stream the output (default: False)

        Returns:
            str: Generated text response from the model
        """
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            pipeline,
            TextStreamer,
        )
        from peft import PeftModel

        # For GCS stored adapters, load base model then adapter from local path
        # For HF Hub adapters, load from repo_id
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id, torch_dtype=torch.float16, device_map="auto"
        )
        model = PeftModel.from_pretrained(
            model, adapter_path
        )  # adapter_path is repo_id
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)

        # Create generation pipeline
        pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer, device_map="auto"
        )

        # Prepare prompt
        chat_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        stop_tokens = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<end_of_turn>"),
        ]

        # Generate
        out = pipe(
            chat_prompt,
            max_new_tokens=256,
            do_sample=False,
            top_k=50,
            eos_token_id=stop_tokens,
            streamer=TextStreamer(tokenizer, skip_prompt=True) if stream else None,
        )
        text = out[0]["generated_text"][len(chat_prompt) :].strip()

        return text

    def _run_inference_unsloth(
        self,
        base_model_id: str,
        adapter_path: str,
        prompt: str,
        stream: bool = False,
    ) -> str:
        """
        Run inference using Unsloth with trained adapter.
        Loads the Unsloth model and adapter, applies chat template, and generates output.

        Args:
            base_model_id (str): Base model ID or repository ID
            adapter_path (str): Path to the adapter artifacts
            prompt (str): Input text to generate a response for
            stream (bool): Whether to stream the output (default: False)

        Returns:
            str: Generated text response from the model
        """
        # Dynamic imports
        import unsloth
        from transformers import TextStreamer
        from unsloth import FastModel
        from unsloth.chat_templates import get_chat_template

        # NOTE: Both unsloth and transformers support loading models directly using model_name=remote_path from hub
        # However, since we need to support GCS as well which requires local path, we need to do this with two steps

        # Load base model
        model, tokenizer = FastModel.from_pretrained(
            model_name=base_model_id,
            max_seq_length=2048,
            load_in_4bit=True,
        )
        # For GCS, adapter_path is local directory
        # For HF Hub, adapter_path is repo_id
        model.load_adapter(adapter_path)

        # Set inference mode (faster according to docs)
        FastModel.for_inference(model)

        # Prepare chat template
        tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        # don't tokenize here because we call the tokenizer again in generate
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        # Generate output
        outputs = model.generate(
            **tokenizer([text], return_tensors="pt").to("cuda"),
            max_new_tokens=256,
            temperature=1.0,
            top_p=0.95,
            top_k=64,
            streamer=TextStreamer(tokenizer, skip_prompt=True) if stream else None,
        )

        # Decode and only return the generated part
        # TODO: It seems like this does not remove the special tokens like <start_of_turn>model
        output_text = tokenizer.batch_decode(outputs)[0][len(text) :].strip()

        return output_text


# default service instance
inference_service = InferenceService()


def run_inference(job_id_or_repo: str, prompt: str, storage_type: str = "gcs") -> str:
    """
    Convenience function for running inference with different storage backends.
    Handles model loading and output generation for both GCS and HF Hub.

    Args:
        job_id_or_repo (str): Job ID for GCS or HF Hub repository ID
        prompt (str): Input text to generate a response for
        storage_type (str): Either "gcs" or "hfhub" to specify storage backend

    Returns:
        str: Generated text response from the model
    """
    return inference_service.run_inference(job_id_or_repo, prompt, storage_type)
