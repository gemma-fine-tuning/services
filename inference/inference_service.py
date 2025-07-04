import logging
import torch
from model_storage import StorageStrategyFactory
from typing import List


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
        return self.run_batch_inference(job_id_or_repo_id, [prompt], storage_type)[0]

    def run_batch_inference(
        self, job_id_or_repo_id: str, prompts: List[str], storage_type: str = "gcs"
    ) -> List[str]:
        """
        Fetch adapter artifacts, run generation for a batch of prompts, and return output texts.
        Handles model loading, prompt formatting, and output postprocessing for a batch.

        Args:
            job_id_or_repo_id (str): Job ID for GCS or HF Hub repository ID
            prompts (list[str]): List of input texts to generate responses for
            storage_type (str): Either "gcs" or "hfhub" to specify storage backend

        Returns:
            list[str]: List of generated text responses from the model

        NOTE: Streaming with TextStreamer has been removed because we believe it will never be used
        for this use case and batch inference is much more suitable. However, if needed just add back
        by setting a TextStreamer object in model.generate() call
        """
        strategy = StorageStrategyFactory.create_strategy(storage_type)
        artifact = strategy.load_model_info(job_id_or_repo_id)

        base_model_id = artifact.base_model_id
        adapter_path = (
            artifact.local_path if storage_type == "gcs" else artifact.remote_path
        )
        use_unsloth = artifact.use_unsloth

        if not base_model_id:
            raise ValueError("Base model ID not found in adapter config")

        try:
            if use_unsloth:
                outputs = self._run_batch_inference_unsloth(
                    base_model_id, adapter_path, prompts
                )
            else:
                outputs = self._run_batch_inference_transformers(
                    base_model_id, adapter_path, prompts
                )
        except Exception as e:
            logging.error(f"Batch inference failed with error: {str(e)}", exc_info=True)
            raise e
        finally:
            strategy.cleanup(artifact)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logging.info(f"Batch inference completed for {job_id_or_repo_id}")
        return outputs

    def _run_batch_inference_transformers(
        self,
        base_model_id: str,
        adapter_path: str,
        prompts: List[str],
    ) -> List[str]:
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoTokenizer,
        )
        from transformers.utils.quantization_config import BitsAndBytesConfig
        # from peft import PeftModel

        model_kwargs = {
            "torch_dtype": torch.float16
            if torch.cuda.get_device_capability()[0] < 8
            else torch.bfloat16,
            "device_map": "auto",
            "quantization_config": BitsAndBytesConfig(load_in_4bit=True),
        }

        if base_model_id == "google/gemma-3-1b-it":
            model = AutoModelForCausalLM.from_pretrained(
                adapter_path,  # We can use adapter_path directly because it is either local or hf repo, no need base_model_id
                **model_kwargs,
            )
        else:
            model = AutoModelForImageTextToText.from_pretrained(
                adapter_path,
                **model_kwargs,
            )
        # NOTE: We no longer need to explicitly add adapter / peft, since the adapter_config.json should specify that
        # model = PeftModel.from_pretrained(model, adapter_path)
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)

        # NOTE: This does not tokenize but only changes each raw input into
        # chatML then tokenizer adds the system tokens based on the format requirements
        chat_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]
        tokenizer.pad_token = tokenizer.eos_token
        # Here this is actually tokenized
        batch_inputs = tokenizer(chat_prompts, return_tensors="pt", padding=True).to(
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

    def _run_batch_inference_unsloth(
        self,
        base_model_id: str,
        adapter_path: str,
        prompts: List[str],
    ) -> List[str]:
        import unsloth
        from unsloth import FastModel
        from unsloth.chat_templates import get_chat_template

        # Directly load the adapter with base model from hub to avoid issues with PEFT config
        model, tokenizer = FastModel.from_pretrained(
            model_name=adapter_path,
            max_seq_length=2048,
            load_in_4bit=True,  # For consistency we load both HF and unsloth in 4 bits!
        )
        FastModel.for_inference(model)
        tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")
        tokenizer.pad_token = tokenizer.eos_token
        # First we need to convert the raw input into chatML and then add system tokens
        chat_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]
        # Tokenize as a batch
        batch_inputs = tokenizer(chat_prompts, return_tensors="pt", padding=True).to(
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
    job_id_or_repo_id: str, prompts: List[str], storage_type: str = "gcs"
) -> List[str]:
    """
    Convenience function for running batch inference with different storage backends.
    Handles model loading and output generation for both GCS and HF Hub.

    Args:
        job_id_or_repo_id (str): Job ID for GCS or HF Hub repository ID
            - If storage_type is "gcs", this is the job ID
            - If storage_type is "hfhub", this is the Hugging Face repository ID
            - The frontend would determine this automatically
        prompts (list[str]): List of input texts to generate responses for
        storage_type (str): Either "gcs" or "hfhub" to specify storage backend

    Returns:
        list[str]: List of generated text responses from the model
    """
    return inference_service.run_batch_inference(
        job_id_or_repo_id, prompts, storage_type
    )
