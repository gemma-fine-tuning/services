import logging
import time
import torch
from typing import List

from storage import StorageStrategyFactory
from utils import infer_modality_from_messages

logger = logging.getLogger(__name__)


class InferenceOrchestrator:
    """
    Main orchestrator for inference operations.

    This class coordinates between storage strategies and inference providers
    to handle the complete inference workflow. It maintains the same public
    API as the original InferenceService for backward compatibility.
    """

    def __init__(self):
        self.providers = {}
        self._register_providers()

    def _register_providers(self):
        """Lazy import and register inference providers"""
        # Import here to avoid circular imports
        from providers import HuggingFaceInferenceProvider, UnslothInferenceProvider

        self.providers = {
            "huggingface": HuggingFaceInferenceProvider(),
            "unsloth": UnslothInferenceProvider(),
        }

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
            messages (list): List of input message conversations to generate responses for
            storage_type (str): Either "gcs" or "hfhub" to specify storage backend

        Returns:
            list[str]: List of generated text responses from the model

        NOTE: Streaming with TextStreamer has been removed because we believe it will never be used
        for this use case and batch inference is much more suitable. However, if needed just add back
        by setting a TextStreamer object in model.generate() call
        """
        logger.info(f"Starting batch inference for {job_id_or_repo_id}...")
        start_time = time.time()

        # Load model artifact info
        strategy = StorageStrategyFactory.create_strategy(storage_type)
        artifact = strategy.load_model_info(job_id_or_repo_id)

        base_model_id = artifact.base_model_id
        adapter_path = (
            artifact.local_path if storage_type == "gcs" else artifact.remote_path
        )
        use_unsloth = artifact.use_unsloth

        if not base_model_id:
            raise ValueError("Base model ID not found in adapter config")

        # Determine modality and provider
        modality = infer_modality_from_messages(messages)
        provider_key = "unsloth" if use_unsloth else "huggingface"
        provider = self.providers[provider_key]

        try:
            outputs = provider.run_batch_inference(
                base_model_id, adapter_path, messages, modality
            )
        except Exception as e:
            logger.error(f"Batch inference failed with error: {str(e)}", exc_info=True)
            raise e
        finally:
            strategy.cleanup(artifact)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.info(
            f"Batch inference for {job_id_or_repo_id} completed in {time.time() - start_time:.2f} seconds."
        )
        return outputs


# Default orchestrator instance
inference_orchestrator = InferenceOrchestrator()


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
    return inference_orchestrator.run_inference(job_id_or_repo_id, prompt, storage_type)


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
    return inference_orchestrator.run_batch_inference(
        job_id_or_repo_id, messages, storage_type
    )
