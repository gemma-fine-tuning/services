import logging
import time
import torch
from typing import List

from storage import StorageStrategyFactory
from utils import infer_modality_from_messages, infer_storage_type_from_path

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

    def run_inference(self, adapter_path: str, base_model_id: str, prompt: str) -> str:
        """
        Run inference with the given adapter and base model.

        Args:
            adapter_path (str): Path to adapter (local, GCS, or HF Hub repo ID)
            base_model_id (str): Base model identifier
            prompt (str): Input text to generate a response for

        Returns:
            str: Generated text response from the model
        """
        return self.run_batch_inference(
            adapter_path, base_model_id, [[{"role": "user", "content": prompt}]]
        )[0]

    def run_batch_inference(
        self, adapter_path: str, base_model_id: str, messages: List
    ) -> List[str]:
        """
        Run generation for a batch of messages with the given adapter and base model.

        Args:
            adapter_path (str): Path to adapter (local, GCS, or HF Hub repo ID)
            base_model_id (str): Base model identifier
            messages (list): List of input message conversations to generate responses for

        Returns:
            list[str]: List of generated text responses from the model
        """
        logger.info(f"Starting batch inference for adapter: {adapter_path}")
        start_time = time.time()

        # Determine storage type and handle model loading
        storage_type = infer_storage_type_from_path(adapter_path)

        if storage_type == "local":
            raise ValueError(
                "local inference is not yet supported, provide adapter path from gcs or hf hub"
            )
        else:
            # Need to use storage strategy for GCS or handle HF Hub
            strategy = StorageStrategyFactory.create_strategy(storage_type)
            if storage_type == "gcs":
                artifact = strategy.load_model_info(adapter_path)
                final_adapter_path = artifact.local_path
                use_unsloth = artifact.use_unsloth
            else:  # hfhub
                artifact = strategy.load_model_info(adapter_path)
                final_adapter_path = artifact.remote_path
                use_unsloth = artifact.use_unsloth

        # Determine modality and provider
        modality = infer_modality_from_messages(messages)
        provider_key = "unsloth" if use_unsloth else "huggingface"
        provider = self.providers[provider_key]

        try:
            outputs = provider.run_batch_inference(
                base_model_id, final_adapter_path, messages, modality
            )
        except Exception as e:
            logger.error(f"Batch inference failed with error: {str(e)}", exc_info=True)
            raise e
        finally:
            if artifact:
                strategy.cleanup(artifact)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.info(
            f"Batch inference for {adapter_path} completed in {time.time() - start_time:.2f} seconds."
        )
        return outputs


# Default orchestrator instance
inference_orchestrator = InferenceOrchestrator()


def run_inference(adapter_path: str, base_model_id: str, prompt: str) -> str:
    """
    Convenience function for running inference with different storage backends.
    **This is just a wrapper around the batch inference function!**

    Args:
        adapter_path (str): Path to adapter (local, GCS, or HF Hub repo ID)
        base_model_id (str): Base model identifier
        prompt (str): Input text to generate a response for

    Returns:
        str: Generated text response from the model
    """
    return inference_orchestrator.run_inference(adapter_path, base_model_id, prompt)


def run_batch_inference(
    adapter_path: str, base_model_id: str, messages: List
) -> List[str]:
    """
    Convenience function for running batch inference with different storage backends.
    Handles model loading and output generation for both GCS and HF Hub.

    Args:
        adapter_path (str): Path to adapter (local, GCS, or HF Hub repo ID)
        base_model_id (str): Base model identifier
        messages (list): List of input texts or formatted messages to generate responses for

    Returns:
        list[str]: List of generated text responses from the model
    """
    return inference_orchestrator.run_batch_inference(
        adapter_path, base_model_id, messages
    )
