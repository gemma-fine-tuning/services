import logging
import time
import torch
from typing import List, Dict, any

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

    def run_evaluation(
        self,
        adapter_path: str,
        base_model_id: str,
        dataset_id: str,
        task_type: str = None,
        metrics: List[str] = None,
        max_samples: int = None,
        num_sample_results: int = 3,
    ) -> Dict[str, any]:
        """
        Run evaluation of a fine-tuned model on a dataset.

        Args:
            adapter_path (str): Path to adapter (local, GCS, or HF Hub repo ID)
            base_model_id (str): Base model identifier
            dataset_id (str): Dataset ID to evaluate on (must have eval split)
            task_type (str, optional): Task type for predefined metric suite
            metrics (List[str], optional): Specific list of metrics to compute
            max_samples (int, optional): Maximum number of samples to evaluate
            num_sample_results (int, optional): Number of sample results to include

        Returns:
            Dict containing evaluation results and metadata
        """
        from evaluation import (
            EvaluationSuite,
            prepare_evaluation_data,
        )
        from storage import storage_service

        logger.info(
            f"Starting evaluation for adapter: {adapter_path} on dataset: {dataset_id}"
        )
        start_time = time.time()

        # Load evaluation dataset
        try:
            _, eval_dataset = storage_service.download_processed_dataset(dataset_id)
            if eval_dataset is None:
                raise ValueError(f"No evaluation split found for dataset {dataset_id}")
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_id}: {e}")
            raise

        # Limit samples if specified
        if max_samples and len(eval_dataset) > max_samples:
            eval_dataset = eval_dataset.select(range(max_samples))
            logger.info(f"Limited evaluation to {max_samples} samples")

        # Prepare evaluation data (messages and references) in a single pass
        eval_messages, references = prepare_evaluation_data(eval_dataset)

        if len(eval_messages) != len(references):
            raise ValueError(
                "Mismatch between number of evaluation messages and references"
            )

        # Generate predictions using batch inference
        try:
            predictions = self.run_batch_inference(
                adapter_path, base_model_id, eval_messages
            )
        except Exception as e:
            logger.error(f"Error during batch inference for evaluation: {e}")
            raise

        # Compute metrics
        evaluation_suite = EvaluationSuite()
        evaluation_results = evaluation_suite.compute_metrics(
            predictions, references, task_type, metrics, num_sample_results
        )

        eval_time = time.time() - start_time
        logger.info(f"Evaluation completed in {eval_time:.2f} seconds")
        logger.info(f"Computed metrics: {evaluation_results['metrics']}")

        return {
            "metrics": evaluation_results["metrics"],
            "samples": evaluation_results["samples"],
            "num_samples": len(eval_dataset),
            "dataset_id": dataset_id,
            "evaluation_time": eval_time,
        }


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


def run_evaluation(
    adapter_path: str,
    base_model_id: str,
    dataset_id: str,
    task_type: str = None,
    metrics: List[str] = None,
    max_samples: int = None,
    num_sample_results: int = 3,
) -> Dict[str, any]:
    """
    Convenience function for running evaluation with different storage backends.

    Args:
        adapter_path (str): Path to adapter (local, GCS, or HF Hub repo ID)
        base_model_id (str): Base model identifier
        dataset_id (str): Dataset ID to evaluate on (must have eval split)
        task_type (str, optional): Task type for predefined metric suite
        metrics (List[str], optional): Specific list of metrics to compute
        max_samples (int, optional): Maximum number of samples to evaluate
        num_sample_results (int, optional): Number of sample results to include

    Returns:
        Dict containing evaluation results and metadata
    """
    return inference_orchestrator.run_evaluation(
        adapter_path,
        base_model_id,
        dataset_id,
        task_type,
        metrics,
        max_samples,
        num_sample_results,
    )
