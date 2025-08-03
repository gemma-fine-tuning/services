import logging
import evaluate
from transformers import EvalPrediction
import numpy as np
from typing import List, Callable
from storage import StorageStrategyFactory, CloudStoredModelMetadata
from job_manager import JobTracker
from schema import ExportConfig
import os
import shutil


def run_evaluation(trainer):
    """
    Run evaluation on the trainer, log and return metrics.
    """
    eval_results = trainer.evaluate()
    logging.info(f"Evaluation results: {eval_results}")
    return eval_results


def _prepare_model_for_export(
    model, tokenizer, export_config: ExportConfig, use_unsloth: bool, temp_dir: str
):
    """
    Prepare model for export based on the export configuration.
    Uses provider-specific saving methods (Unsloth vs HuggingFace).
    We currently don't support ONNX yet might do that soon!

    NOTE: This will save everything locally first, then upload to either gcs or hub later.
    We can choose to directly call push_to_hub or push_to_hub_merged as well they are functionally identical.

    After exporting model, this guide tells you how to match the quantisation when loading e.g. from transformer.js
    https://huggingface.co/docs/transformers.js/en/guides/dtypes

    Args:
        model: The trained model
        tokenizer: The tokenizer
        export_config: Export configuration
        use_unsloth: Whether using Unsloth provider
        temp_dir: Temporary directory for saving

    Returns:
        Tuple of (processed_model, tokenizer, actual_temp_dir)
    """
    logging.info(
        f"Preparing model for export with format: {export_config.format}, provider: {'unsloth' if use_unsloth else 'huggingface'}"
    )

    # NOTE: Backward compatible with default adapter export
    # For unsloth this also works: model.save_pretrained_merged("model", tokenizer, save_method = "lora",)
    if export_config.format == "adapter":
        # Save model locally first
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(temp_dir)
        else:
            # For trainers
            model.save_model(temp_dir)
        tokenizer.save_pretrained(temp_dir)

    # Almost all other formats require different handling between unsloth and hf
    if use_unsloth:
        return _prepare_unsloth_export(model, tokenizer, export_config, temp_dir)
    else:
        return _prepare_huggingface_export(model, tokenizer, export_config, temp_dir)


def _prepare_unsloth_export(
    model, tokenizer, export_config: ExportConfig, temp_dir: str
):
    """Handle Unsloth-specific export methods"""
    if export_config.format == "merged":
        # Use Unsloth's merged saving methods
        # https://docs.unsloth.ai/basics/running-and-saving-models/saving-to-vllm
        logging.info(
            f"Saving Unsloth merged model with quantization: {export_config.quantization}"
        )

        if export_config.quantization == "fp16":
            # This format (merged 16bit) works for vLLM
            model.save_pretrained_merged(
                temp_dir, tokenizer, save_method="merged_16bit"
            )
        elif export_config.quantization == "q4":
            # fp4 can be easily loaded with HF transformers libraries
            model.save_pretrained_merged(temp_dir, tokenizer, save_method="merged_4bit")
        else:
            logging.error(
                f"Unsupported quantization for Unsloth merged export: {export_config.quantization}"
            )

        return model, tokenizer, temp_dir

    elif export_config.format == "gguf":
        # Use Unsloth's native GGUF export
        # https://docs.unsloth.ai/basics/running-and-saving-models/saving-to-gguf
        # NOTE: This does not yet work will be fixed sooon by unsloth team
        logging.info(
            f"Saving Unsloth GGUF with quantization: {export_config.quantization}"
        )

        try:
            model.save_pretrained_gguf(
                temp_dir, tokenizer, quantization_method=export_config.quantization
            )
            return model, tokenizer, temp_dir
        except Exception as e:
            logging.error(f"Error during Unsloth GGUF export: {e}")
            raise

    else:
        raise ValueError(
            f"Unsupported export format for Unsloth: {export_config.format}"
        )


def _prepare_huggingface_export(
    model, tokenizer, export_config: ExportConfig, temp_dir: str
):
    """
    Handle HuggingFace-specific export methods
    """
    if export_config.format == "merged":
        # Merge adapter weights into base model using HuggingFace/PEFT methods
        logging.info(
            f"Merging HuggingFace adapter with quantization: {export_config.quantization}"
        )

        try:
            if hasattr(model, "merge_and_unload"):
                # only for PEFT models
                merged_model = model.merge_and_unload()
            else:
                # Already a merged model or full fine-tuning
                merged_model = model

            # Apply quantization if specified
            if export_config.quantization == "fp16":
                # This is equivalent as: model.to(dtype=torch.float16)
                # NOTE: This will not work for 4bit or 8bit because pytorch only works with floats
                merged_model = merged_model.half()

            # save_pretrained automatically handles 8bit and 4bit using bitsandbytes
            # it also uploads the quantisation configuration so it is loaded properly
            merged_model.save_pretrained(temp_dir, safe_serialization=True)
            tokenizer.save_pretrained(temp_dir)
            return merged_model, tokenizer, temp_dir

        except Exception as e:
            logging.error(f"Error during HuggingFace model merging: {e}")
            raise

    elif export_config.format == "gguf":
        # HuggingFace GGUF export requires external tools
        logging.error(
            "HuggingFace GGUF export not implemented - requires llama.cpp conversion"
        )
        # TODO: This would require:
        # 1. First merge the model and save as HF format
        # 2. Use llama.cpp convert-hf-to-gguf.py script
        # 3. This is complex and might be better as a separate service
        raise NotImplementedError(
            "HuggingFace GGUF export requires llama.cpp conversion tools"
        )

    else:
        raise ValueError(
            f"Unsupported export format for HuggingFace: {export_config.format}"
        )


# TODO: Update the actual merging and etc method here since you have the model AND tokenizer and config
def save_and_track(
    export_config: ExportConfig,
    model,
    tokenizer,
    job_id: str,
    base_model_id: str,
    use_unsloth: bool,
    job_tracker: JobTracker,
    metrics: dict | None = None,
):
    """
    Save the model using storage strategy, cleanup, and mark job as completed.
    Saves the model in the format specified by export_config (adapter, merged, or gguf).
    The inference service will handle both adapter and merged models automatically.
    """
    # Determine temp directory based on export format
    temp_dir = f"/tmp/{job_id}_{export_config.format}"

    try:
        # Prepare model for export based on configuration
        processed_model, processed_tokenizer, actual_temp_dir = (
            _prepare_model_for_export(
                model, tokenizer, export_config, use_unsloth, temp_dir
            )
        )

        # Use export_config.destination to determine storage strategy
        storage_strategy = StorageStrategyFactory.create_strategy(
            export_config.destination
        )
        metadata = CloudStoredModelMetadata(
            job_id=job_id,
            base_model_id=base_model_id,
            gcs_prefix="",  # Will be set by storage service based on format
            use_unsloth=use_unsloth,
            local_dir=actual_temp_dir,
            hf_repo_id=export_config.hf_repo_id,
            export_format=export_config.format,
            quantization=export_config.quantization,
        )

        # The model has already been saved locally by our export logic above
        artifact = storage_strategy.save_model(
            actual_temp_dir,  # Local directory containing saved files
            metadata,
        )
        storage_strategy.cleanup(artifact)
        job_tracker.completed(artifact.remote_path, artifact.base_model_id, metrics)
        return artifact

    except Exception as e:
        logging.error(f"Error during model export: {e}")
        # Clean up temp directory on error
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def create_compute_metrics(
    selected_metrics: List[str] = None, use_batched_eval: bool = False
) -> Callable:
    """
    Create a compute_metrics function with user-selected metrics using the evaluate library.
    Supports: accuracy, perplexity, loss
    Supports both regular and batch evaluation modes.
    NOTE: This does not handle more complex evaluation like task specific evaluators.

    Args:
        selected_metrics: List of metric names to compute (e.g., ["accuracy", "perplexity"])
        use_batched_eval: If True, enables batch evaluation mode for metrics computation.

    Returns:
        A compute_metrics function for use with HuggingFace Trainer
    """
    if selected_metrics is None:
        selected_metrics = ["accuracy", "perplexity"]

    # Load accuracy metric from evaluate library
    # TODO: load other metrics similarly
    accuracy_metric = None
    if "accuracy" in selected_metrics:
        try:
            accuracy_metric = evaluate.load("accuracy")
        except Exception as e:
            logging.warning(f"Could not load accuracy metric: {e}")

    def compute_metrics(eval_pred: EvalPrediction) -> dict:
        """
        Compute evaluation metrics using the evaluate library (regular mode).
        """
        logits, labels = eval_pred
        if (
            isinstance(logits, tuple)
            and hasattr(logits[0], "shape")
            and logits[0].shape == (0,)
        ):
            return {}
        predictions = np.argmax(logits, axis=-1)
        results = {}

        loss = (
            eval_pred.metrics.get("eval_loss", 0)
            if hasattr(eval_pred, "metrics")
            else 0
        )

        results["loss"] = loss

        # Compute accuracy using evaluate library
        if "accuracy" in selected_metrics and accuracy_metric is not None:
            # Flatten and filter out ignored tokens (-100)
            flat_predictions = predictions.flatten()
            flat_labels = labels.flatten()
            valid_mask = flat_labels != -100
            valid_predictions = flat_predictions[valid_mask]
            valid_labels = flat_labels[valid_mask]

            accuracy_result = accuracy_metric.compute(
                predictions=valid_predictions, references=valid_labels
            )
            results["accuracy"] = accuracy_result["accuracy"]

        # Compute perplexity from eval_loss (if available)
        if "perplexity" in selected_metrics:
            results["perplexity"] = 2**loss

        return results

    if not use_batched_eval:
        return compute_metrics

    # Storage for batch-level statistics
    batch_stats = {
        "total_correct": 0,
        "total_samples": 0,
        "total_loss": 0.0,
        "batch_count": 0,
    }

    def compute_metrics_batched(
        eval_pred: EvalPrediction, compute_result: bool = True
    ) -> dict:
        """
        Compute evaluation metrics using the evaluate library (batch mode).
        TODO: This does not yet use evaluate it simply tracks all the stat using a dict lol

        Args:
            eval_pred: EvalPrediction containing logits and labels
            compute_result: If True, compute final results. If False, accumulate batch stats.

        Returns:
            Dictionary of computed metrics
        """
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        # If not computing final result, accumulate batch stats
        if not compute_result:
            # Accumulate batch-level statistics
            flat_predictions = predictions.flatten()
            flat_labels = labels.flatten()
            valid_mask = flat_labels != -100
            valid_predictions = flat_predictions[valid_mask]
            valid_labels = flat_labels[valid_mask]

            # Accumulate accuracy stats
            if "accuracy" in selected_metrics:
                correct = (valid_predictions == valid_labels).sum()
                batch_stats["total_correct"] += correct
                batch_stats["total_samples"] += len(valid_labels)

            # Accumulate loss if available (for perplexity)
            if "perplexity" in selected_metrics and hasattr(eval_pred, "metrics"):
                batch_loss = eval_pred.metrics.get("eval_loss", 0)
                batch_stats["total_loss"] += batch_loss
                batch_stats["batch_count"] += 1

            # Return empty dict for batch accumulation
            return {}

        # Compute final results from accumulated statistics
        results = {}

        if "accuracy" in selected_metrics and batch_stats["total_samples"] > 0:
            accuracy = batch_stats["total_correct"] / batch_stats["total_samples"]
            results["accuracy"] = accuracy

        if "perplexity" in selected_metrics and batch_stats["batch_count"] > 0:
            avg_loss = batch_stats["total_loss"] / batch_stats["batch_count"]
            results["perplexity"] = 2**avg_loss

        results["loss"] = (
            batch_stats["total_loss"] / batch_stats["batch_count"]
            if batch_stats["batch_count"] > 0
            else 0
        )

        # Reset batch stats for next evaluation
        batch_stats.update(
            {
                "total_correct": 0,
                "total_samples": 0,
                "total_loss": 0.0,
                "batch_count": 0,
            }
        )

        return results

    return compute_metrics_batched
