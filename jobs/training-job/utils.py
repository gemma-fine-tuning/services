import logging
import evaluate
from transformers import EvalPrediction
import numpy as np
from typing import List, Callable
from storage import StorageStrategyFactory
from job_manager import JobTracker


def run_evaluation(trainer):
    """
    Run evaluation on the trainer, log and return metrics.
    """
    eval_results = trainer.evaluate()
    logging.info(f"Evaluation results: {eval_results}")
    return eval_results


def save_and_track(
    export: str,
    model,
    tokenizer,
    job_id: str,
    base_model_id: str,
    use_unsloth: bool,
    hf_repo_id: str,
    job_tracker: JobTracker,
    metrics: dict | None = None,
    tmp_prefix: str = "adapter",
):
    """
    Save the model using storage strategy, cleanup, and mark job as completed.
    """
    storage_strategy = StorageStrategyFactory.create_strategy(export)
    artifact = storage_strategy.save_model(
        model,
        tokenizer,
        f"/tmp/{job_id}_{tmp_prefix}",
        {
            "job_id": job_id,
            "base_model_id": base_model_id,
            "use_unsloth": use_unsloth,
            "hf_repo_id": hf_repo_id,
        },
    )
    storage_strategy.cleanup(artifact)
    job_tracker.completed(artifact.remote_path, artifact.base_model_id, metrics)
    return artifact


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
        predictions = np.argmax(logits, axis=-1)
        results = {}

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
            loss = (
                eval_pred.metrics.get("eval_loss", 0)
                if hasattr(eval_pred, "metrics")
                else 0
            )
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
