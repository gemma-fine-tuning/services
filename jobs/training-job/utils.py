import logging
import torch
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


def compute_metrics(eval_pred):
    """
    Compute simple evaluation metrics (perplexity and accuracy) for HuggingFace Trainer.
    """
    logits, labels = eval_pred
    # Convert ignored token labels (-100) to 0 for perplexity
    labels[labels == -100] = 0

    # Ensure logits and labels are on CPU
    logits = torch.from_numpy(logits).cpu()
    labels = torch.from_numpy(labels).cpu()

    # Predictions: argmax over logits
    predictions = logits.argmax(axis=-1)

    # Filter out ignored tokens for accuracy
    valid_labels = labels[labels != -100]
    valid_predictions = predictions[labels != -100]
    accuracy = (valid_predictions == valid_labels).float().mean().item()

    # Perplexity from eval_loss if provided in eval_pred.metrics
    try:
        perp = 2 ** eval_pred.metrics.get("eval_loss", 0)
    except Exception:
        perp = None

    return {"perplexity": perp, "accuracy": accuracy}
