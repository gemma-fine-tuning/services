import logging
import numpy as np
from typing import List, Dict, Any, Tuple
from datasets import Dataset
import evaluate

logger = logging.getLogger(__name__)


class EvaluationSuite:
    """
    Evaluation suite for computing metrics on model generations.
    Supports individual metrics and combined metric suites for specific tasks.
    """

    def __init__(self):
        # Combined metric suites for specific task types
        self.task_metric_suites = {
            "conversation": ["bertscore", "rouge"],
            "qa": ["exact_match", "bertscore"],
            "summarization": ["rouge", "bertscore"],
            "translation": ["bleu", "meteor"],
            "classification": ["accuracy", "recall", "precision", "f1"],
            "general": ["bertscore", "rouge"],
        }

        # stores loaded individual or grouped metrics
        self.loaded_metrics = {}

    def _load_metric(self, metric_name: str):
        """Load a single metric by name."""
        try:
            return evaluate.load(metric_name)
        except Exception as e:
            logger.warning(f"Could not load metric {metric_name}: {e}")
            return None

    def _load_combined_metrics(self, metric_names: List[str]):
        """Load and combine multiple metrics into one evaluator."""
        try:
            return evaluate.combine(metric_names)
        except Exception as e:
            logger.warning(f"Could not load combined metrics {metric_names}: {e}")
            return None

    def load_metrics(self, metric_names: List[str]):
        """Load specified metrics individually."""
        for metric_name in metric_names:
            if metric_name not in self.loaded_metrics:
                metric = self._load_metric(metric_name)
                if metric is not None:
                    self.loaded_metrics[metric_name] = metric
                    logger.info(f"Loaded metric: {metric_name}")

    def compute_metrics(
        self,
        predictions: List[str],
        references: List[str],
        task_type: str = None,
        metrics: List[str] = None,
        num_samples: int = 3,
    ) -> Dict[str, Any]:
        """
        Compute metrics on predictions vs references.

        Args:
            predictions: List of model generated texts
            references: List of reference/ground truth texts
            task_type: Task type to use predefined metric suite (mutually exclusive with metrics)
            metrics: Specific list of metrics to compute (mutually exclusive with task_type)
            num_samples: Number of sample results to include in the response

        Returns:
            Dictionary containing metric scores and sample results
        """
        # Determine which metrics to use
        if task_type:
            final_metrics = self.task_metric_suites.get(
                task_type, self.task_metric_suites["general"]
            )
        else:
            final_metrics = metrics or []

        # Load metrics
        self.load_metrics(final_metrics)

        results = {}  # results[metric_name] = int or dict

        # Compute individual metrics
        for metric_name in final_metrics:
            if metric_name not in self.loaded_metrics:
                logger.warning(f"Metric {metric_name} not available, skipping")
                continue

            metric = self.loaded_metrics[metric_name]
            try:
                if metric_name == "bleu":
                    # BLEU expects references as list of lists
                    refs = [[ref] for ref in references]
                    result = metric.compute(predictions=predictions, references=refs)
                    results[metric_name] = result["bleu"]

                elif metric_name == "bertscore":
                    result = metric.compute(
                        predictions=predictions, references=references, lang="en"
                    )
                    # This returns: {'precision': [0.7, 0.5], 'recall': [], 'f1': []} if there are two pred ref pairs
                    # This takes the mean across all the sample similarities
                    results["bertscore"] = {
                        "precision_mean": np.mean(result["precision"]),
                        "recall_mean": np.mean(result["recall"]),
                        "f1_mean": np.mean(result["f1"]),
                    }

                elif metric_name == "f1":
                    # F1 returns {'f1': 0.26666666666666666} or {'f1': array([0.8, 0.0, 0.0])} depending on binary or multi-class
                    result = metric.compute(
                        predictions=predictions, references=references
                    )
                    if len(set(references)) <= 2:
                        results["f1"] = result["f1"]  # Binary classification case
                    else:
                        # Multi-class case, alternatively you can set average="micro"
                        results["f1"] = np.mean(result["f1"])
                        results["f1_per_class"] = result["f1"]

                # no need special handling
                else:
                    result = metric.compute(
                        predictions=predictions, references=references
                    )
                    # Handle different result formats
                    if isinstance(result, dict) and metric_name in result:
                        results[metric_name] = result[
                            metric_name
                        ]  # e.g. {"accuracy": 0.8}
                    else:
                        results[metric_name] = (
                            result  # e.g. {"rougeL": xxx, "rouge1": xxx, "rouge2": xxx}
                        )

            except Exception as e:
                logger.error(f"Error computing {metric_name}: {e}")
                results[metric_name] = 0.0

        # Add sample results for inspection
        sample_results = []
        if len(predictions) > 0:
            sample_indices = np.random.choice(
                len(predictions), min(num_samples, len(predictions)), replace=False
            )

            for idx in sample_indices:
                sample_results.append(
                    {
                        "prediction": predictions[idx],
                        "reference": references[idx],
                        "sample_index": int(idx),
                    }
                )

        return {"metrics": results, "samples": sample_results}


def prepare_evaluation_data(
    dataset: Dataset,
) -> Tuple[List[List[Dict[str, Any]]], List[str]]:
    """
    Prepare evaluation data by extracting both input messages and reference texts.
    This combines the functionality of prepare_evaluation_messages and extract_references_from_dataset
    for better efficiency (single pass through the dataset).

    Args:
        dataset: HuggingFace dataset with messages in ChatML format

    Returns:
        Tuple of (eval_messages, references) where:
        - eval_messages: List of message conversations with only user/system messages (for generation)
        - references: List of reference texts (assistant responses)
    """
    eval_messages = []
    references = []

    for example in dataset:
        messages = example.get("messages", [])

        # Prepare input messages (user/system only)
        user_messages = []
        reference = None

        for message in messages:
            if message.get("role") in ["user", "system"]:
                user_messages.append(message)
            elif message.get("role") == "assistant":
                # First assistant message becomes the reference
                if reference is None:
                    reference = message.get("content", "")
                # Stop at first assistant message - we want to generate from this point
                break

        # Add to results if we have user messages
        if user_messages:
            eval_messages.append(user_messages)
            # Add reference (or empty string if no assistant message found)
            if reference is not None:
                references.append(reference.strip())
            else:
                references.append("")
                logger.warning(
                    "No assistant message found in example, using empty reference"
                )
        else:
            logger.warning("No user messages found in example, skipping")

    return eval_messages, references
