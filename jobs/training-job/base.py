from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

from schema import TrainRequest, WandbConfig, TrainingConfig
from utils import run_evaluation, save_and_track
from job_manager import JobTracker


class BaseTrainingService(ABC):
    """
    Template Method for training services.
    Defines the standard workflow; subclasses implement hooks below.
    """

    def run_training(self, req: TrainRequest, tracker: JobTracker) -> Dict[str, Any]:
        # 1. Preparation
        tracker.preparing()

        # 2. Download datasets
        train_ds, eval_ds = self._download_dataset(req.processed_dataset_id)

        # 3. Model + tokenizer setup
        model, tokenizer = self._setup_model(req.training_config)

        # 4. Dataset preparation
        train_dataset, eval_dataset = self._prepare_dataset(
            train_ds, eval_ds, tokenizer, req.training_config
        )

        # 5. Optional PEFT wrapping
        model = self._apply_peft_if_needed(model, req.training_config)

        # 6. WandB initialization
        report_to, wandb_url = self._setup_wandb(req.wandb_config, tracker.job_id)

        # 7. Build training arguments
        training_args = self._build_training_args(
            req.training_config, tracker.job_id, report_to
        )

        # 8. Instantiate trainer
        trainer = self._create_trainer(
            model,
            tokenizer,
            train_dataset,
            eval_dataset,
            training_args,
            req.training_config,
        )

        # 9. Train
        tracker.training(wandb_url)
        trainer.train()

        # 10. Evaluate
        metrics = self._evaluate_if_needed(trainer, eval_dataset)

        # 11. Save + record in tracker
        artifact = self._save_and_track(model, tokenizer, tracker, metrics, req)

        return {
            "adapter_path": artifact.remote_path,
            "base_model_id": artifact.base_model_id,
        }

    # --- Hooks to implement in subclasses ---
    @abstractmethod
    def _download_dataset(self, dataset_id: str) -> Tuple[Any, Any]: ...

    @abstractmethod
    def _setup_model(self, cfg: TrainingConfig) -> Tuple[Any, Any]: ...

    @abstractmethod
    def _prepare_dataset(
        self, train_ds: Any, eval_ds: Any, tokenizer: Any, cfg: TrainingConfig
    ) -> Tuple[Any, Any]: ...

    @abstractmethod
    def _apply_peft_if_needed(self, model: Any, cfg: TrainingConfig) -> Any: ...

    @abstractmethod
    def _build_training_args(
        self, cfg: TrainingConfig, job_id: str, report_to: str
    ) -> Any: ...

    @abstractmethod
    def _create_trainer(
        self,
        model: Any,
        tokenizer: Any,
        train_ds: Any,
        eval_ds: Any,
        args: Any,
        cfg: TrainingConfig,
    ) -> Any: ...

    # --- Default behaviors ---
    def _evaluate_if_needed(
        self, trainer: Any, eval_ds: Any
    ) -> Optional[Dict[str, Any]]:
        return run_evaluation(trainer) if eval_ds is not None else None

    def _save_and_track(
        self,
        model: Any,
        tokenizer: Any,
        tracker: JobTracker,
        metrics: Optional[Dict[str, Any]],
        req: TrainRequest,
    ) -> Any:
        # derive file prefix from modality
        prefix = "hf_vision" if req.training_config.modality == "vision" else "adapter"
        return save_and_track(
            req.export,
            model,
            tokenizer,
            tracker.job_id,
            req.training_config.base_model_id,
            req.training_config.provider == "unsloth",
            req.hf_repo_id or "",
            tracker,
            metrics,
            tmp_prefix=prefix,
        )

    def _setup_wandb(
        self, config: Optional[WandbConfig], job_id: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Setup WandB and return (report_to, wandb_url).
        """
        if not config or not config.api_key:
            return "none", ""

        # Delay import to avoid dependency at module load
        import wandb

        # Login
        wandb.login(key=config.api_key)

        wandb.init(
            project=config.project or "gemma-fine-tuning",
            name=job_id,
            tags=["fine-tuning"],
        )

        wandb_url = wandb.run.get_url() if wandb.run else ""
        return "wandb", wandb_url
