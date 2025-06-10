"""
Simple training metrics logger - stores progress in memory for REST API access
"""

import time
import logging
from typing import Dict, Any, Optional
from transformers import TrainerCallback


class TrainingMetricsLogger(TrainerCallback):
    """Simple metrics logger that stores progress in memory"""

    # Class-level storage for training jobs
    _active_jobs: Dict[str, Dict[str, Any]] = {}

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.start_time = time.time()

        # Initialize job data
        TrainingMetricsLogger._active_jobs[job_id] = {
            "job_id": job_id,
            "status": "starting",
            "start_time": self.start_time,
            "current_step": 0,
            "total_steps": 0,
            "current_epoch": 0,
            "total_epochs": 0,
            "progress": 0.0,
            "latest_loss": None,
            "latest_lr": None,
            "elapsed_time": 0,
            "message": "Training starting...",
            "metrics_history": [],
        }

        logging.info(f"Training metrics logger initialized for job {job_id}")

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Called when training begins"""
        self._update_job_data(
            {
                "status": "training",
                "total_steps": state.max_steps,
                "total_epochs": state.num_train_epochs,
                "message": "Training in progress...",
            }
        )

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """Called when training logs are generated"""
        if logs is None:
            return

        current_time = time.time()
        elapsed_time = current_time - self.start_time

        # Extract metrics
        latest_loss = logs.get("train_loss")
        latest_lr = logs.get("learning_rate")

        # Calculate progress
        progress = state.global_step / state.max_steps if state.max_steps > 0 else 0

        # Update job data
        update_data = {
            "current_step": state.global_step,
            "current_epoch": state.epoch,
            "progress": progress,
            "elapsed_time": elapsed_time,
            "latest_loss": latest_loss,
            "latest_lr": latest_lr,
            "message": f"Step {state.global_step}/{state.max_steps} - Loss: {latest_loss:.4f}"
            if latest_loss
            else f"Step {state.global_step}/{state.max_steps}",
        }

        # Add to history
        metric_entry = {
            "step": state.global_step,
            "epoch": state.epoch,
            "timestamp": current_time,
            "train_loss": latest_loss,
            "learning_rate": latest_lr,
        }

        job_data = TrainingMetricsLogger._active_jobs.get(self.job_id, {})
        history = job_data.get("metrics_history", [])
        history.append(metric_entry)

        # Keep only last 100 entries to avoid memory issues
        if len(history) > 100:
            history = history[-100:]

        update_data["metrics_history"] = history

        self._update_job_data(update_data)

        logging.info(
            f"Step {state.global_step}: Loss={latest_loss:.4f}"
            if latest_loss
            else f"Step {state.global_step}"
        )

    def on_train_end(self, args, state, control, model=None, logs=None, **kwargs):
        """Called when training ends"""
        end_time = time.time()
        total_time = end_time - self.start_time

        self._update_job_data(
            {
                "status": "completed",
                "progress": 1.0,
                "elapsed_time": total_time,
                "end_time": end_time,
                "message": f"Training completed in {total_time:.2f}s",
            }
        )

        logging.info(
            f"Training completed for job {self.job_id}. Total time: {total_time:.2f}s"
        )

    def on_train_error(self, error):
        """Called when training encounters an error"""
        self._update_job_data(
            {
                "status": "error",
                "error": str(error),
                "message": f"Training failed: {str(error)}",
            }
        )

        logging.error(f"Training error for job {self.job_id}: {error}")

    def _update_job_data(self, update_data: Dict[str, Any]):
        """Update job data in memory"""
        if self.job_id in TrainingMetricsLogger._active_jobs:
            TrainingMetricsLogger._active_jobs[self.job_id].update(update_data)
            TrainingMetricsLogger._active_jobs[self.job_id]["last_updated"] = (
                time.time()
            )

    @classmethod
    def get_job_status(cls, job_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a training job"""
        return cls._active_jobs.get(job_id)

    @classmethod
    def list_jobs(cls) -> Dict[str, Dict[str, Any]]:
        """List all active/recent jobs"""
        return cls._active_jobs.copy()

    @classmethod
    def cleanup_old_jobs(cls, max_age_hours: int = 24):
        """Clean up old job data"""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600

        jobs_to_remove = []
        for job_id, job_data in cls._active_jobs.items():
            last_updated = job_data.get("last_updated", job_data.get("start_time", 0))
            if current_time - last_updated > max_age_seconds:
                jobs_to_remove.append(job_id)

        for job_id in jobs_to_remove:
            del cls._active_jobs[job_id]
            logging.info(f"Cleaned up old job data for {job_id}")
