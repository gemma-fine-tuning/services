import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Dict, Any
from google.cloud import firestore
from dataclasses import dataclass


class JobStatus(Enum):
    """Enum for job status values"""

    QUEUED = "queued"
    PREPARING = "preparing"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class JobMetadata:
    """Job metadata structure"""

    job_id: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    processed_dataset_id: str
    base_model_id: str
    adapter_path: Optional[str] = None
    wandb_url: Optional[str] = None
    error: Optional[str] = None
    progress_info: Optional[Dict[str, Any]] = None


class JobStateManager:
    """
    Centralized job state management using Firestore using repository pattern.
    Provides clean API for tracking training job progress.
    """

    def __init__(
        self, db_client: firestore.Client, collection_name: str = "training_jobs"
    ):
        """
        Initialize job state manager.

        Args:
            db_client: Firestore client instance
            collection_name: Name of the Firestore collection for jobs
        """
        self.db = db_client
        self.collection = db_client.collection(collection_name)
        self.logger = logging.getLogger(__name__)

    def ensure_job_document_exists(
        self, job_id: str, job_metadata: Optional[JobMetadata] = None
    ):
        doc_ref = self.collection.document(job_id)
        if not doc_ref.get().exists:
            if job_metadata:
                doc_ref.set(
                    {
                        **job_metadata.__dict__,
                        "status": job_metadata.status.value
                        if isinstance(job_metadata.status, Enum)
                        else job_metadata.status,
                    }
                )
            else:
                doc_ref.set(
                    {
                        "status": JobStatus.QUEUED.value,
                        "created_at": datetime.now(timezone.utc),
                    }
                )

    def mark_preparing(
        self, job_id: str, info: Optional[Dict[str, Any]] = None
    ) -> None:
        self.collection.document(job_id).update(
            {
                "status": JobStatus.PREPARING.value,
                "updated_at": datetime.now(timezone.utc),
                "progress_info": info or {},
            }
        )
        self.logger.info(f"Marked job {job_id} as preparing")

    def mark_training(self, job_id: str, wandb_url: Optional[str] = None) -> None:
        update_data = {
            "status": JobStatus.TRAINING.value,
            "updated_at": datetime.now(timezone.utc),
        }
        if wandb_url:
            update_data["wandb_url"] = wandb_url
        self.collection.document(job_id).update(update_data)
        self.logger.info(f"Marked job {job_id} as training")

    def mark_completed(
        self, job_id: str, adapter_path: str, base_model_id: str
    ) -> None:
        self.collection.document(job_id).update(
            {
                "status": JobStatus.COMPLETED.value,
                "updated_at": datetime.now(timezone.utc),
                "adapter_path": adapter_path,
                "base_model_id": base_model_id,
            }
        )
        self.logger.info(f"Marked job {job_id} as completed")

    def mark_failed(self, job_id: str, error: str) -> None:
        self.collection.document(job_id).update(
            {
                "status": JobStatus.FAILED.value,
                "updated_at": datetime.now(timezone.utc),
                "error": error,
            }
        )
        self.logger.info(f"Marked job {job_id} as failed: {error}")


# Context manager for automatic status updates
class JobTracker:
    """
    Context manager that automatically handles job status transitions.
    Usage:
        with JobTracker(job_manager, job_id, job_metadata) as tracker:
            tracker.preparing("Loading model...")
            # do preparation work
            tracker.training(wandb_url="https://...")
            # do training work
            tracker.completed(adapter_path="/path/to/adapter")
    """

    def __init__(
        self,
        job_manager: JobStateManager,
        job_id: str,
        job_metadata: Optional[JobMetadata] = None,
    ):
        self.job_manager = job_manager
        self.job_id = job_id
        self.job_metadata = job_metadata
        self._error_occurred = False

    def __enter__(self):
        self.job_manager.ensure_job_document_exists(self.job_id, self.job_metadata)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type and not self._error_occurred:
            error_msg = f"{exc_type.__name__}: {str(exc_val)}"
            self.job_manager.mark_failed(self.job_id, error_msg)
        return False  # Don't suppress exceptions

    def preparing(self, info: Optional[str] = None):
        """Mark job as preparing"""
        progress_info = {"message": info} if info else None
        self.job_manager.mark_preparing(self.job_id, progress_info)

    def training(self, wandb_url: Optional[str] = None):
        """Mark job as training"""
        self.job_manager.mark_training(self.job_id, wandb_url)

    def completed(self, adapter_path: str, base_model_id: str):
        """Mark job as completed"""
        self.job_manager.mark_completed(self.job_id, adapter_path, base_model_id)

    def failed(self, error: str):
        """Mark job as failed"""
        self._error_occurred = True
        self.job_manager.mark_failed(self.job_id, error)
