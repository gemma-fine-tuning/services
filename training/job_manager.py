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

    def create_job(
        self, job_id: str, processed_dataset_id: str, base_model_id: str
    ) -> JobMetadata:
        """
        Create a new job with QUEUED status.

        Args:
            job_id: Unique job identifier
            processed_dataset_id: ID of the processed dataset
            base_model_id: Base model identifier

        Returns:
            JobMetadata: Created job metadata
        """
        now = datetime.now(timezone.utc)
        job_data = {
            "job_id": job_id,
            "status": JobStatus.QUEUED.value,
            "created_at": now,
            "updated_at": now,
            "processed_dataset_id": processed_dataset_id,
            "base_model_id": base_model_id,
            "adapter_path": None,
            "wandb_url": None,
            "error": None,
            "progress_info": {},
        }

        try:
            self.collection.document(job_id).set(job_data)
            self.logger.info(
                f"Created job {job_id} with status {JobStatus.QUEUED.value}"
            )

            return JobMetadata(
                job_id=job_id,
                status=JobStatus.QUEUED,
                created_at=now,
                updated_at=now,
                processed_dataset_id=processed_dataset_id,
                base_model_id=base_model_id,
            )
        except Exception as e:
            self.logger.error(f"Failed to create job {job_id}: {e}")
            raise

    def update_status(
        self,
        job_id: str,
        status: JobStatus,
        error: Optional[str] = None,
        progress_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update job status with optional error and progress info.

        Args:
            job_id: Job identifier
            status: New job status
            error: Error message if status is FAILED
            progress_info: Additional progress information
        """
        update_data = {"status": status.value, "updated_at": datetime.now(timezone.utc)}

        if error:
            update_data["error"] = error

        if progress_info:
            update_data["progress_info"] = progress_info

        try:
            self.collection.document(job_id).update(update_data)
            self.logger.info(f"Updated job {job_id} status to {status.value}")
        except Exception as e:
            self.logger.error(f"Failed to update job {job_id} status: {e}")
            raise

    def set_wandb_url(self, job_id: str, wandb_url: str) -> None:
        """
        Set the WandB URL for a job.

        Args:
            job_id: Job identifier
            wandb_url: WandB run URL
        """
        try:
            self.collection.document(job_id).update(
                {"wandb_url": wandb_url, "updated_at": datetime.now(timezone.utc)}
            )
            self.logger.info(f"Set WandB URL for job {job_id}: {wandb_url}")
        except Exception as e:
            self.logger.error(f"Failed to set WandB URL for job {job_id}: {e}")
            raise

    def set_adapter_path(self, job_id: str, adapter_path: str) -> None:
        """
        Set the adapter path for a completed job.

        Args:
            job_id: Job identifier
            adapter_path: Path to the trained adapter
        """
        try:
            self.collection.document(job_id).update(
                {"adapter_path": adapter_path, "updated_at": datetime.now(timezone.utc)}
            )
            self.logger.info(f"Set adapter path for job {job_id}: {adapter_path}")
        except Exception as e:
            self.logger.error(f"Failed to set adapter path for job {job_id}: {e}")
            raise

    def get_job(self, job_id: str) -> Optional[JobMetadata]:
        """
        Retrieve job metadata by ID.

        Args:
            job_id: Job identifier

        Returns:
            JobMetadata or None if job not found
        """
        try:
            doc = self.collection.document(job_id).get()
            if not doc.exists:
                return None

            data = doc.to_dict()
            return JobMetadata(
                job_id=data["job_id"],
                status=JobStatus(data["status"]),
                created_at=data["created_at"],
                updated_at=data["updated_at"],
                processed_dataset_id=data["processed_dataset_id"],
                base_model_id=data["base_model_id"],
                adapter_path=data.get("adapter_path"),
                wandb_url=data.get("wandb_url"),
                error=data.get("error"),
                progress_info=data.get("progress_info", {}),
            )
        except Exception as e:
            self.logger.error(f"Failed to get job {job_id}: {e}")
            raise

    def mark_preparing(
        self, job_id: str, info: Optional[Dict[str, Any]] = None
    ) -> None:
        """Mark job as preparing (loading model, dataset, etc.)"""
        self.update_status(job_id, JobStatus.PREPARING, progress_info=info)

    def mark_training(self, job_id: str, wandb_url: Optional[str] = None) -> None:
        """
        Mark job as training and optionally set WandB URL.

        Args:
            job_id: Job identifier
            wandb_url: WandB run URL if available
        """
        self.update_status(job_id, JobStatus.TRAINING)
        if wandb_url:
            self.set_wandb_url(job_id, wandb_url)

    def mark_completed(self, job_id: str, adapter_path: str) -> None:
        """
        Mark job as completed with adapter path.

        Args:
            job_id: Job identifier
            adapter_path: Path to the trained adapter
        """
        self.update_status(job_id, JobStatus.COMPLETED)
        self.set_adapter_path(job_id, adapter_path)

    def mark_failed(self, job_id: str, error: str) -> None:
        """
        Mark job as failed with error message.

        Args:
            job_id: Job identifier
            error: Error description
        """
        self.update_status(job_id, JobStatus.FAILED, error=error)

    def get_job_status_dict(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get job status as dictionary for API responses.

        Args:
            job_id: Job identifier

        Returns:
            Dictionary with job status info or None if not found
        """
        job = self.get_job(job_id)
        if not job:
            return None

        return {
            "job_id": job.job_id,
            "status": job.status.value,
            "created_at": job.created_at.isoformat(),
            "updated_at": job.updated_at.isoformat(),
            "processed_dataset_id": job.processed_dataset_id,
            "base_model_id": job.base_model_id,
            "adapter_path": job.adapter_path,
            "wandb_url": job.wandb_url,
            "error": job.error,
            "progress_info": job.progress_info,
        }


# Context manager for automatic status updates
class JobTracker:
    """
    Context manager that automatically handles job status transitions.
    Usage:
        with JobTracker(job_manager, job_id) as tracker:
            tracker.preparing("Loading model...")
            # do preparation work
            tracker.training(wandb_url="https://...")
            # do training work
            tracker.completed(adapter_path="/path/to/adapter")
    """

    def __init__(self, job_manager: JobStateManager, job_id: str):
        self.job_manager = job_manager
        self.job_id = job_id
        self._error_occurred = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type and not self._error_occurred:
            # Automatically mark as failed if an exception occurred
            error_msg = f"{exc_type.__name__}: {str(exc_val)}"
            self.job_manager.mark_failed(self.job_id, error_msg)
        return False  # Don't suppress exceptions

    def preparing(self, info: str = None):
        """Mark job as preparing"""
        progress_info = {"message": info} if info else None
        self.job_manager.mark_preparing(self.job_id, progress_info)

    def training(self, wandb_url: str = None):
        """Mark job as training"""
        self.job_manager.mark_training(self.job_id, wandb_url)

    def completed(self, adapter_path: str):
        """Mark job as completed"""
        self.job_manager.mark_completed(self.job_id, adapter_path)

    def failed(self, error: str):
        """Mark job as failed"""
        self._error_occurred = True
        self.job_manager.mark_failed(self.job_id, error)
