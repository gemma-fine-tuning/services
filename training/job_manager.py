import logging
from enum import Enum
from typing import Optional, Dict, Any
from google.cloud import firestore
from dataclasses import dataclass
from datetime import datetime


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
            if not data:
                return None
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
