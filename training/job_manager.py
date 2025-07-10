import logging
from enum import Enum
from typing import Optional, Dict, Any
from google.cloud import firestore
from dataclasses import dataclass
from datetime import datetime, timezone


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
    job_name: str
    adapter_path: Optional[str] = None
    wandb_url: Optional[str] = None
    error: Optional[str] = None


class JobStateManager:
    """
    Centralized job state management using Firestore using repository pattern.
    Provides clean API for tracking training job progress.
    """

    def __init__(self, project_id: str, collection_name: str = "training_jobs"):
        """
        Initialize job state manager.

        Args:
            project_id: Google Cloud project ID
            collection_name: Name of the Firestore collection for jobs
        """
        self.db = firestore.Client(project=project_id)
        self.collection = self.db.collection(collection_name)
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
                job_name=data["job_name"],
                status=JobStatus(data["status"]),
                created_at=data["created_at"],
                updated_at=data["updated_at"],
                processed_dataset_id=data["processed_dataset_id"],
                base_model_id=data["base_model_id"],
                adapter_path=data.get("adapter_path"),
                wandb_url=data.get("wandb_url"),
                error=data.get("error"),
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
            "job_name": job.job_name,
            "status": job.status.value,
            "created_at": job.created_at.isoformat(),
            "updated_at": job.updated_at.isoformat(),
            "processed_dataset_id": job.processed_dataset_id,
            "base_model_id": job.base_model_id,
            "adapter_path": job.adapter_path,
            "wandb_url": job.wandb_url,
            "error": job.error,
        }

    def ensure_job_document_exists(
        self, job_id: str, job_metadata: Optional[JobMetadata] = None
    ):
        """
        Ensure a job document exists in Firestore. If not, create it with the provided metadata or minimal info.
        """
        doc_ref = self.collection.document(job_id)
        doc = doc_ref.get()
        if not doc.exists:
            if job_metadata is None:
                # Create with minimal info
                job_metadata = JobMetadata(
                    job_id=job_id,
                    job_name="unnamed job",
                    status=JobStatus.QUEUED,
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                    processed_dataset_id="",
                    base_model_id="",
                )
            doc_ref.set(
                {
                    "job_id": job_metadata.job_id,
                    "job_name": job_metadata.job_name,
                    "status": job_metadata.status.value,
                    "created_at": job_metadata.created_at,
                    "updated_at": job_metadata.updated_at,
                    "processed_dataset_id": job_metadata.processed_dataset_id,
                    "base_model_id": job_metadata.base_model_id,
                    "adapter_path": job_metadata.adapter_path,
                    "wandb_url": job_metadata.wandb_url,
                    "error": job_metadata.error,
                }
            )

    def list_jobs(self) -> list:
        """
        List all jobs with job_id and job_name.
        """
        jobs = []
        try:
            docs = self.collection.stream()
            for doc in docs:
                data = doc.to_dict()
                jobs.append(
                    {
                        "job_id": data.get("job_id"),
                        "job_name": data.get("job_name"),
                        "job_status": data.get("status"),
                    }
                )
            return jobs
        except Exception as e:
            self.logger.error(f"Failed to list jobs: {e}")
            raise
