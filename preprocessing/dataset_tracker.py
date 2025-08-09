import logging
from typing import List
from google.cloud import firestore
from dataclasses import dataclass
from datetime import datetime


@dataclass
class RawDatasetMetadata:
    """Metadata for uploaded raw datasets"""

    dataset_id: str
    gcs_path: str
    user_id: str
    filename: str
    content_type: str
    size_bytes: int
    created_at: datetime


@dataclass
class ProcessedDatasetMetadata:
    """Metadata for processed datasets"""

    processed_dataset_id: str
    dataset_name: str
    user_id: str
    source_dataset_id: str
    dataset_source: str
    created_at: str
    num_examples: int
    splits: List[str]


class DatasetTracker:
    """
    Centralized dataset metadata management using Firestore.
    Tracks both raw uploads and processed datasets with user ownership.
    """

    def __init__(self, project_id: str):
        """
        Initialize dataset tracker.

        Args:
            project_id: Google Cloud project ID
        """
        self.db = firestore.Client(project=project_id)
        self.raw_collection = self.db.collection("datasets")
        self.processed_collection = self.db.collection("processed_datasets")
        self.logger = logging.getLogger(__name__)

    def track_raw_dataset(self, metadata: RawDatasetMetadata) -> None:
        """
        Track a raw uploaded dataset.

        Args:
            metadata: Raw dataset metadata
        """
        try:
            self.raw_collection.document(metadata.dataset_id).set(
                {
                    "dataset_id": metadata.dataset_id,
                    "gcs_path": metadata.gcs_path,
                    "user_id": metadata.user_id,
                    "filename": metadata.filename,
                    "content_type": metadata.content_type,
                    "size_bytes": metadata.size_bytes,
                    "created_at": metadata.created_at,
                }
            )
            self.logger.info(f"Tracked raw dataset: {metadata.dataset_id}")
        except Exception as e:
            self.logger.error(f"Failed to track raw dataset {metadata.dataset_id}: {e}")
            raise

    def track_processed_dataset(self, metadata: ProcessedDatasetMetadata) -> None:
        """
        Track a processed dataset using its unique processed_dataset_id.

        Args:
            metadata: Processed dataset metadata
        """
        try:
            self.processed_collection.document(metadata.processed_dataset_id).set(
                {
                    "processed_dataset_id": metadata.processed_dataset_id,
                    "dataset_name": metadata.dataset_name,
                    "user_id": metadata.user_id,
                    "source_dataset_id": metadata.source_dataset_id,
                    "dataset_source": metadata.dataset_source,
                    "created_at": metadata.created_at,
                    "num_examples": metadata.num_examples,
                    "splits": metadata.splits,
                }
            )
            self.logger.info(
                f"Tracked processed dataset: {metadata.processed_dataset_id}"
            )
        except Exception as e:
            self.logger.error(
                f"Failed to track processed dataset {metadata.processed_dataset_id}: {e}"
            )
            raise

    def verify_raw_dataset_ownership(self, dataset_id: str, user_id: str) -> bool:
        """
        Verify that a user owns a raw dataset.

        Args:
            dataset_id: Raw dataset ID
            user_id: User ID

        Returns:
            True if user owns the dataset, False otherwise
        """
        try:
            doc = self.raw_collection.document(dataset_id).get()
            if not doc.exists:
                return False
            return doc.to_dict().get("user_id") == user_id
        except Exception as e:
            self.logger.error(
                f"Failed to verify raw dataset ownership {dataset_id}: {e}"
            )
            return False

    def verify_processed_dataset_ownership(
        self, processed_dataset_id: str, user_id: str
    ) -> bool:
        """
        Verify that a user owns a processed dataset.

        Args:
            processed_dataset_id: Processed dataset unique ID
            user_id: User ID

        Returns:
            True if user owns the dataset, False otherwise
        """
        try:
            doc = self.processed_collection.document(processed_dataset_id).get()
            if not doc.exists:
                return False
            return doc.to_dict().get("user_id") == user_id
        except Exception as e:
            self.logger.error(
                f"Failed to verify processed dataset ownership {processed_dataset_id}: {e}"
            )
            return False

    def get_user_processed_datasets(self, user_id: str) -> List[str]:
        """
        Get all processed dataset IDs owned by a user.

        Args:
            user_id: User ID

        Returns:
            List of processed dataset IDs
        """
        try:
            docs = self.processed_collection.where("user_id", "==", user_id).stream()
            return [doc.id for doc in docs]
        except Exception as e:
            self.logger.error(f"Failed to get user processed datasets: {e}")
            return []

    def delete_processed_dataset_metadata(self, processed_dataset_id: str) -> bool:
        """
        Delete processed dataset metadata.

        Args:
            processed_dataset_id: Processed dataset unique ID

        Returns:
            True if deleted, False if not found
        """
        try:
            doc_ref = self.processed_collection.document(processed_dataset_id)
            doc = doc_ref.get()
            if doc.exists:
                doc_ref.delete()
                self.logger.info(
                    f"Deleted processed dataset metadata: {processed_dataset_id}"
                )
                return True
            return False
        except Exception as e:
            self.logger.error(
                f"Failed to delete processed dataset metadata {processed_dataset_id}: {e}"
            )
            raise

    def delete_raw_dataset_metadata(self, dataset_id: str) -> bool:
        """
        Delete raw dataset metadata.

        Args:
            dataset_id: Raw dataset ID

        Returns:
            True if deleted, False if not found
        """
        try:
            doc_ref = self.raw_collection.document(dataset_id)
            doc = doc_ref.get()
            if doc.exists:
                doc_ref.delete()
                self.logger.info(f"Deleted raw dataset metadata: {dataset_id}")
                return True
            return False
        except Exception as e:
            self.logger.error(
                f"Failed to delete raw dataset metadata {dataset_id}: {e}"
            )
            raise
