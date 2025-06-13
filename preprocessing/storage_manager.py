import json
import logging
from typing import Union, List, Dict
from google.cloud import storage

logger = logging.getLogger(__name__)


class StorageManager:
    """Enhanced version of existing GCS functions"""

    def __init__(self, bucket_name: str):
        self.client = storage.Client()
        self.bucket_name = bucket_name
        self.bucket = self.client.bucket(bucket_name)

    async def upload_data(
        self, data: Union[str, List[Dict], bytes], path: str, metadata: Dict = None
    ) -> str:
        """Enhanced upload_to_gcs with metadata"""
        try:
            blob = self.bucket.blob(path)

            if isinstance(data, bytes):
                blob.upload_from_string(data)
            elif isinstance(data, str):
                blob.upload_from_string(data, content_type="text/plain")
            else:
                blob.upload_from_string(
                    json.dumps(data), content_type="application/json"
                )

            # Add metadata if provided
            if metadata:
                blob.metadata = metadata
                blob.patch()

            return f"gs://{self.bucket_name}/{path}"

        except Exception as e:
            logger.error(f"Error uploading to GCS: {str(e)}")
            raise

    async def download_data(self, path: str) -> str:
        """Enhanced download_from_gcs"""
        try:
            blob = self.bucket.blob(path)
            return blob.download_as_text()
        except Exception as e:
            logger.error(f"Error downloading from GCS: {str(e)}")
            raise

    def list_datasets(self, prefix: str = "") -> List[str]:
        """List available datasets"""
        try:
            blobs = self.bucket.list_blobs(prefix=prefix)
            return [blob.name for blob in blobs]
        except Exception as e:
            logger.error(f"Error listing datasets: {str(e)}")
            raise

    def get_metadata(self, path: str) -> Dict:
        """Get file metadata"""
        try:
            blob = self.bucket.blob(path)
            if blob.exists():
                return {
                    "size": blob.size,
                    "created": blob.time_created.isoformat()
                    if blob.time_created
                    else None,
                    "updated": blob.updated.isoformat() if blob.updated else None,
                    "metadata": blob.metadata or {},
                }
            return {}
        except Exception as e:
            logger.error(f"Error getting metadata: {str(e)}")
            raise

    def cleanup_old_data(self, retention_days: int = 30):
        """Automatic cleanup - placeholder for future implementation"""
        pass

    def blob_exists(self, path: str) -> bool:
        """Check if blob exists"""
        blob = self.bucket.blob(path)
        return blob.exists()
