import json
import logging
from typing import Union, List, Dict
from google.cloud import storage
from .base import StorageInterface

logger = logging.getLogger(__name__)

# TODO: GCS MODULE STILL NEEDS TESTING


class GCSStorageManager(StorageInterface):
    """Google Cloud Storage implementation"""

    def __init__(self, bucket_name: str):
        self.client = storage.Client()
        self.bucket_name = bucket_name
        self.bucket = self.client.bucket(bucket_name)

    async def upload_data(
        self, data: Union[str, List[Dict], bytes], path: str, metadata: Dict = None
    ) -> str:
        """Upload data to GCS"""
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

            if metadata:
                blob.metadata = metadata
                blob.patch()

            return f"gs://{self.bucket_name}/{path}"

        except Exception as e:
            logger.error(f"Error uploading to GCS: {str(e)}")
            raise

    async def download_data(self, path: str) -> str:
        """Download data from GCS"""
        try:
            blob = self.bucket.blob(path)
            return blob.download_as_text()
        except Exception as e:
            logger.error(f"Error downloading from GCS: {str(e)}")
            raise

    async def download_binary_data(self, path: str) -> bytes:
        """Download binary data from GCS"""
        try:
            blob = self.bucket.blob(path)
            return blob.download_as_bytes()
        except Exception as e:
            logger.error(f"Error downloading binary data from GCS: {str(e)}")
            raise

    def list_files(self, prefix: str = "") -> List[str]:
        """List files in GCS bucket"""
        try:
            blobs = self.bucket.list_blobs(prefix=prefix)
            return [blob.name for blob in blobs]
        except Exception as e:
            logger.error(f"Error listing files: {str(e)}")
            raise

    def get_metadata(self, path: str) -> Dict:
        """Get file metadata from GCS"""
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

    def file_exists(self, path: str) -> bool:
        """Check if file exists in GCS"""
        blob = self.bucket.blob(path)
        return blob.exists()

    def delete_file(self, path: str) -> bool:
        """Delete file from GCS"""
        try:
            blob = self.bucket.blob(path)
            if blob.exists():
                blob.delete()
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting file: {str(e)}")
            raise
