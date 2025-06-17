from abc import ABC, abstractmethod
from typing import Union, List, Dict
import logging

logger = logging.getLogger(__name__)


class StorageInterface(ABC):
    """Base interface for storage operations"""

    @abstractmethod
    async def upload_data(
        self, data: Union[str, List[Dict], bytes], path: str, metadata: Dict = None
    ) -> str:
        """Upload data to storage and return the storage path"""
        pass

    @abstractmethod
    async def download_data(self, path: str) -> str:
        """Download data from storage as text"""
        pass

    @abstractmethod
    def list_files(self, prefix: str = "") -> List[str]:
        """List files with optional prefix filter"""
        pass

    @abstractmethod
    def get_metadata(self, path: str) -> Dict:
        """Get file metadata"""
        pass

    @abstractmethod
    def file_exists(self, path: str) -> bool:
        """Check if file exists"""
        pass

    @abstractmethod
    def delete_file(self, path: str) -> bool:
        """Delete file from storage"""
        pass
