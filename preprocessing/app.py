import os
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from storage import GCSStorageManager, LocalStorageManager
from services.dataset_service import DatasetService
from schema import (
    DatasetUploadResponse,
    PreprocessingRequest,
    ProcessingResult,
    DatasetsInfoResponse,
    DatasetInfoResponse,
    DatasetDeleteResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Gemma Dataset Preprocessing Service",
    version="2.0.0",
    description="A modular service for preprocessing datasets into ChatML format",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

storage_type = os.getenv("STORAGE_TYPE", "local")  # "gcs" or "local"

if storage_type == "gcs":
    bucket_name = os.getenv("GCS_DATA_BUCKET_NAME", "gemma-dataset-bucket")
    storage_manager = GCSStorageManager(bucket_name)
    logger.info(f"Using GCS storage with bucket: {bucket_name}")
else:
    data_path = os.getenv("LOCAL_DATA_PATH", "./data")
    storage_manager = LocalStorageManager(data_path)
    logger.info(f"Using local storage at: {data_path}")

dataset_service = DatasetService(storage_manager)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "preprocessing",
        "version": "2.0.0",
        "storage_type": storage_type,
    }


@app.post("/upload", response_model=DatasetUploadResponse)
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a dataset file to storage"""
    try:
        file_content = await file.read()

        result = dataset_service.upload_dataset(
            file_data=file_content,
            filename=file.filename or "unknown",
            metadata={"content_type": file.content_type},
        )

        return result

    except Exception as e:
        logger.error(f"Error uploading dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/process", response_model=ProcessingResult)
def process_dataset(request: PreprocessingRequest):
    """Process a dataset into ChatML format

    The processing converts the dataset to ChatML format using the provided configuration.
    The configuration can include field mappings that specify either direct column mappings or template strings
    with column references.

    Example field mappings:
    ```python
    {
        "system_field": {"type": "template", "value": "You are a helpful assistant."},
        "user_field": {"type": "column", "value": "question"},
        "assistant_field": {"type": "template", "value": "Answer: {answer}"}
    }
    ```
    """
    try:
        result = dataset_service.process_dataset(
            dataset_name=request.dataset_name,
            dataset_source=request.dataset_source,
            dataset_id=request.dataset_id,
            dataset_subset=request.dataset_subset,
            config=request.config,
        )
        return result

    except Exception as e:
        logger.error(f"Error processing dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.get("/datasets", response_model=DatasetsInfoResponse)
def get_datasets_info():
    """
    Get information about all the processed datasets.
    """
    try:
        result = dataset_service.get_datasets_info()
        return result
    except Exception as e:
        logger.error(f"Error getting datasets info: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get datasets info: {str(e)}"
        )


@app.get("/datasets/{dataset_name}", response_model=DatasetInfoResponse)
def get_dataset_info(dataset_name: str):
    """
    Get information about a dataset.
    """
    try:
        result = dataset_service.get_dataset_info(dataset_name)
        return result
    except Exception as e:
        logger.error(f"Error getting dataset info: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get dataset info: {str(e)}"
        )


@app.delete("/datasets/{dataset_name}/delete", response_model=DatasetDeleteResponse)
def delete_dataset(dataset_name: str):
    """
    Delete a dataset and all associated files.
    NOTE: This only deletes preprocessed dataset, NOT raw dataset files!
    """
    try:
        deleted_resources = []
        total_deleted_files = 0

        # Delete processed dataset (parquet files and metadata.json)
        processed_prefix = f"processed_datasets/{dataset_name}"
        processed_files = storage_manager.list_files(prefix=processed_prefix)
        if processed_files:
            deleted_count = storage_manager.delete_directory(processed_prefix)
            total_deleted_files += deleted_count
            deleted_resources.append(
                f"Processed dataset: {processed_prefix} ({deleted_count} files)"
            )

        if total_deleted_files > 0:
            message = f"Successfully deleted dataset '{dataset_name}' and all associated files"
            logger.info(message)
        else:
            message = f"Dataset '{dataset_name}' not found or already deleted"
            logger.warning(message)

        return DatasetDeleteResponse(
            dataset_name=dataset_name,
            deleted=total_deleted_files > 0,
            message=message,
            deleted_files_count=total_deleted_files,
            deleted_resources=deleted_resources,
        )

    except Exception as e:
        logger.error(f"Failed to delete dataset {dataset_name}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to delete dataset: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
