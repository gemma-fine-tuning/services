import os
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from storage import GCSStorageManager, LocalStorageManager
from services.dataset_service import DatasetService
from schema import (
    DatasetUploadResponse,
    DatasetAnalysisResponse,
    PreprocessingRequest,
    ProcessingResult,
    DatasetInfoResponse,
    PreviewRequest,
    PreviewResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Gemma Dataset Preprocessing Service",
    version="2.0.0",
    description="A modular service for preprocessing datasets into ChatML format",
)

storage_type = os.getenv("STORAGE_TYPE", "local")  # "gcs" or "local"

if storage_type == "gcs":
    bucket_name = os.getenv("GCS_DATA_BUCKET_NAME", "gemma-dataset-dev")
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

        result = await dataset_service.upload_dataset(
            file_data=file_content,
            filename=file.filename,
            metadata={"content_type": file.content_type},
        )

        return result

    except Exception as e:
        logger.error(f"Error uploading dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/analyze", response_model=DatasetAnalysisResponse)
async def analyze_dataset(
    dataset_source: str, dataset_id: str, sample_size: int = None
):
    """Analyze a dataset and return metadata with suggested field mappings"""
    try:
        result = await dataset_service.analyze_dataset(
            dataset_source=dataset_source,
            dataset_id=dataset_id,
            sample_size=sample_size,
        )
        return result

    except Exception as e:
        logger.error(f"Error analyzing dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/preview", response_model=PreviewResponse)
async def preview_processing(request: PreviewRequest):
    """Preview how the dataset would look after processing

    The preview shows how the dataset would be converted to ChatML format using the provided configuration.
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
        result = await dataset_service.preview_processing(
            dataset_source=request.dataset_source,
            dataset_id=request.dataset_id,
            config=request.config,
            num_samples=request.sample_size,
        )
        return result

    except Exception as e:
        logger.error(f"Error previewing processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Preview failed: {str(e)}")


@app.post("/process", response_model=ProcessingResult)
async def process_dataset(request: PreprocessingRequest):
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
        result = await dataset_service.process_dataset(
            dataset_source=request.dataset_source,
            dataset_id=request.dataset_id,
            config=request.config,
        )
        return result

    except Exception as e:
        logger.error(f"Error processing dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.get("/dataset/{dataset_id}", response_model=DatasetInfoResponse)
async def get_dataset_info(dataset_id: str, dataset_type: str = None):
    """Get information about a dataset"""
    try:
        result = await dataset_service.get_dataset_info(dataset_id, dataset_type)
        return result

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset not found")
    except Exception as e:
        logger.error(f"Error getting dataset info: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get dataset info: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
