import os
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from dataset_handler import DatasetHandler
from preprocessing.data_preprocessor import PreprocessingEngine
from storage_manager import StorageManager
from config import ConfigManager
from models import (
    DatasetUploadResponse,
    PreprocessingRequest,
    ProcessingResult,
    DatasetInfoResponse,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Gemma Preprocessing Service", version="1.0.0")

# Initialize components
storage_manager = StorageManager(
    bucket_name=os.getenv("GCS_DATA_BUCKET_NAME", "gemma-dataset-dev")
)
dataset_handler = DatasetHandler(storage_manager)
preprocessing_engine = PreprocessingEngine(storage_manager)
config_manager = ConfigManager()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "preprocessing"}


@app.post("/upload", response_model=DatasetUploadResponse)
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload a dataset file to cloud storage
    """
    try:
        # Read file content
        file_content = await file.read()

        # Upload using dataset handler
        result = await dataset_handler.upload_dataset(
            file_data=file_content, filename=file.filename, metadata={}
        )

        return result

    except Exception as e:
        logger.error(f"Error uploading dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/preprocess", response_model=ProcessingResult)
async def start_preprocessing(request: PreprocessingRequest):
    """
    Start preprocessing a dataset
    """
    try:
        # Load dataset
        dataset = await dataset_handler.load_dataset(
            dataset_source=request.dataset_source,
            dataset_id=request.dataset_id,
            sample_size=request.sample_size,
        )

        # Process dataset
        result = await preprocessing_engine.process_dataset(
            dataset=dataset, config=request.options
        )

        return result

    except Exception as e:
        logger.error(f"Error preprocessing dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/dataset/{dataset_id}", response_model=DatasetInfoResponse)
async def get_dataset_info(dataset_id: str):
    """
    Get information about a processed dataset
    """
    try:
        result = await dataset_handler.get_dataset_info(dataset_id)
        return result

    except Exception as e:
        logger.error(f"Error getting dataset info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/presets")
async def list_presets():
    """List available preprocessing presets"""
    return config_manager.get_available_presets()


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
