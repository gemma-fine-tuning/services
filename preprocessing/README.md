# Preprocessing Service

FastAPI based preprocessing service for [Gemma fine-tuning](https://github.com/gemma-fine-tuning/). It handles uploading local datasets, preprocessing them and storing them in Google Cloud Storage / Local file system.

## Architecture

This service follows a simplified 5-component architecture:

1. **DatasetHandler** - Uploads raw and processed datasets with the help of storage service
2. **DatasetLoader** - Loads datasets from local file system or HuggingFace datasets into a `DatasetDict` object
3. **FormatConverter** - Converts datasets to ChatML format
4. **Storage Service** - Handles storage of datasets in Google Cloud Storage / Local File System
    1. **GCSStorageManager** - Handles storage of datasets in Google Cloud Storage
    2. **LocalStorageManager** - Handles storage of datasets in Local File System
5. **DatasetService** - Utilizes the above components to handle dataset upload, processing and storage. This is the main entry point for the API.

## API Endpoints

- `GET /health` - Health check
- `POST /upload` - Upload dataset files
- `POST /process` - Start preprocessing job
- `GET /datasets` - List all datasets
- `GET /datasets/{dataset_name}` - Get dataset information

## Supported Formats

- JSON, JSONL, CSV, Parquet and Excel files
- HuggingFace datasets

## Deployment

### Cloud Deployment

> IMPORTANT: This has been updated so pls use this command below:

```bash
gcloud builds submit --config cloudbuild.yaml --ignore-file=.gcloudignore
```

This command will handle building, pushing, and deploying Cloud Run service.

### Local Development

This application uses [`uv`](https://docs.astral.sh/uv/) for dependency management instead of `pip`, hence, the following commands use `uv` instead of `pip`. However, you can use `pip` if you prefer.

```bash
uv run uvicorn app:app
```

Add `--reload` if you are developing and want to automatically reload the server on code changes. (Not recommended for production)

### Docker

```bash
docker build -t preprocessing-service .
docker run -p 8080:8080 -e GCS_DATA_BUCKET_NAME=your-bucket preprocessing-service
```

### Cloud Run

```bash
gcloud run deploy preprocessing-service \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GCS_DATA_BUCKET_NAME=your-bucket
```

> **Note**: We recommend using the `us-central1` region because GPU support is available in this region. So it'll be required for the fine-tuning service to run. Having both the services and the data bucket in the same region will help in reducing the latency.

## Environment Variables

> Checkout the [`.env.example`](../.env.example) file for the list of environment variables.

## Features

### Current Features

- ✅ Dataset upload to Google Cloud Storage / Local File System
- ✅ HuggingFace dataset loading
- ✅ Conversation format conversion
- ✅ Train/test dataset splitting
- ✅ Multiple file format support (JSON, JSONL, CSV, Parquet, Excel)
- ✅ Storage of processed datasets in Google Cloud Storage / Local File System
- ✅ Getting processed datasets' information
- ✅ Data augmentation capabilities

### Planned Features (Placeholders)

- 🔄 PDF file processing
- 🔄 Processing datasets for different fine-tuning tasks

## API Documentation

Once running, visit `http://localhost:8080/docs` for interactive API documentation.

## Development

To add new preprocessing features:

1. Add methods to [`DatasetService`](./services/dataset_service.py)
2. Add new API endpoints in [`app.py`](./app.py)
3. Update Pydantic models in [`schema.py`](./schema.py)

The modular design makes it easy to extend functionality while maintaining clean separation of concerns.
