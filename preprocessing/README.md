# Preprocessing Service

FastAPI-based preprocessing service for [Gemma fine-tuning](https://github.com/gemma-fine-tuning/). Handles uploading, processing, and storing datasets in Google Cloud Storage or the local file system. Supports both text and vision (multimodal) ChatML format conversion.

## Architecture

This service uses a modular architecture for scalable and flexible preprocessing:

1. **DatasetHandler** - Uploads raw and processed datasets with the help of storage service
2. **DatasetLoader** - Loads datasets from local file system or HuggingFace datasets into a `DatasetDict` object
3. **FormatConverter** - Converts datasets to ChatML format (automatically detects and handles vision datasets)
4. **Storage Service** - Handles storage of datasets in Google Cloud Storage / Local File System
   - **GCSStorageManager** - Handles storage of datasets in Google Cloud Storage
   - **LocalStorageManager** - Handles storage of datasets in Local File System
5. **DatasetService** - Main entry point for API, orchestrates upload, processing, and storage

## API Endpoints

- `GET /health` - Health check
- `POST /upload` - Upload dataset files
- `POST /process` - Start preprocessing job (supports vision datasets)
- `GET /datasets` - List all datasets
- `GET /datasets/{dataset_name}` - Get dataset information

## Supported Formats

- JSON, JSONL, CSV, Parquet, Excel files
- HuggingFace datasets
- **Vision datasets** with image columns (JPEG, PNG, BMP, GIF, TIFF, WebP)

> **Note:** Uploading custom vision datasets is not currently supported, but you can use existing HuggingFace multimodal datasets.

## ChatML Format

Datasets are converted to the standardized ChatML format for conversational AI training. This format supports both text-only and multimodal (vision) conversations.

### Text ChatML Example

```json
{
  "messages": [
    { "role": "system", "content": "You are a helpful assistant." },
    { "role": "user", "content": "What is machine learning?" },
    { "role": "assistant", "content": "Machine learning is a field of AI..." }
  ]
}
```

### Vision ChatML Example

```json
{
  "messages": [
    { "role": "system", "content": "Describe the images." },
    {
      "role": "user",
      "content": [
        { "type": "text", "text": "Compare these images." },
        { "type": "image", "image": "<base64 or image object>" },
        { "type": "image", "image": "<base64 or image object>" }
      ]
    },
    { "role": "assistant", "content": "The first image shows..." }
  ]
}
```

> **Note:** Images are always included in the user message as a list of content items. See below for vision configuration.

## Vision Configuration

Vision processing is automatically enabled when image field mappings are detected. Simply add image field mappings to your configuration (you can include multiple images in a single user message).

```json
{
  "config": {
    "vision_enabled": true,
    "field_mappings": {
      "user_field": {
        "type": "template",
        "value": "Compare these images and tell me the differences."
      },
      "assistant_field": {
        "type": "column",
        "value": "comparison"
      },
      "image_field_1": {
        "type": "image",
        "value": "image1"
      },
      "image_field_2": {
        "type": "image",
        "value": "image2"
      },
      "image_field_3": {
        "type": "image",
        "value": "image3"
      }
    }
  }
}
```

- Images are **always added to user messages only**
- Images are processed in the order they appear in the field_mappings
- Supported image formats: PIL Image objects, base64 strings, file paths, HuggingFace dataset format with `bytes` field

## Deployment

### Cloud Deployment

Use the following command to build, push, and deploy the Cloud Run service:

```bash
gcloud builds submit --config cloudbuild.yaml --ignore-file=.gcloudignore
```

### Local Development

This application uses [`uv`](https://docs.astral.sh/uv/) for dependency management. You can also use `pip` if preferred.

```bash
uv run uvicorn app:app --port 8080
```

Add `--reload` for development mode to auto-reload on code changes.

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

> **Note:** The recommended region is `us-central1` for GPU support and lower latency with the fine-tuning service and data bucket.

## Environment Variables

- `GCS_DATA_BUCKET_NAME`: Google Cloud Storage bucket name (required for GCS storage)
- See `.env.example` for more options

## Features

### Current Features

- ✅ Dataset upload to Google Cloud Storage / Local File System
- ✅ HuggingFace dataset loading
- ✅ ChatML format conversion (text and vision)
- ✅ Train/test dataset splitting
- ✅ Multiple file format support (JSON, JSONL, CSV, Parquet, Excel)
- ✅ Storage of processed datasets in Google Cloud Storage / Local File System
- ✅ Getting processed datasets' information
- ✅ Data augmentation capabilities

### Planned Features

- 🔄 PDF file processing
- 🔄 Processing datasets for different fine-tuning tasks

## API Documentation

Once running, visit `http://localhost:8080/docs` for interactive API documentation.

## How It Works

1. **Dataset Upload**: Upload raw datasets via API or HuggingFace datasets.
2. **Format Conversion**: Converts datasets to ChatML format (text or vision) using configurable field mappings.
3. **Vision Handling**: Automatically detects image fields and processes images for multimodal ChatML output.
4. **Storage**: Stores processed datasets in GCS or local file system.
5. **Dataset Info**: Query processed datasets and their metadata via API.

## Note on Vision Datasets

- Dataset must contain at least one image column
- Images can be in various formats: PIL Image objects, base64 encoded strings, file paths, HuggingFace dataset format with `bytes` field
- Example: `unsloth/LaTeX_OCR`

## Development

To add new preprocessing features:

1. Add methods to [`DatasetService`](./services/dataset_service.py)
2. Add new API endpoints in [`app.py`](./app.py)
3. Update Pydantic models in [`schema.py`](./schema.py)

The modular design makes it easy to extend functionality while maintaining clean separation of concerns.
