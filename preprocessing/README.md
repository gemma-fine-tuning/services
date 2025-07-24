# Preprocessing Service

FastAPI service for preprocessing datasets for Gemma fine-tuning. Handles uploading, processing, and storing datasets in Google Cloud Storage or the local file system. Supports both text and vision (multimodal) ChatML format conversion.

## Structure

- **`app.py`** – FastAPI application with endpoints
- **`services/`** – Core logic for dataset handling, loading, conversion
- **`storage/`** – Storage backends for GCS and local file system
- **`schema.py`** – Request/response models

## Deployment

The `cloudbuild.yaml` handles the build, push to artifact, and deploying / updating service.

```bash
cd preprocessing
gcloud builds submit --config cloudbuild.yaml --ignore-file=.gcloudignore
```

## Endpoints

### POST `/upload`

Upload a dataset file.

**Request:**
Upload a file using multipart/form-data. See API docs for details.

**Response:**

```json
{
  "dataset_name": "your_dataset_name",
  "status": "uploaded"
}
```

### POST `/process`

Start preprocessing job (supports text and vision datasets).

**Request:**

```json
{
  "dataset_name": "your_dataset_name",
  "config": {
    /* field mappings and options, see below */
  }
}
```

**Response:**

```json
{
  "status": "processing_started",
  "processed_dataset": "your_dataset_name_processed"
}
```

### GET `/datasets`

List all datasets.

**Response:**

```json
["dataset1", "dataset2"]
```

### GET `/datasets/{dataset_name}`

Get dataset information.

**Response:**

```json
{
  "dataset_name": "your_dataset_name",
  "num_rows": 1234,
  "columns": ["col1", "col2", ...],
  "info": { /* additional metadata */ }
}
```

### GET `/health`

Health check endpoint.

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

- Cloud Run service
- Environment: `GCS_DATA_BUCKET_NAME` required for GCS storage
- Port: 8080 (default)

## Environment Variables

- `GCS_DATA_BUCKET_NAME`: Google Cloud Storage bucket name (required for GCS storage)
- See `.env.example` for more options
