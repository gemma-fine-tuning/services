# Preprocessing Service

This service handles data preprocessing for the Gemma Fine-Tuning Platform. It provides dataset upload, preprocessing, and storage capabilities using Google Cloud Storage. Designed to run on Google Cloud Run as a service using CPU resources.

## Features

- Dataset upload to Google Cloud Storage
- Standard dataset loading from Hugging Face Hub
- Format conversion (with templates or custom mapping)
- Preprocessing options (e.g., removing HTML, URLs, normalizing text)
- Train/test dataset splitting
- Cloud storage integration

## API Endpoints

### Health Check

- `GET /health` - Health check endpoint

### Dataset Management

- `POST /upload` - Upload dataset files to cloud storage
- `POST /preprocess` - Start preprocessing workflow
- `GET /dataset/<dataset_id>` - Get processed dataset information

## Usage Examples

### Upload Dataset

```bash
curl -X POST http://localhost:8080/upload \
  -F "file=@my_dataset.json"
```

### Start Preprocessing (Uploaded Dataset)

```bash
curl -X POST http://localhost:8080/preprocess \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_source": "upload",
    "dataset_id": "uuid-from-upload-response",
    "options": {
      "remove_html": true,
      "remove_urls": true,
      "to_lowercase": true,
      "normalize_unicode": true
    }
  }'
```

### Start Preprocessing (Standard Dataset)

```bash
curl -X POST http://localhost:8080/preprocess \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_source": "standard",
    "dataset_id": "squad",
    "sample_size": 5000,
    "options": {
      "normalize_unicode": true,
      "to_lowercase": false
    }
  }'
```

### Get Dataset Info

```bash
curl http://localhost:8080/dataset/processed-dataset-uuid
```

## Deployment

### Local Development

```bash
pip install -r requirements.txt
python app.py
```

### Docker Build

```bash
docker build -t preprocessing-service .
docker run -p 8080:8080 preprocessing-service
```

### Google Cloud Run Deployment

Set the `PROJECT_ID` and `DATA_BUCKET_NAME` environment variables first!

Also make sure `GOOGLE_APPLICATION_CREDENTIALS` is a path to service account JSON file (otherwise do `gcloud auth login` to use your account).

```bash
# Build Docker image for the correct platform (amd64/linux)
docker build --platform linux/amd64 -t us-central1-docker.pkg.dev/$PROJECT_ID/gemma-fine-tuning/preprocessing-service .

# Push to Artifact Registry
docker push us-central1-docker.pkg.dev/$PROJECT_ID/gemma-fine-tuning/preprocessing-service

# To build on cloud
gcloud builds submit --tag us-central1-docker.pkg.dev/$PROJECT_ID/gemma-fine-tuning/preprocessing-service .

# Deploy to Cloud Run
gcloud run deploy preprocessing-service \
  --image us-central1-docker.pkg.dev/$PROJECT_ID/gemma-fine-tuning/preprocessing-service \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1 \
  --timeout 3600 \
  --set-env-vars GCS_DATA_BUCKET_NAME=$DATA_BUCKET_NAME
```

## Configuration Options

The preprocessing service supports the following processing options:

We take a prompt and reformats the dataset into a converstaion format specified by your prompt, and then the tokenizer will apply the ChatML format when using the SFT Trainer.

### Format Configuration (`format_config`)

- `type`: Format type ("custom", or "default")
- `system_message`: Custom system message for conversation format
- `user_prompt_template`: Template for user prompts (supports {field} placeholders)
- `include_system`: Whether to include system message in conversation

Field mappings:

- `input_field`: Field name for input text (default: "input")
- `output_field`: Field name for output text (default: "output")
- `context_field`: Field name for context text (default: "context")

### General Options

- `normalize_whitespace`: Normalize whitespace (default: true)
- `train_test_split`: Split dataset into train/test sets (default: false)
- `test_size`: Proportion of test set when splitting (default: 0.2)

## Workflow

1. **Upload Dataset**: User uploads dataset file via `/upload` endpoint
2. **Configure Preprocessing**: Frontend sends preprocessing configuration
3. **Process Dataset**: Service downloads/loads dataset, applies preprocessing, saves to GCS
4. **Validation**: Service validates processed dataset and returns metrics
5. **Storage**: Processed dataset is stored in GCS for training service access

## TODO

- Once we have the account and dashboard things setup, the user shoudl be able to reuse one of the existing datasets either from the preprocess or upload stage!

- Maybe we don't save both the raw and preprocessed dataset just the latest one?

- Also avoid doing so many health checks whenever the page reloads, in real deployment this would make the uptime for the service much longer and result in more costs.

- Validation for "default" format type to ensure required fields are present.
