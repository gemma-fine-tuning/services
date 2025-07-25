# Inference Service

FastAPI service for running inference on fine-tuned Gemma models.

## Structure

- **`app.py`** - FastAPI application with endpoints
- **`inference_service.py`** - Core inference logic
- **`storage.py`** - Model loading from GCS/HuggingFace Hub
- **`schema.py`** - Request/response models

## Deployment

The `cloudbuild.yaml` handles the build, push to artifact, and deploying / updating service.

```bash
cd inference
gcloud builds submit --config cloudbuild.yaml --ignore-file=.gcloudignore
```

## Endpoints

### POST `/inference`

Single inference request.

**Request:**

```json
{
  "hf_token": "hf_...",
  "storage_type": "gcs" | "hfhub",
  "job_id_or_repo_id": "training_abc123_gemma-2b" | "user/repo",
  "prompt": "Your prompt here"
}
```

**Response:**

```json
{
  "result": "Generated text"
}
```

### POST `/batch_inference`

Batch inference for multiple prompts.

**Request:**

```json
{
  "hf_token": "hf_...",
  "storage_type": "gcs" | "hfhub",
  "job_id_or_repo_id": "training_abc123_gemma-2b" | "user/repo",
  "messages": [
    [
      {"role": "user", "content": "What is the capital of France?"}
    ]
  ]
}
```

**Response:**

```json
{
  "results": ["Generated text 1", "Generated text 2"]
}
```

### GET `/health`

Health check endpoint.

## Usage

1. **GCS Models**: Use `job_id` from training service
2. **HuggingFace Models**: Use `repo_id` (e.g., `"user/model-name"`)
3. **HF Token**: Required for Gemma models (gated access)

## Deployment

- Cloud Run service
- Environment: `PROJECT_ID` required
- Port: 8000 (default)
