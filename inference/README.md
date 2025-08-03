# Inference Service

FastAPI service for running inference on fine-tuned Gemma models.

## Structure

- **`app.py`** - FastAPI application with endpoints
- **`base.py`** - Core inference orchestration logic
- **`providers.py`** - Inference provider implementations (HuggingFace, Unsloth)
- **`storage.py`** - Model loading from GCS/HuggingFace Hub
- **`schema.py`** - Request/response models
- **`utils.py`** - Utility functions for modality detection and storage type inference

## Deployment

The `cloudbuild.yaml` handles the build, push to artifact, and deploying / updating service.

```bash
cd inference
gcloud builds submit --config cloudbuild.yaml --ignore-file=.gcloudignore
```

## Endpoints

### POST `/inference`

Single inference request. The service automatically detects storage type from the adapter path.

**Request:**

```json
{
  "hf_token": "hf_your_token_here",
  "adapter_path": "/path/to/adapter",
  "base_model_id": "google/gemma-3-2b-pt",
  "prompt": "Your input text here"
}
```

**Adapter Path Examples:**

- **GCS path**: `gs://bucket/trained_adapters/job_123/adapter`
- **HuggingFace Hub**: `username/model-name`

**Response:**

```json
{
  "result": "Generated text"
}
```

### POST `/batch_inference`

Batch inference for multiple conversations.

**Request:**

```json
{
  "hf_token": "hf_your_token_here",
  "adapter_path": "username/model-name",
  "base_model_id": "google/gemma-3-2b-pt",
  "messages": [
    [{ "role": "user", "content": "What is the capital of France?" }],
    [{ "role": "user", "content": "Explain quantum computing." }]
  ]
}
```

**Response:**

```json
{
  "results": ["Paris is the capital of France.", "Quantum computing is..."]
}
```

### GET `/health`

Health check endpoint.

**Response:**

```json
{
  "status": "healthy",
  "service": "inference"
}
```

## Key Features

1. **Automatic Storage Type Detection**: The service automatically detects whether the adapter path is local, GCS, or HuggingFace Hub
2. **Adapter and Merged Model Support**: Works with both LoRA adapters and fully merged models
3. **Framework Support**: Supports both HuggingFace Transformers and Unsloth inference
4. **Vision and Text Models**: Handles both text-only and vision models based on message content
5. **HF Token**: Required for Gemma models (gated access)

## Environment

- Cloud Run service
- Environment variable: `PROJECT_ID` required for Firestore
- Port: 8000 (default)
