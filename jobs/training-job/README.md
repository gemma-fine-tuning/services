# Training Job Service

Cloud Run job that executes fine-tuning on Gemma models.

## Structure

- **`main.py`** - Job entry point, loads config and starts training
- **`training_service.py`** - Core training logic with provider support
- **`job_manager.py`** - Job state tracking and Firestore integration
- **`storage.py`** - Model and dataset saving/loading from GCS/HF Hub
- **`schema.py`** - Training configuration models

## Deployment

```bash
gcloud builds submit --config cloudbuild.yaml --ignore-file=.gcloudignore
```

- Cloud Run job with GPU support (L4)
- Memory: 16Gi, CPU: 4 cores
- GPU: 1x NVIDIA L4

## Execution Flow

1. **Start**: Job triggered by training service with `JOB_ID`
2. **Config**: Load training config from GCS bucket
3. **Login**: Authenticate with HuggingFace using token
4. **Train**: Execute fine-tuning with selected provider
5. **Track**: Update job status in Firestore
6. **Export**: Save model to GCS or push to HF Hub

## Training Methods

- **LoRA**: Low-rank adaptation (memory efficient)
- **QLoRA**: Quantized LoRA (4-bit precision)
- **Full**: Full fine-tuning (requires more memory)
- **RL**: Reinforcement learning fine-tuning

## Providers

- **unsloth**: Optimized training with Unsloth library
- **huggingface**: Standard HuggingFace training

## Configuration

**Environment Variables:**

- `JOB_ID` - Unique job identifier
- `PROJECT_ID` - GCP project for Firestore
- `GCS_CONFIG_BUCKET_NAME` - Bucket for training configs

**Training Config:**

```json
{
  "processed_dataset_id": "dataset_abc123",
  "hf_token": "hf_...",
  "training_config": {
    "method": "LoRA",
    "base_model_id": "google/gemma-2b",
    "learning_rate": 0.0001,
    "batch_size": 4,
    "epochs": 3
  },
  "export": "gcs"
}
```

## Monitoring

- **Weights & Biases**: Training metrics and model logging
- **Firestore**: Job status tracking
- **Cloud Logging**: Execution logs
