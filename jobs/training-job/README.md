# Training Job Service

Cloud Run job that executes fine-tuning on Gemma models.

## Structure

- **`main.py`** - Job entry point, loads config and starts training
- **`base.py`** - Base class for training jobs with common workflow / functionality
- **`providers.py`** - Core training logic with provider support
- **`job_manager.py`** - Job state tracking and Firestore integration
- **`storage.py`** - Model and dataset saving/loading from GCS/HF Hub
- **`schema.py`** - Training configuration models

## Deployment

```bash
gcloud builds submit --config cloudbuild.yaml --ignore-file=.gcloudignore
```

- Cloud Run job with GPU support (L4)
- Memory: 32Gi, CPU: 8 cores
- GPU: 1x NVIDIA L4

## Execution Flow

1. **Start**: Job triggered by training service with `JOB_ID`
2. **Config**: Load training config from GCS bucket
3. **Login**: Authenticate with HuggingFace using token
4. **Train**: Execute fine-tuning with selected provider
5. **Evaluate**: Run evaluation if configured
6. **Track**: Update job status in Firestore
7. **Export**: Save model to GCS or push to HF Hub

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

> NOTE: This is completely identically shared with `TrainRequest` and `TrainConfig` from `training/README.md`

```json
{
  "processed_dataset_id": "dataset_abc123",
  "hf_token": "hf_...",
  "training_config": {
    "method": "LoRA",
    "base_model_id": "google/gemma-2b",
    "learning_rate": 0.0001,
    "batch_size": 4,
    "epochs": 3,
    "eval_strategy": "epoch",
    "evaluation_metrics": ["accuracy", "perplexity"]
  },
  "export_config": {
    "format": "merged",
    "quantization": "fp16",
    "destination": "gcs"
  }
}
```

## Monitoring

- **Weights & Biases**: Training metrics and model logging
- **Firestore**: Job status tracking
- **Cloud Logging**: Execution logs

## Export Formats

### Adapter Export

- **Format**: `adapter`
- **Quantization**: Not applicable
- **Use case**: Lightweight LoRA/QLoRA adapter weights only
- **Storage**: ~MB in size, fast upload/download

### Merged Export

- **Format**: `merged`
- **Quantization**: `none`, `fp16`, `q4` (4-bit)
- **Use case**: Full standalone model for deployment
- **Providers**:
  - Unsloth: `merged_16bit` (vLLM compatible), `merged_4bit`
  - HuggingFace: Standard merge with PEFT, supports FP16 quantization

### GGUF Export

- **Format**: `gguf`
- **Quantization**: `f16`, `q8_0`, `q4_k_m`, `q5_k_m`, `q2_k`
- **Use case**: CPU-optimized inference (llama.cpp, Ollama)
- **Providers**:
  - Unsloth: Native GGUF export with quantization
  - HuggingFace: Not implemented (requires llama.cpp conversion)

## Evaluation

Configure evaluation during training:

- **eval_strategy**: When to run evaluation (`no`, `steps`, `epoch`)
- **eval_steps**: Evaluation frequency when using `steps` strategy
- **evaluation_metrics**: Metrics to compute (`accuracy`, `perplexity`)
- **batch_eval_metrics**: Enable batch evaluation mode for better performance

Supported metrics:

- **Accuracy**: Token-level classification accuracy
- **Perplexity**: Model confidence metric (2^loss)
- **Loss**: Training/validation loss

> If you want more complicated metrics, currently they are only supported in the inference service, you can use the `/evaluate` endpoint there.

## Known Issues

- GGUF export is currently not supported by HuggingFace but you can do it manually using `llama.cpp` or `ollama` tools.

- GGUF export for Unsloth is being fixed by the unsloth team...

- Evaluation during training is not supported for Unsloth, this might be worked on by unsloth team (use eval in inference service instead)

- Evaluation metrics might not work on hugging face due to OOM depending on your config...
