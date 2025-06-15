# Preprocessing Service

A FastAPI-based preprocessing service for Gemma fine-tuning that handles dataset upload, validation, and preprocessing with support for both domain and task adaptation.

## Architecture

This service follows a simplified 5-component architecture:

1. **DatasetHandler** - Dataset upload, loading, and validation
2. **PreprocessingEngine** - Text processing and formatting
3. **StorageManager** - Google Cloud Storage operations
4. **ConfigManager** - Configuration presets and management
5. **QualityMetrics** - Data quality assessment

## Migration from Flask

This service replaces the original `preprocessing/app.py` Flask application with:

- FastAPI for better async support and automatic API docs
- Modular component design for easier maintenance
- Enhanced error handling and logging
- Built-in configuration presets for common use cases

## API Endpoints

### Core Endpoints

- `GET /health` - Health check
- `POST /upload` - Upload dataset files
- `POST /preprocess` - Start preprocessing job
- `GET /dataset/{dataset_id}` - Get dataset information
- `GET /presets` - List available configuration presets

### Supported Formats

- JSON, JSONL, CSV files
- HuggingFace datasets
- Text files

## Configuration Presets

### Domain Adaptation

- `medical` - Medical domain with specialized terminology preservation
- `legal` - Legal domain with citation handling
- `financial` - Financial domain with currency normalization

### Task Adaptation

- `question_answering` - Q&A format with context support
- `text_classification` - Classification with label formatting
- `code_generation` - Code generation with structure preservation
- `summarization` - Document summarization format

## Deployment

### Local Development

```bash
pip install -r requirements.txt
uvicorn app:app --reload --port 8080
```

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

## Environment Variables

- `GCS_DATA_BUCKET_NAME` - Google Cloud Storage bucket name (default: "gemma-dataset-dev")
- `PORT` - Service port (default: 8080)

## Features

### Current Features (Migrated from Flask)

- ✅ Dataset upload to Google Cloud Storage
- ✅ HuggingFace dataset loading
- ✅ Text preprocessing and normalization
- ✅ Conversation format conversion
- ✅ Train/test dataset splitting
- ✅ Multiple file format support (JSON, JSONL, CSV)

### New Features

- ✅ Configuration presets for domain/task adaptation
- ✅ Modular component architecture
- ✅ Enhanced error handling
- ✅ API documentation (FastAPI auto-docs at `/docs`)

### Planned Features (Placeholders)

- 🔄 Advanced text cleaning pipeline
- 🔄 Data augmentation capabilities
- 🔄 Quality metrics and validation
- 🔄 Schema auto-detection
- 🔄 Preview functionality

## API Documentation

Once running, visit `http://localhost:8080/docs` for interactive API documentation.

## Migration Notes

This service maintains backward compatibility with the original Flask API while adding new capabilities. Key improvements:

1. **Better Structure** - Separated concerns into focused components
2. **Type Safety** - Pydantic models for request/response validation
3. **Async Support** - Better performance for I/O operations
4. **Configuration** - Built-in presets for common use cases
5. **Extensibility** - Easy to add new preprocessing features

## Development

To add new preprocessing features:

1. Add methods to `PreprocessingEngine`
2. Update configuration presets in `config.py`
3. Add new API endpoints in `app.py`
4. Update Pydantic models in `models.py`

The modular design makes it easy to extend functionality while maintaining clean separation of concerns.
