# Preprocessing Service

FastAPI service for preprocessing datasets for Gemma fine-tuning. Handles uploading, processing, and storing datasets in Google Cloud Storage or the local file system.

> This service chooses to use conversational format with the ChatML variation. It covers multiple use cases by supporting different types.

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

- Cloud Run service (CPU only)
- Environment: `GCS_DATA_BUCKET_NAME` required for GCS storage
- Port: 8080 (default)

## Endpoints

### POST `/datasets/upload`

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

### POST `/datasets/process`

Start preprocessing job (supports text and vision datasets).

**Request:**

```json
{
  "dataset_name": "your_dataset_name",
  "dataset_source": "huggingface",
  "dataset_id": "org/repo",
  "dataset_subset": "default",
  "processing_mode": "language_modeling" | "prompt_only",
  "config": {
    /* field mappings and options, see below */
  }
}
```

**Response:**

```json
{
  "dataset_name": "your_dataset_name"
  // OTHER METADATA FIELDS NOT USED BY THE FRONTEND
  // the frontend redirects to the `/datasets/{processed_dataset_id}` endpoint directly
}
```

### GET `/datasets`

List all datasets.

**Response:**

```json
["dataset1", "dataset2"]
```

### GET `/datasets/{processed_dataset_id}`

Get dataset information.

**Response:**

```json
{
  "dataset_name": "your_dataset_name",
  "processed_dataset_id": "uuid",
  "num_rows": 1234,
  "columns": ["col1", "col2", ...],
  "info": {
    "dataset_subset": "default",
    "dataset_source": "upload",
    "dataset_id": "abc123",
    "created_at": "2025-07-25T12:34:56",
    "modality": "text", // or "vision"
    "splits": [
      {
        "split_name": "train",
        "num_rows": 1000,
        "path": "processed_datasets/your_dataset_name/train.parquet",
        "samples": [
          // For vision datasets:
          {
            "messages": [
              { "role": "system", "content": "Describe the images." },
              {
                "role": "user",
                "content": [
                  { "type": "text", "text": "Compare these images." },
                  { "type": "image", "image": "data:image/png;base64,..." },
                  { "type": "image", "image": "data:image/png;base64,..." }
                ]
              },
              { "role": "assistant", "content": "The first image shows..." }
            ]
          }
          // up to five samples per split...
        ]
      },
      // ... more splits (e.g., "test")
    ]
  }
}
```

> NOTE: Modality is returned when you fetch info for a dataset and it is determined by the service and saved to metadata during processing. It is NOT set by the user.

> Samples dict is structured according to the ChatML format and the specific type of the dataset (as well as modality). See the [Conversion Examples](#conversion-examples) section for details.

### DELETE `/datasets/{processed_dataset_id}`

Delete a dataset and all associated files.

**Response:**

```json
{
  "dataset_name": "your_dataset_name",
  "deleted": true,
  "message": "Dataset and all associated files deleted successfully.",
  "deleted_files_count": 3,
  "deleted_resources": [
    "processed_datasets/your_dataset_name_processed/train.parquet",
    "processed_datasets/your_dataset_name_processed/test.parquet",
    "raw_datasets/your_dataset_name.csv"
  ]
}
```

### GET `/health`

Health check endpoint.

## Conversion Examples

The only format used in this project is conversation format as opposed to standard format defined in TRL docs. The specific variation of this format used in ChatML. The reason is that this format supports both text-only and multimodal (vision) conversations.

We have standardized to use list format (containing `type`) in `content` field for consistency between vision and text datasets, **even if the dataset is text-only**. The handler is backward compatible but this format is always preferred for consistency.

### Supported Formats & Modes

This service can ingest standard tabular formats (JSON, JSONL, CSV, Parquet, Excel) and HuggingFace datasets. It then preprocesses them into specific conversational formats tailored for different fine-tuning tasks. The desired output is controlled via a `processing_mode` parameter in the process request.

- **Language Modeling (`language_modeling`)**

  - **Use Case**: Continued pre-training or domain adaptation using `SFTTrainer`.
  - **Output Format**: The source text is mapped to the `assistant` role in a single-turn conversation. This presents the text as a valid completion for the model to learn from without a direct prompt.

- **Prompt-Only (`prompt_only`)**
  - **Use Case**: Policy optimization with trainers that require only a prompt, such as `GRPOTrainer`. The trainer uses the prompt to generate its own responses for optimization.
  - **Output Format**: The source prompt is mapped to the `user` role in a single-turn conversation.

Examples are shown below.

Planned:

- **Instruction Tuning (`prompt_completion`)**

  - **Use Case**: Standard supervised fine-tuning with `SFTTrainer` to teach the model to follow instructions.
  - **Source Data**: Two columns for `prompt` and `completion`.
  - **Output Format**: A standard two-turn conversation.
  - **Example**: `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`

- **Preference Tuning (`preference`)**
  - **Use Case**: Alignment with preference-based algorithms like DPO using `DPOTrainer`.
  - **Source Data**: Three columns: `prompt`, `chosen` (the preferred response), and `rejected` (the less-preferred response).
  - **Output Format**: This mode will produce a dataset with three distinct columns (`prompt`, `chosen`, `rejected`), as this specific structure is required by the `DPOTrainer`.

### Language Modelling

#### Text Example

```json
{
  "messages": [
    {
      "role": "system",
      "content": [{ "type": "text", "text": "You are a helpful assistant." }]
    },
    {
      "role": "user",
      "content": [{ "type": "text", "text": "What is deep learning?" }]
    },
    {
      "role": "assistant",
      "content": [
        {
          "type": "text",
          "text": "Deep learning is a subset of machine learning..."
        }
      ]
    }
  ]
}
```

#### Vision Example

```json
{
  "messages": [
    {
      "role": "system",
      "content": [{ "type": "text", "text": "Describe the images." }]
    },
    {
      "role": "user",
      "content": [
        { "type": "text", "text": "Compare these images." },
        { "type": "image", "image": "<base64 or image object>" },
        { "type": "image", "image": "<base64 or image object>" }
      ]
    },
    {
      "role": "assistant",
      "content": [{ "type": "text", "text": "The first image shows..." }]
    }
  ]
}
```

> **Note:** Images are always included in the user message as a list of content items. See below for vision configuration.

### Prompt-Only

```json
{
  "prompt": [
    {
      "role": "system",
      "content": [{ "type": "text", "content": "REASONING_PROMPT" }]
    },
    {
      "role": "user",
      "content": [
        { "type": "text", "content": "What color is the sky?" },
        { "type": "image", "image": "<base64 or image object>" }
      ]
    }
  ],
  "answer": "10",
  "reasoning": "<think>Some math logic</think>",
  "any_other_additional_fields": "additional_info"
}
```

## Conversion Configuration

### Language Modeling (Vision)

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

### Prompt-only (Text)

```json
{
  "config": {
    "vision_enabled": true,
    "field_mappings": {
      "system_field": {
        "type": "template",
        "value": "This is how you should reason... put stuff in this tag... put answer in..."
      },
      "user_field": {
        "type": "template",
        "value": "This is the user's prompt {prompt}."
      },
      "answer": {
        "type": "column",
        "value": "answer"
      },
      "reasoning": {
        "type": "column",
        "value": "reasoning"
      },
      "any_other_additional_fields": {
        "type": "column",
        "value": "additional_info"
      }
    }
  }
}
```

- `user_field` and `system_field` work the same way as in language modeling, but the `assistant_field` is not used.

## Metadata Management

The preprocessing service uses a hybrid storage approach:

- **Dataset Files**: Stored in Google Cloud Storage (or local filesystem) as parquet files
- **Metadata**: Centrally managed in Firestore for consistency and user ownership tracking

Each preprocessed dataset is identified by a unique 8-char UUID based identifier, this field is called `processed_dataset_id`. It is used as the ID for the document and firestore and the folder in GCS. This is different from `dataset_id` which refers to the ID of the **source** dataset, e.g. ID at hugging face or uploaded files.
