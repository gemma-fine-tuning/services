from flask import Flask, request, jsonify
import re
import json
import os
import uuid
import pandas as pd
from typing import Dict, List, Any
import logging
from google.cloud import storage
from datasets import load_dataset, Dataset
from werkzeug.utils import secure_filename

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize Google Cloud Storage client
storage_client = storage.Client()
DATA_BUCKET_NAME = os.environ.get("GCS_DATA_BUCKET_NAME", "gemma-dataset-dev")
ALLOWED_EXTENSIONS = {"txt", "csv", "json", "jsonl", "zip"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def upload_to_gcs(data, blob_name):
    """Upload data to Google Cloud Storage"""
    try:
        bucket = storage_client.bucket(DATA_BUCKET_NAME)
        blob = bucket.blob(blob_name)

        if isinstance(data, str):
            blob.upload_from_string(data, content_type="text/plain")
        else:
            blob.upload_from_string(json.dumps(data), content_type="application/json")

        return f"gs://{DATA_BUCKET_NAME}/{blob_name}"
    except Exception as e:
        logging.error(f"Error uploading to GCS: {str(e)}")
        raise


def download_from_gcs(blob_name):
    """Download data from Google Cloud Storage"""
    try:
        bucket = storage_client.bucket(DATA_BUCKET_NAME)
        blob = bucket.blob(blob_name)
        return blob.download_as_text()
    except Exception as e:
        logging.error(f"Error downloading from GCS: {str(e)}")
        raise


def load_standard_dataset(dataset_name, sample_size=None):
    """Load a standard dataset from Hugging Face"""
    try:
        # Load dataset from Hugging Face
        dataset = load_dataset(dataset_name, split="train")

        # Shuffle and sample if needed
        if sample_size and len(dataset) > sample_size:
            dataset = dataset.shuffle().select(range(sample_size))

        return list(dataset)

    except Exception as e:
        logging.error(f"Error loading standard dataset: {str(e)}")
        raise


def create_conversation_format(sample, format_config):
    """Convert dataset sample to conversation format with customizable field mappings"""

    # Extract configuration
    format_type = format_config.get("type", "default")

    if format_type == "default":
        # Keep original format - no conversion
        return sample

    # Custom format with field mappings
    system_message = format_config.get("system_message", "")
    user_prompt_template = format_config.get("user_prompt_template", "")
    include_system = format_config.get("include_system", False)
    field_mappings = format_config.get("field_mappings", {})

    # Build replacement dict from field mappings
    replacement_dict = {}
    for placeholder, dataset_field in field_mappings.items():
        replacement_dict[placeholder] = sample.get(dataset_field, "")

    # If no field mappings provided, use legacy fields for backward compatibility
    if not field_mappings:
        input_field = format_config.get("input_field", "input")
        output_field = format_config.get("output_field", "output")
        context_field = format_config.get("context_field", "context")

        # Try to infer common mappings
        replacement_dict = {
            "question": sample.get(input_field, ""),
            "query": sample.get(input_field, ""),
            "input": sample.get(input_field, ""),
            "context": sample.get(context_field, ""),
            "schema": sample.get(context_field, ""),
            "background": sample.get(context_field, ""),
        }

        # Add all sample fields as potential replacements
        replacement_dict.update(sample)

    # Format the user prompt using the replacement dictionary
    try:
        if user_prompt_template:
            user_content = user_prompt_template.format(**replacement_dict)
        else:
            # Default to just the input if no template provided
            user_content = replacement_dict.get(
                "question", replacement_dict.get("input", "")
            )
    except KeyError as e:
        # If a placeholder is missing, log warning and use the template as-is
        logging.warning(f"Missing field mapping for placeholder: {e}")
        user_content = user_prompt_template

    # Get the output field
    output_content = ""
    if field_mappings:
        # Try to find output in field mappings
        for placeholder, dataset_field in field_mappings.items():
            if placeholder.lower() in ["answer", "output", "response", "sql"]:
                output_content = sample.get(dataset_field, "")
                break

        # If no output mapping found, try common output fields
        if not output_content:
            output_content = sample.get(
                "sql", sample.get("output", sample.get("answer", ""))
            )
    else:
        # Legacy fallback
        output_field = format_config.get("output_field", "output")
        output_content = sample.get(output_field, "")

    # Build the conversation
    messages = []
    if include_system and system_message:
        messages.append({"role": "system", "content": system_message})

    messages.extend(
        [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output_content},
        ]
    )

    return {"messages": messages}


class DatasetProcessor:
    def __init__(self):
        self.whitespace_pattern = re.compile(r"\s+")

    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace by removing extra spaces"""
        return (
            self.whitespace_pattern.sub(" ", text).strip()
            if isinstance(text, str)
            else text
        )

    # NOTE: We don't necessarily need to format it here, we can create a formatting function and pass it to the SFTTrainer as well!
    def format_conversation(self, example: Dict, format_config: Dict) -> Dict:
        """Format a single example into conversation format"""
        if format_config.get("type") == "default":
            return example

        # Extract config
        system_message = format_config.get("system_message", "")
        user_template = format_config.get("user_prompt_template", "")
        include_system = format_config.get("include_system", False)
        field_mappings = format_config.get("field_mappings", {})

        # Build template variables
        template_vars = {
            placeholder: example.get(field, "")
            for placeholder, field in field_mappings.items()
        }

        # Format user message
        try:
            user_content = (
                user_template.format(**template_vars)
                if user_template
                else example.get("input", "")
            )
        except KeyError as e:
            logging.warning(f"Missing field mapping for {e}")
            user_content = user_template

        # Get assistant response
        output_field = next(
            (
                field
                for placeholder, field in field_mappings.items()
                if placeholder.lower() in ["answer", "output", "response", "sql"]
            ),
            None,
        )
        assistant_content = (
            example.get(output_field, "") if output_field else example.get("output", "")
        )

        # Build messages
        messages = []
        if include_system and system_message:
            messages.append({"role": "system", "content": system_message})
        messages.extend(
            [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]
        )

        return {"messages": messages}

    def process_dataset(self, raw_data: List[Dict], options: Dict[str, Any]) -> Dataset:
        """Process dataset using HuggingFace Datasets"""
        # Convert to Dataset
        dataset = Dataset.from_list(raw_data)

        # Apply format conversion if needed
        format_config = options.get("format_config", {})
        if format_config.get("type") != "default":
            dataset = dataset.map(
                lambda x: self.format_conversation(x, format_config),
                desc="Formatting conversations",
                batched=True,
            )

        # Clean text fields
        if options.get("normalize_whitespace", True):
            dataset = dataset.map(
                lambda x: {k: self.normalize_whitespace(v) for k, v in x.items()},
                desc="Normalizing whitespace",
            )

        return dataset


processor = DatasetProcessor()


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "preprocessing"}), 200


@app.route("/upload", methods=["POST"])
def upload_dataset():
    """
    Upload a dataset file to cloud storage
    Expected: multipart/form-data with 'file' field
    """
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if not allowed_file(file.filename):
            return jsonify(
                {"error": f"File type not allowed. Allowed: {ALLOWED_EXTENSIONS}"}
            ), 400

        # Generate unique filename
        file_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        blob_name = f"raw_datasets/{file_id}_{filename}"

        # Read file content
        file_content = file.read()

        # Upload to GCS
        bucket = storage_client.bucket(DATA_BUCKET_NAME)
        blob = bucket.blob(blob_name)
        blob.upload_from_string(file_content)

        return jsonify(
            {
                "dataset_id": file_id,
                "filename": filename,
                "gcs_path": f"gs://{DATA_BUCKET_NAME}/{blob_name}",
                "size_bytes": len(file_content),
            }
        ), 200

    except Exception as e:
        logging.error(f"Error uploading dataset: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route("/preprocess", methods=["POST"])
def start_preprocessing():
    """
    Start preprocessing a dataset
    Expected payload:
    {
        "dataset_source": "upload" | "standard",
        "dataset_id": "uuid-for-uploaded" | "philschmid/gretel-synthetic-text-to-sql",
        "sample_size": 5000,  # for standard datasets
        "options": {
            "format_config": {
                "type": "custom" | "default",
                "system_message": "You are a helpful assistant...",
                "user_prompt_template": "Question: {question}\nContext: {context}",
                "include_system": false,
                "field_mappings": {
                    "question": "sql_prompt",
                    "context": "sql_context",
                    "answer": "sql"
                }
            },
            "normalize_whitespace": true,
            "train_test_split": true,
            "test_size": 0.2
        }
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No data provided"}), 400

        dataset_source = data.get("dataset_source")
        dataset_id = data.get("dataset_id")
        options = data.get("options", {})

        if not dataset_source or not dataset_id:
            return jsonify({"error": "Missing dataset_source or dataset_id"}), 400

        # Load dataset
        if dataset_source == "upload":
            # Find uploaded file in GCS
            bucket = storage_client.bucket(DATA_BUCKET_NAME)
            blobs = list(bucket.list_blobs(prefix=f"raw_datasets/{dataset_id}_"))
            if not blobs:
                return jsonify({"error": "Uploaded dataset not found"}), 404

            blob = blobs[0]
            file_content = blob.download_as_text()

            # Parse file based on extension
            filename = blob.name.split("_", 1)[1]
            if filename.endswith(".csv"):
                import io

                df = pd.read_csv(io.StringIO(file_content))
                dataset = df.to_dict("records")
            elif filename.endswith(".json"):
                dataset = json.loads(file_content)
            elif filename.endswith(".jsonl"):
                dataset = [
                    json.loads(line) for line in file_content.strip().split("\n")
                ]
            else:
                return jsonify({"error": "Unsupported file format"}), 400

        elif dataset_source == "standard":
            sample_size = data.get("sample_size")
            dataset = load_standard_dataset(dataset_id, sample_size)
        else:
            return jsonify({"error": "Invalid dataset_source"}), 400

        # Convert raw data to Dataset and process
        processed_dataset = processor.process_dataset(dataset, options)

        # Generate unique ID for processed dataset
        processed_id = str(uuid.uuid4())

        # Handle train/test split
        if options.get("train_test_split", False):
            # Split dataset
            split = processed_dataset.train_test_split(
                test_size=options.get("test_size", 0.2), shuffle=True, seed=42
            )
            train_dataset = split["train"]
            test_dataset = split["test"]

            # Save splits to GCS
            train_blob_name = f"processed_datasets/{processed_id}_train.json"
            test_blob_name = f"processed_datasets/{processed_id}_test.json"

            upload_to_gcs(train_dataset.to_list(), train_blob_name)
            upload_to_gcs(test_dataset.to_list(), test_blob_name)

            return jsonify(
                {
                    "processed_dataset_id": processed_id,
                    "train_gcs_path": f"gs://{DATA_BUCKET_NAME}/{train_blob_name}",
                    "test_gcs_path": f"gs://{DATA_BUCKET_NAME}/{test_blob_name}",
                    "original_count": len(processed_dataset),
                    "train_count": len(train_dataset),
                    "test_count": len(test_dataset),
                    "sample_comparison": {
                        "original": dataset[0] if len(dataset) > 0 else None,
                        "processed": train_dataset[0]
                        if len(train_dataset) > 0
                        else None,
                    },
                }
            ), 200
        else:
            # Save full dataset as train
            processed_blob_name = f"processed_datasets/{processed_id}_train.json"
            upload_to_gcs(processed_dataset.to_list(), processed_blob_name)

            return jsonify(
                {
                    "processed_dataset_id": processed_id,
                    "gcs_path": f"gs://{DATA_BUCKET_NAME}/{processed_blob_name}",
                    "original_count": len(dataset),
                    "processed_count": len(processed_dataset),
                    "sample_comparison": {
                        "original": dataset[0] if len(dataset) > 0 else None,
                        "processed": processed_dataset[0]
                        if len(processed_dataset) > 0
                        else None,
                    },
                }
            ), 200

    except Exception as e:
        logging.error(f"Error preprocessing dataset: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route("/dataset/<dataset_id>", methods=["GET"])
def get_dataset_info(dataset_id):
    """
    Get information about a processed dataset

    Expected URL: /dataset/<dataset_id>
    """
    try:
        # dataset_id is now passed as a parameter
        blob_name = f"processed_datasets/{dataset_id}.json"

        # Check if dataset exists in GCS
        bucket = storage_client.bucket(DATA_BUCKET_NAME)
        blob = bucket.blob(blob_name)

        if not blob.exists():
            return jsonify({"error": "Dataset not found"}), 404

        # Get dataset metadata
        dataset_content = json.loads(blob.download_as_text())

        return jsonify(
            {
                "dataset_id": dataset_id,
                "gcs_path": f"gs://{DATA_BUCKET_NAME}/{blob_name}",
                "size": blob.size,
                "created": blob.time_created.isoformat(),
                "sample": dataset_content[:3]
                if len(dataset_content) > 3
                else dataset_content,
            }
        ), 200

    except Exception as e:
        logging.error(f"Error getting dataset info: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
