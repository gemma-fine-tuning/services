import os
import json
import sys
from google.cloud import firestore
from google.cloud import storage
from training_service import TrainingService
from job_manager import JobStateManager, JobTracker, JobMetadata, JobStatus
from schema import TrainRequest
from huggingface_hub import login
import logging
from datetime import datetime, timezone

GCS_CONFIG_BUCKET = os.getenv("GCS_CONFIG_BUCKET", "gemma-train-config")


def load_training_config(job_id):
    gcs_path = os.getenv("TRAINING_CONFIG_GCS_PATH")
    if gcs_path:
        # Download config from GCS path specified in env var
        try:
            client = storage.Client()
            if gcs_path.startswith("gs://"):
                bucket_name, blob_name = gcs_path[5:].split("/", 1)
            else:
                raise ValueError("GCS path must start with gs://")
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            config_json = blob.download_as_text()
            return config_json
        except Exception as e:
            print(f"Error downloading config from GCS: {e}")
            sys.exit(1)
    else:
        # Default to gs://{GCS_CONFIG_BUCKET}/{JOB_ID}.json
        default_gcs_path = f"gs://{GCS_CONFIG_BUCKET}/{job_id}.json"
        try:
            client = storage.Client()
            bucket_name, blob_name = default_gcs_path[5:].split("/", 1)
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            if blob.exists():
                config_json = blob.download_as_text()
                return config_json
        except Exception as e:
            print(f"Error downloading default config from GCS: {e}")
            sys.exit(1)


def main():
    job_id = os.getenv("JOB_ID")
    if not job_id:
        print("Error: JOB_ID environment variable is required")
        sys.exit(1)

    train_config_json = load_training_config(job_id)

    if train_config_json:
        train_request = TrainRequest.model_validate(json.loads(train_config_json))
    else:
        raise ValueError("Training config not found")

    # Construct JobMetadata for Firestore
    now = datetime.now(timezone.utc)
    job_metadata = JobMetadata(
        job_id=job_id,
        status=JobStatus.QUEUED,
        created_at=now,
        updated_at=now,
        processed_dataset_id=train_request.processed_dataset_id,
        base_model_id=train_request.training_config.base_model_id,
    )

    if train_request.hf_token:
        login(token=train_request.hf_token)
    else:
        raise ValueError("Huggingface token is required for training")

    project_id = os.environ.get("PROJECT_ID")
    # NOTE: Project ID should always be provided else will fail
    if not project_id:
        raise ValueError(
            "PROJECT_ID environment variable must be set for Firestore client"
        )
    db = firestore.Client(project=project_id)
    job_manager = JobStateManager(db)

    try:
        # Use JobTracker context manager for automatic status transitions
        with JobTracker(job_manager, job_id, job_metadata) as tracker:
            # Execute training with job tracker (WandB and status handled in training service)
            train_req = TrainRequest.model_validate(train_request)
            service = TrainingService.from_provider(train_req.training_config.provider)
            result = service.run_training(train_req, job_tracker=tracker)

            # Mark as completed
            # We no longer need a training response because we can just save the result to firestore
            tracker.completed(result["adapter_path"], result["base_model_id"])

    except Exception as e:
        logging.error(f"Training job {job_id} failed: {str(e)}", exc_info=True)
        # Exception handling is done by JobTracker context manager


if __name__ == "__main__":
    main()
