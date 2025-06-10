import os
import json
import logging
import torch
from flask import Flask, request, jsonify
from google.cloud import storage
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, SFTConfig
import shutil

# from metrics_logger import TrainingMetricsLogger
from huggingface_hub import login

app = Flask(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",  # Add timestamps to logs
)

hf_token = os.environ.get("HUGGINGFACE_TOKEN")
if hf_token:
    login(token=hf_token)
    logging.info("Logged into Hugging Face")
else:
    logging.warning("HUGGINGFACE_TOKEN not provided. Hugging Face login skipped.")

# Initialize Google Cloud Storage client
storage_client = storage.Client()
DATA_BUCKET_NAME = os.environ.get("GCS_DATA_BUCKET_NAME", "gemma-dataset-dev")
EXPORT_BUCKET_NAME = os.environ.get("GCS_EXPORT_BUCKET_NAME", "gemma-export-dev")

# Training metrics will be available via REST API endpoints
logging.info("✅ Training service ready - metrics via REST API")


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "training"}), 200


@app.route("/train", methods=["POST"])
def start_training():
    """
    Start training a Gemma model with LoRA
    Expected payload:
    {
        "processed_dataset_id": "uuid-of-processed-dataset",
        "model_config": {
            "model_id": "google/gemma-3-1b-pt",
            "method": "LoRA",
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "learning_rate": 2e-4,
            "batch_size": 1,
            "epochs": 3,
            "max_seq_length": 512,
            "gradient_accumulation_steps": 4
        }
    }
    """
    try:
        data = request.get_json()
        processed_dataset_id = data.get("processed_dataset_id")
        model_config = data.get("model_config", {})

        if not processed_dataset_id:
            return jsonify({"error": "processed_dataset_id is required"}), 400

        # Validate model_id
        model_id = model_config.get(
            "model_id", "google/gemma-3-1b-pt"
        )  # Changed default to pretrained
        supported_models = [
            "google/gemma-3-1b-pt",  # Changed to pretrained models
            "google/gemma-3-4b-pt",
            "google/gemma-3-12b-pt",
            "google/gemma-3-27b-pt",
        ]

        if model_id not in supported_models:
            return jsonify(
                {"error": f"Unsupported model. Select from: {supported_models}"}
            ), 400

        # Generate job ID
        job_id = f"training_{processed_dataset_id}_{model_id.split('/')[-1]}"
        logging.info(f"Starting training job {job_id} with model {model_id}")

        # Run training synchronously (blocking)
        result = run_training(processed_dataset_id, model_config, job_id)
        logging.info(f"Training completed successfully for job {job_id}")

        return jsonify({"job_id": job_id, "status": "completed", "result": result}), 200

    except Exception as e:
        return jsonify({"error": f"Failed to start training: {str(e)}"}), 500


def run_training(processed_dataset_id, model_config, job_id):
    """Run the actual training process"""

    logging.info(f"Starting training for job {job_id}")

    # 1. Download processed dataset from GCS
    try:
        train_dataset, eval_dataset = download_processed_dataset(processed_dataset_id)
        if train_dataset is None:
            raise ValueError("Failed to load training dataset - dataset is None")
        if len(train_dataset) == 0:
            raise ValueError("Training dataset is empty")

    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {str(e)}") from e

    # 2. Setup model and tokenizer
    try:
        model_id = model_config.get("model_id")
        model, tokenizer = setup_model(model_id, model_config.get("method", "LoRA"))
        logging.info(f"Model loaded successfully: {model_id}")
    except Exception as e:
        raise RuntimeError(f"Failed to set up model: {str(e)}") from e

    # 3. Setup LoRA configuration
    try:
        peft_config = setup_lora_config(model_config)
        logging.info(
            f"Setup LoRA config: rank={peft_config.r}, alpha={peft_config.lora_alpha}"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to set up LoRA: {str(e)}") from e

    # 4. Setup training arguments
    try:
        training_args = setup_training_args(model_config, job_id)
        logging.info(
            f"Setup training: {training_args.num_train_epochs} epochs, lr={training_args.learning_rate}"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to set up training args: {str(e)}") from e

    # 5. Setup metrics logging
    # metrics_logger = TrainingMetricsLogger(job_id)
    # logging.info("Training metrics logger enabled")

    # 6. Create trainer with metrics callback
    try:
        # when we pass tokenizer into the trainer it already handles chat template formatting, tokenization, and input prep
        # NOTE: in the future to support more customization in preprocessing we might want to explicilty handle this instead
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,
            processing_class=tokenizer,
            # callbacks=[metrics_logger],
        )
    except Exception as e:
        raise RuntimeError(f"Failed to set up trainer: {str(e)}") from e

    # 7. Start training
    try:
        # Clear cache before training
        torch.cuda.empty_cache()
        trainer.train()
    except Exception as e:
        # Try to clean up GPU memory
        torch.cuda.empty_cache()
        raise RuntimeError(f"Training failed: {str(e)}") from e

    # 8. Save adapter
    try:
        adapter_path = f"/tmp/{job_id}_adapter"
        trainer.save_model(adapter_path)
        logging.info(f"Saved adapter to {adapter_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to save adapter: {str(e)}") from e

    # 9. Upload adapter to GCS
    try:
        adapter_gcs_path = upload_adapter_to_gcs(
            adapter_path, job_id, model_id, peft_config
        )
        logging.info(f"Uploaded adapter and config to {adapter_gcs_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to upload adapter: {str(e)}") from e

    # 10. Clean up
    shutil.rmtree(adapter_path, ignore_errors=True)
    del model
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"adapter_path": adapter_gcs_path, "model_id": model_id, "job_id": job_id}


def download_processed_dataset(processed_dataset_id):
    """Download processed dataset from GCS"""
    try:
        bucket = storage_client.bucket(DATA_BUCKET_NAME)

        # Check if training dataset exists first
        train_blob = bucket.blob(
            f"processed_datasets/{processed_dataset_id}_train.json"
        )
        if not train_blob.exists():
            raise FileNotFoundError(
                f"Training dataset not found in GCS: processed_datasets/{processed_dataset_id}_train.json"
            )

        # Download train dataset
        try:
            train_data = json.loads(train_blob.download_as_text())
            if not train_data:
                raise ValueError("Training dataset is empty")
            train_dataset = Dataset.from_list(train_data)
            logging.info(
                f"Successfully loaded training dataset with {len(train_dataset)} samples"
            )
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse training dataset: {str(e)}")

        # Download eval dataset (if exists)
        eval_blob = bucket.blob(f"processed_datasets/{processed_dataset_id}_test.json")
        eval_dataset = None

        if eval_blob.exists():
            try:
                eval_data = json.loads(eval_blob.download_as_text())
                if eval_data:
                    eval_dataset = Dataset.from_list(eval_data)
                    logging.info(
                        f"Successfully loaded evaluation dataset with {len(eval_dataset)} samples"
                    )
            except Exception as e:
                logging.warning(f"Failed to load evaluation dataset: {str(e)}")
        else:
            logging.info("No evaluation dataset found (this is okay)")

        return train_dataset, eval_dataset

    except Exception as e:
        error_msg = f"Failed to load dataset: {str(e)}"
        logging.error(error_msg, exc_info=True)  # Add full traceback to logs
        raise RuntimeError(error_msg) from e


def setup_model(model_id, method):
    """Setup model and tokenizer based on configuration"""

    # Check GPU capabilities for dtype
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float16

    # Model loading arguments
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "attn_implementation": "eager",  # TODO: add "flash_attention_2" for newer GPUs need to install flash-attn first
        "device_map": "auto",
    }

    # Add quantization for QLoRA
    if method == "QLoRA":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        )

    try:
        # Load model with memory tracking
        logging.info("\nLoading model...")
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")

        # Add padding token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logging.info(f"- Model loaded successfully: {model_id}")

        return model, tokenizer

    except Exception as e:
        raise RuntimeError(f"Failed to load model {model_id}: {str(e)}") from e


def setup_lora_config(model_config):
    """Setup LoRA configuration"""
    return LoraConfig(
        lora_alpha=model_config.get("lora_alpha", 16),
        lora_dropout=model_config.get("lora_dropout", 0.05),
        r=model_config.get("lora_rank", 16),
        bias="none",
        # target_modules=[
        #     "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
        #     "gate_proj", "up_proj", "down_proj",     # MLP layers
        #     "embed_tokens", "lm_head"                # Embedding & output layers
        # ],
        target_modules="all-linear",  # Apply to all linear layers
        task_type="CAUSAL_LM",
        modules_to_save=[
            "lm_head",
            "embed_tokens",
        ],  # Save these modules during training
    )


def setup_training_args(model_config, job_id):
    """Setup training arguments"""

    # Determine precision based on GPU
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        fp16 = False
        bf16 = True
    else:
        fp16 = True
        bf16 = False

    return SFTConfig(
        output_dir=f"/tmp/{job_id}",
        max_seq_length=model_config.get("max_seq_length", 512),
        num_train_epochs=model_config.get("epochs", 3),
        per_device_train_batch_size=model_config.get("batch_size", 1),
        gradient_accumulation_steps=model_config.get("gradient_accumulation_steps", 4),
        learning_rate=model_config.get("learning_rate", 2e-4),
        packing=True,
        optim="adamw_torch_fused",
        fp16=fp16,
        bf16=bf16,
        logging_steps=10,
        save_strategy="epoch",
        push_to_hub=False,
        report_to="none",
    )


def upload_adapter_to_gcs(adapter_path, job_id, model_id, peft_config):
    """Upload trained adapter and config to GCS"""
    from datetime import datetime

    bucket = storage_client.bucket(EXPORT_BUCKET_NAME)

    # Save config file with model info
    config = {
        "base_model_id": model_id,
        "created_at": datetime.now().isoformat(),
        "lora_rank": peft_config.r,
        "lora_alpha": peft_config.lora_alpha,
    }

    config_blob = bucket.blob(f"trained_adapters/{job_id}/config.json")
    config_blob.upload_from_string(json.dumps(config), content_type="application/json")
    logging.info(
        f"Uploaded adapter config to gs://{EXPORT_BUCKET_NAME}/trained_adapters/{job_id}/config.json"
    )

    # Upload adapter files
    for root, dirs, files in os.walk(adapter_path):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, adapter_path)
            gcs_path = f"trained_adapters/{job_id}/{relative_path}"

            blob = bucket.blob(gcs_path)
            blob.upload_from_filename(local_path)
            logging.info(f"Uploaded {relative_path} to {gcs_path}")

    return f"gs://{EXPORT_BUCKET_NAME}/trained_adapters/{job_id}/"


@app.route("/inference/<job_id>", methods=["POST"])
def run_inference(job_id):
    """Run inference using a trained model adapter"""
    try:
        data = request.get_json()
        prompt = data.get("prompt")

        if not prompt:
            return jsonify({"error": "prompt is required"}), 400

        # Get adapter path from GCS
        bucket = storage_client.bucket(EXPORT_BUCKET_NAME)
        adapter_path = f"/tmp/inference_{job_id}"
        prefix = f"trained_adapters/{job_id}/"

        # Check if adapter files exist
        blobs = list(bucket.list_blobs(prefix=prefix))
        if not blobs:
            return jsonify({"error": f"Training job {job_id} not found"}), 404

        # Create local directory for adapter
        os.makedirs(adapter_path, exist_ok=True)

        # Download adapter files from GCS
        for blob in blobs:
            relative_path = os.path.relpath(blob.name, prefix)
            local_path = os.path.join(adapter_path, relative_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            blob.download_to_filename(local_path)

        # Load adapter config and model
        config_blob = bucket.blob(f"trained_adapters/{job_id}/config.json")
        if not config_blob.exists():
            return jsonify({"error": "Adapter config not found"}), 404

        config = json.loads(config_blob.download_as_text())
        model_id = config["base_model_id"]

        # Load model and tokenizer with correct base model
        model, tokenizer = setup_model(
            model_id, "LoRA"
        )  # Always use LoRA for inference

        # Load adapter weights
        try:
            model = PeftModel.from_pretrained(model, adapter_path)
        except Exception as e:
            shutil.rmtree(adapter_path, ignore_errors=True)  # Clean up on error
            raise RuntimeError(f"Failed to load adapter: {str(e)}")

        # Create pipeline for text generation
        pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer, device_map="auto"
        )

        # Prepare chat template
        messages = [{"role": "user", "content": prompt}]
        chat_prompt = pipe.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Set up stop tokens
        stop_tokens = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<end_of_turn>"),
        ]

        # Generate response
        outputs = pipe(
            chat_prompt,
            max_new_tokens=256,
            do_sample=False,
            top_k=50,
            eos_token_id=stop_tokens,
            disable_compile=True,  # Add this to prevent compilation issues
        )

        # Extract generated text
        generated_text = outputs[0]["generated_text"][len(chat_prompt) :].strip()

        # Clean up
        shutil.rmtree(adapter_path, ignore_errors=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return jsonify(
            {
                "result": generated_text.strip(),
                "model_info": {
                    "base_model": config["base_model_id"],
                    "lora_rank": config["lora_rank"],
                    "lora_alpha": config["lora_alpha"],
                    "created_at": config["created_at"],
                },
            }
        ), 200

    except Exception as e:
        error_msg = f"Failed to run inference: {str(e)}"
        logging.error(error_msg, exc_info=True)
        return jsonify({"error": error_msg}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8081))
    app.run(host="0.0.0.0", port=port, debug=True)
