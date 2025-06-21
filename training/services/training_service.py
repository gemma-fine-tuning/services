import os
import logging
import torch
import shutil
from google.cloud import storage
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from training.schema import TrainRequest
from services.model_storage import storage_service, CloudStoredModelMetadata


class TrainingService:
    def __init__(self):
        # Initialize GCS client and bucket names
        self.storage_client = storage.Client()
        self.data_bucket = os.environ.get("GCS_DATA_BUCKET_NAME", "gemma-dataset-dev")
        self.export_bucket = os.environ.get(
            "GCS_EXPORT_BUCKET_NAME", "gemma-export-dev"
        )

    def run_training(self, req: TrainRequest) -> dict:
        """Orchestrate full training workflow and return result"""
        processed_dataset_id = req.processed_dataset_id
        model_cfg = req.model_config.dict()
        model_id = model_cfg.get("model_id")
        job_id = f"training_{processed_dataset_id}_{model_id.split('/')[-1]}"
        logging.info(f"Starting training job {job_id} with model {model_id}")

        # Call core training logic
        result = self._run_training_core(processed_dataset_id, model_cfg, job_id)
        result["job_id"] = job_id
        return result

    def _run_training_core(self, processed_dataset_id, model_config, job_id):
        # 1. Download processed dataset
        train_dataset, eval_dataset = storage_service.download_processed_dataset(
            processed_dataset_id
        )

        # 2. Setup model and tokenizer
        model, tokenizer = self._setup_model(
            model_config.get("model_id", "google/gemma-3-1b-it"),
            model_config.get("method", "LoRA"),
        )

        # 3. Setup LoRA config
        peft_config = self._setup_lora_config(model_config)

        # 4. Setup training args
        training_args = self._setup_training_args(model_config, job_id)

        # 5. Train
        torch.cuda.empty_cache()
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,
            processing_class=tokenizer,
        )
        trainer.train()

        # 6. Save & upload adapter
        adapter_path = f"/tmp/{job_id}_adapter"
        trainer.save_model(adapter_path)
        # upload via storage manager
        metadata = CloudStoredModelMetadata(
            job_id=job_id,
            model_id=model_config.get("model_id"),
            gcs_prefix=f"trained_adapters/{job_id}",
            use_unsloth=False,
            local_dir=adapter_path,
        )
        adapter_gcs_path = storage_service.upload_model(adapter_path, metadata)

        # 7. Cleanup
        shutil.rmtree(adapter_path, ignore_errors=True)
        del model, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "adapter_path": adapter_gcs_path,
            "model_id": model_config.get("model_id"),
        }

    def _setup_model(self, model_id, method):
        # dtype
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float16

        # check available models
        supported_models = [
            "google/gemma-3-1b-it",
            "google/gemma-3-4b-it",
            "google/gemma-3-12b-it",
            "google/gemma-3-27b-it",
        ]

        if model_id not in supported_models:
            raise ValueError(
                f"Unsupported model {model_id}. Supported models are: {', '.join(supported_models)}"
            )

        model_kwargs = {
            "torch_dtype": torch_dtype,
            "attn_implementation": "eager",
            "device_map": "auto",
        }
        if method == "QLoRA":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
            )
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer

    def _setup_lora_config(self, model_config):
        return LoraConfig(
            lora_alpha=model_config.get("lora_alpha", 16),
            lora_dropout=model_config.get("lora_dropout", 0.05),
            r=model_config.get("lora_rank", 16),
            bias="none",
            target_modules="all-linear",
            task_type="CAUSAL_LM",
            modules_to_save=["lm_head", "embed_tokens"],
        )

    def _setup_training_args(self, model_config, job_id):
        fp16 = True
        bf16 = False
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            fp16 = False
            bf16 = True
        return SFTConfig(
            output_dir=f"/tmp/{job_id}",
            max_seq_length=model_config.get("max_seq_length", 512),
            num_train_epochs=model_config.get("epochs", 3),
            per_device_train_batch_size=model_config.get("batch_size", 1),
            gradient_accumulation_steps=model_config.get(
                "gradient_accumulation_steps", 4
            ),
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
