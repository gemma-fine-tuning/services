import os
import logging
import torch
import shutil
from abc import ABC, abstractmethod
from model_storage import storage_service, CloudStoredModelMetadata
from training.schema import TrainRequest, TrainResponse


class BaseTrainingService(ABC):
    """Abstract base class for training services"""

    @abstractmethod
    def run_training(self, req: TrainRequest) -> TrainResponse:
        """
        Run training with the given configuration

        Args:
            res: `TrainRequest` containing model config and dataset info

        Returns:
            TrainResponse: Training result with job_id, adapter_path, and model_id
        """
        pass


class HuggingFaceTrainingService(BaseTrainingService):
    """Training service using HuggingFace Transformers and PEFT"""

    def __init__(self):
        # Import HF libraries only when this service is instantiated
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import LoraConfig
        from trl import SFTTrainer, SFTConfig

        # Store imports as instance variables to avoid namespace pollution
        self.AutoTokenizer = AutoTokenizer
        self.AutoModelForCausalLM = AutoModelForCausalLM
        self.BitsAndBytesConfig = BitsAndBytesConfig
        self.LoraConfig = LoraConfig
        self.SFTTrainer = SFTTrainer
        self.SFTConfig = SFTConfig

    def run_training(self, req: TrainRequest) -> TrainResponse:
        """Orchestrate full training workflow and return result"""
        processed_dataset_id = req.processed_dataset_id
        model_cfg = req.model_config.dict()
        model_id = model_cfg.get("model_id")
        job_id = f"training_{processed_dataset_id}_{model_id.split('/')[-1]}"
        logging.info(
            f"Starting HuggingFace training job {job_id} with model {model_id}"
        )

        # Call core training logic
        result = self._run_training_core(processed_dataset_id, model_cfg, job_id)

        return TrainResponse(
            job_id=job_id,
            adapter_path=result["adapter_path"],
            model_id=result["model_id"],
        )

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
        trainer = self.SFTTrainer(
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

        # Upload via storage manager
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
        # Determine dtype based on GPU capability
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float16

        # Check supported models
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

        # Add quantization for QLoRA
        if method == "QLoRA":
            model_kwargs["quantization_config"] = self.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
            )

        model = self.AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        tokenizer = self.AutoTokenizer.from_pretrained(model_id)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer

    def _setup_lora_config(self, model_config):
        return self.LoraConfig(
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

        return self.SFTConfig(
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


class UnslothTrainingService(BaseTrainingService):
    """Training service using Unsloth framework"""

    def __init__(self):
        # Import Unsloth libraries only when this service is instantiated
        from unsloth import FastModel
        from unsloth.chat_templates import (
            get_chat_template,
            standardize_data_formats,
            train_on_responses_only,
        )
        from datasets import load_dataset
        from trl import SFTTrainer, SFTConfig

        # Store imports as instance variables
        self.FastModel = FastModel
        self.get_chat_template = get_chat_template
        self.standardize_data_formats = standardize_data_formats
        self.train_on_responses_only = train_on_responses_only
        self.load_dataset = load_dataset
        self.SFTTrainer = SFTTrainer
        self.SFTConfig = SFTConfig

        # Available 4-bit models for frontend selection
        self.fourbit_models = [
            "unsloth/gemma-3-1b-it-unsloth-bnb-4bit",
            "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
            "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
            "unsloth/gemma-3-27b-it-unsloth-bnb-4bit",
        ]

    def run_training(self, req: TrainRequest) -> TrainResponse:
        """Orchestrate full Unsloth training workflow and return result"""
        processed_dataset_id = req.processed_dataset_id
        model_cfg = req.model_config.dict()
        model_id = model_cfg.get("model_id")
        job_id = f"training_{processed_dataset_id}_{model_id.split('/')[-1]}"
        logging.info(f"Starting Unsloth training job {job_id} with model {model_id}")

        # Call core training logic
        result = self._run_training_core(processed_dataset_id, model_cfg, job_id)

        return TrainResponse(
            job_id=job_id,
            adapter_path=result["adapter_path"],
            model_id=result["model_id"],
        )

    def _run_training_core(self, processed_dataset_id, model_config, job_id):
        # 1. Download processed dataset
        train_dataset, eval_dataset = storage_service.download_processed_dataset(
            processed_dataset_id
        )

        # 2. Setup model and tokenizer with Unsloth
        model, tokenizer = self._setup_unsloth_model(
            model_config.get("model_id"), model_config
        )

        # 3. Prepare dataset for Unsloth format
        # TODO: Figure out if this is necessary since we've already done preprocessing
        # If so we should do post-preprocessing to match Unsloth's expected format
        train_dataset = self._prepare_unsloth_dataset(train_dataset, tokenizer)

        # 4. Setup training args
        training_args = self._setup_unsloth_training_args(model_config, job_id)

        # 5. Train with Unsloth
        torch.cuda.empty_cache()
        trainer = self.SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=None,  # Unsloth typically doesn't use eval during training
            args=training_args,
        )

        # Apply response-only training
        trainer = self.train_on_responses_only(
            trainer,
            instruction_part="<start_of_turn>user\n",
            response_part="<start_of_turn>model\n",
        )

        trainer.train()

        # 6. Save & upload adapter with Unsloth flag
        adapter_path = f"/tmp/{job_id}_adapter"
        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)

        # Upload via storage manager with Unsloth flag
        metadata = CloudStoredModelMetadata(
            job_id=job_id,
            model_id=model_config.get("model_id"),
            gcs_prefix=f"trained_adapters/{job_id}",
            use_unsloth=True,  # Mark as Unsloth model
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

    def _setup_unsloth_model(self, model_id, model_config):
        """Setup Unsloth model and tokenizer"""
        # TODO: This sounds like a dangerous use of environment variables, prone to breaking software
        # NOTE: Considering passing this in as a parameter or storing it somewhere
        hf_token = os.environ.get("HUGGINGFACE_TOKEN")

        # Load base model with Unsloth
        model, tokenizer = self.FastModel.from_pretrained(
            model_name=model_id,
            max_seq_length=model_config.get("max_seq_length", 2048),
            load_in_4bit=True,
            load_in_8bit=False,
            full_finetuning=False,
            token=hf_token,
        )

        # Apply PEFT with Unsloth
        model = self.FastModel.get_peft_model(
            model,
            finetune_vision_layers=False,  # turn off since we're just doing text right now
            finetune_language_layers=True,  # optionally on or off
            finetune_attention_modules=False,  # this is helpful for GRPO, not PEFT use cases
            finetune_mlp_modules=True,  # should always be on according to docs
            r=model_config.get("lora_rank", 8),
            lora_alpha=model_config.get("lora_alpha", 8),
            lora_dropout=model_config.get("lora_dropout", 0.0),
            bias="none",
            random_state=3407,
        )

        # Setup chat template
        tokenizer = self.get_chat_template(
            tokenizer,
            chat_template="gemma-3",
        )

        return model, tokenizer

    def _prepare_unsloth_dataset(self, dataset, tokenizer):
        """Prepare dataset for Unsloth training format"""
        # Standardize format
        dataset = self.standardize_data_formats(dataset)

        def formatting_prompts_func(examples):
            convos = examples["conversations"]
            texts = [
                tokenizer.apply_chat_template(
                    convo, tokenize=False, add_generation_prompt=False
                ).removeprefix("<bos>")
                for convo in convos
            ]
            return {"text": texts}

        dataset = dataset.map(formatting_prompts_func, batched=True)
        return dataset

    def _setup_unsloth_training_args(self, model_config, job_id):
        """Setup training arguments for Unsloth"""
        return self.SFTConfig(
            dataset_text_field="text",
            output_dir=f"/tmp/{job_id}",
            per_device_train_batch_size=model_config.get("batch_size", 2),
            gradient_accumulation_steps=model_config.get(
                "gradient_accumulation_steps", 4
            ),
            warmup_steps=5,
            num_train_epochs=model_config.get("epochs", 1),
            learning_rate=model_config.get("learning_rate", 2e-4),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            report_to="none",
            dataset_num_proc=2,
        )


class TrainingService:
    """
    Training service with factory method design pattern to support multiple providers.
    This allows easy extension to add new training providers in the future.

    Usage:
    ```python
    # Create HuggingFace training service
    service = TrainingService.from_provider("huggingface")

    # Create Unsloth training service
    service = TrainingService.from_provider("unsloth")

    # Run training
    result = service.run_training(train_request)
    ```
    """

    _providers = {
        "huggingface": HuggingFaceTrainingService,
        "unsloth": UnslothTrainingService,
    }

    @classmethod
    def from_provider(cls, provider: str) -> BaseTrainingService:
        """
        Create training service instance using provider name.

        Args:
            provider: Training provider name. Supported values:
                - "huggingface": HuggingFace Transformers (TRL + SFT + PEFT)
                - "unsloth": Unsloth framework (built on top of HuggingFace)

        Returns:
            BaseTrainingService: Configured training service instance

        Raises:
            ValueError: If provider is not supported

        Learn more about Unsloth [here](https://www.unsloth.ai).
        """
        provider = provider.lower()

        if provider not in cls._providers:
            supported = list(cls._providers.keys())
            raise ValueError(
                f"Unsupported training provider '{provider}'. "
                f"Supported providers: {supported}"
            )

        service_class = cls._providers[provider]
        return service_class()

    @classmethod
    def list_providers(cls) -> list:
        """List all available training providers"""
        return list(cls._providers.keys())
