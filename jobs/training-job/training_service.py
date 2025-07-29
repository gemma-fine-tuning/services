import logging
import torch
import io
from abc import ABC, abstractmethod
from storage import storage_service, StorageStrategyFactory
from schema import TrainRequest, WandbConfig
from typing import Optional, Tuple, List, Dict, Any
from job_manager import JobTracker


def compute_metrics(eval_pred):
    """
    This does not use evaluate library for now
    in the future we will use this: https://huggingface.co/docs/evaluate/en/transformers_integrations

    Steps:
    1. First test and validate whatever we already have right now
    1.5 Refactor whatever we have right now this is too many changes for one feature (boilerplate lmao)
    2. setup the integration with evaluate library using "metric"
    3. Pick metrics, measurements, etc. to use and put them down here and ship!
    """
    logits, labels = eval_pred
    # SFTTrainer will provide logits and labels.
    # Labels might have -100 for ignored tokens.
    # Convert labels to 0s for perplexity calculation where -100 are ignored
    labels[labels == -100] = 0

    # Ensure logits and labels are on CPU for metric calculation if they are on GPU
    logits = torch.from_numpy(logits).cpu()
    labels = torch.from_numpy(labels).cpu()

    # Perplexity expects logits as input (or probabilities)
    # The metric will handle masking out the 0s (originally -100) if configured
    # You might need to adjust based on how your model outputs and how perplexity metric expects
    # Often, you need to flatten the logits and labels for token-level metrics.

    # Example for perplexity (might need adjustment based on your specific model output)
    # For causal LMs, the labels are shifted input_ids.
    # The perplexity metric typically works directly on inputs and model outputs.
    # It's often easier to let the Trainer handle eval loss (which is perplexity-related)
    # and add custom metrics for other things like accuracy if needed.

    # For a simple eval_loss:
    # The Trainer already calculates eval_loss which is cross-entropy, related to perplexity.
    # If you want to compute perplexity explicitly, it's 2 ** eval_loss.

    # For other metrics like accuracy, you'd convert logits to predictions
    predictions = logits.argmax(axis=-1)

    # Flatten everything to compare token by token, ignoring -100
    valid_labels = labels[labels != -100]
    valid_predictions = predictions[labels != -100]

    # Simple accuracy example
    accuracy = (valid_predictions == valid_labels).float().mean().item()

    # You can add more metrics here (ROUGE, BLEU for generation tasks, but that's more complex)

    return {"perplexity": 2 ** eval_pred.metrics["eval_loss"], "accuracy": accuracy}


class BaseTrainingService(ABC):
    """
    Abstract base class for all training services.
    Defines the interface for running training jobs and WandB setup.
    Subclasses must implement the run_training method for their specific framework.
    """

    @abstractmethod
    def run_training(self, req: TrainRequest, job_tracker: JobTracker) -> dict:
        """
        Run training with the given configuration and optional job tracker.
        This method orchestrates the full training workflow, including model setup,
        dataset preparation, training, and artifact saving.

        NOTE: This should save the result to firestore because otherwise we won't return anything from cloud run jobs!

        Args:
            req: `TrainRequest` containing model config and dataset info
            job_tracker: Optional JobTracker instance for granular status updates

        Returns:
            dict: Training result with job_id, adapter_path, and base_model_id
        """
        pass

    def _setup_wandb(
        self, config: Optional[WandbConfig], job_id: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Setup WandB and return (report_to, wandb_url).
        Handles WandB login and initialization for experiment tracking.

        Args:
            config: WandB configuration
            job_id: Job identifier for WandB run name

        Returns:
            Tuple of (report_to, wandb_url) where wandb_url is "" if not configured
        """
        if not config or not config.api_key:
            return "none", ""

        # Import wandb here to avoid import at module level
        import wandb

        # Login to wandb (sets API key locally)
        wandb.login(key=config.api_key)

        # Initialize wandb with user settings
        wandb.init(
            project=config.project or "gemma-fine-tuning",
            name=job_id,
            tags=["fine-tuning"],
        )

        # Get the actual wandb URL
        # NOTE: This assumes that wandb generates a URL that is consistent with the run
        wandb_url = wandb.run.get_url() if wandb.run else None

        return "wandb", wandb_url or ""


class HuggingFaceTrainingService(BaseTrainingService):
    """
    Training service using HuggingFace Transformers and PEFT.
    Supports LoRA, QLoRA, and full fine-tuning for Gemma models.
    Handles model setup, dataset preparation, training, and artifact export.
    """

    def __init__(self) -> None:
        # Import HF libraries only when this service is instantiated
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
            BitsAndBytesConfig,
            AutoModelForImageTextToText,
        )
        from peft import LoraConfig, get_peft_model
        from trl import SFTTrainer, SFTConfig

        # Store imports as instance variables to avoid namespace pollution
        self.AutoTokenizer = AutoTokenizer
        self.AutoModelForCausalLM = AutoModelForCausalLM
        self.BitsAndBytesConfig = BitsAndBytesConfig
        self.AutoModelForImageTextToText = AutoModelForImageTextToText
        self.LoraConfig = LoraConfig
        self.SFTTrainer = SFTTrainer
        self.SFTConfig = SFTConfig
        self.get_peft_model = get_peft_model

        self.supported_models = [
            "google/gemma-3-1b-it",
            "google/gemma-3-4b-it",
            "google/gemma-3-12b-it",
            # "google/gemma-3-27b-it",
            "google/gemma-3n-E2B-it",
            "google/gemma-3n-E4B-it",
        ]

        # Determine dtype based on GPU capability
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            self.torch_dtype = torch.bfloat16
        else:
            self.torch_dtype = torch.float16

    def run_training(
        self, req: TrainRequest, job_tracker: JobTracker
    ) -> Dict[str, Any]:
        """
        Orchestrate full training workflow and return result.
        Handles all steps from dataset download to model export for HuggingFace models.

        Args:
            req: `TrainRequest` containing model config and dataset info
            job_tracker: Optional JobTracker instance for granular status updates

        Returns:
            dict: Training result with job_id, adapter_path, and base_model_id
        """
        try:
            processed_dataset_id = req.processed_dataset_id
            model_cfg = req.training_config.model_dump()
            modality = model_cfg.get("modality", "text")
            base_model_id = model_cfg.get("base_model_id", "google/gemma-3-1b-it")
            job_id = job_tracker.job_id

            logging.info(
                f"Starting HuggingFace training job {job_id} with model {base_model_id} (modality={modality})"
            )
            logging.info(f"Training config: {model_cfg}")
            logging.info(f"Export type: {req.export}")

            # Route to appropriate core method based on modality
            if modality == "vision":
                result = self._run_training_hf_vision_core(
                    processed_dataset_id,
                    model_cfg,
                    job_id,
                    req.export,
                    job_tracker,
                    req.hf_repo_id,
                    req.wandb_config,
                )
            else:
                result = self._run_training_core(
                    processed_dataset_id,
                    model_cfg,
                    job_id,
                    req.export,
                    job_tracker,
                    req.hf_repo_id,
                    req.wandb_config,
                )

            logging.info(f"Training completed successfully for job {job_id}")
            return {
                "job_id": job_id,
                "adapter_path": result["adapter_path"],
                "base_model_id": result["base_model_id"],
            }
        except Exception as e:
            logging.error(f"HuggingFace training failed: {str(e)}", exc_info=True)
            raise e

    def _run_training_core(
        self,
        processed_dataset_id: str,
        model_config: Dict[str, Any],
        job_id: str,
        export: str,
        job_tracker: JobTracker,
        hf_repo_id: Optional[str] = None,
        wandb_config: Optional[WandbConfig] = None,
    ) -> Dict[str, Any]:
        """
        Core training logic for HuggingFace models.
        Handles dataset download, model/tokenizer setup, LoRA config, training, and export.
        Updates job status via job_tracker if provided.

        Args:
            processed_dataset_id: ID of the processed dataset to train on
            model_config: Model configuration parameters
            job_id: Unique identifier for the training job
            export: Export type (e.g., to GCS or HuggingFace Hub)
            hf_repo_id: Optional HuggingFace repository ID for model saving
            wandb_config: Optional WandB configuration for experiment tracking
            job_tracker: Optional JobTracker instance for granular status updates

        Returns:
            dict: Contains `adapter_path` and `base_model_id` of the trained model
        """
        job_tracker.preparing()

        # 1. Download processed dataset
        train_dataset, eval_dataset = storage_service.download_processed_dataset(
            processed_dataset_id
        )

        # 2. Setup model and tokenizer
        model, tokenizer = self._setup_model(
            model_config.get("base_model_id", "google/gemma-3-1b-it"),
            model_config.get("method", "LoRA"),
            model_config.get("use_fa2", False),
        )

        # 3. Prepare datasets for HuggingFace training format
        train_dataset = self._prepare_hf_dataset(train_dataset, tokenizer)
        prepared_eval_dataset = None
        if eval_dataset is not None:
            prepared_eval_dataset = self._prepare_hf_dataset(eval_dataset, tokenizer)

        # 4. Setup LoRA config
        # We wrap the model with PEFT as long as the method is not specified as full fine tuning
        if model_config.get("method", "LoRA") != "Full":
            model = self._setup_peft_with_lora(model, model_config)
            print(model.print_trainable_parameters())
        else:
            logging.info(
                "Full fine-tuning selected, not applying LoRA or QLoRA. This will be slow and might result in GPU OOM"
            )

        # 4.5. Setup WandB and get URL
        report_to, wandb_url = self._setup_wandb(wandb_config, job_id)

        # 5. Setup training args
        training_args = self._setup_training_args(model_config, job_id, report_to)
        # Configure evaluation strategy if eval dataset provided
        if prepared_eval_dataset is not None:
            # run evaluation each epoch
            setattr(training_args, "evaluation_strategy", "epoch")

        # 6. Train
        # Clear CUDA cache before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Get available GPU memory
            available_mem = torch.cuda.get_device_properties(
                0
            ).total_memory - torch.cuda.memory_allocated(0)
            if available_mem < 4 * 1024 * 1024 * 1024:  # 4GB threshold
                raise RuntimeError("Not enough GPU memory available for training")

        # Initialize trainer with metrics if evaluation dataset is present
        trainer = self.SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=prepared_eval_dataset,
            processing_class=tokenizer,
            compute_metrics=compute_metrics
            if prepared_eval_dataset is not None
            else None,
        )

        # Progress tracking: start actual training with WandB URL
        job_tracker.training(wandb_url)

        trainer.train()

        # Run evaluation explicitly at the end if eval_dataset is provided
        metrics = None
        if prepared_eval_dataset is not None:
            # eval_dataset has already been set and tokenized before
            eval_results = trainer.evaluate()
            logging.info(f"Evaluation results: {eval_results}")
            metrics = eval_results

        # Use storage strategy to save model and mark completion
        try:
            storage_strategy = StorageStrategyFactory.create_strategy(export)
            artifact = storage_strategy.save_model(
                model,
                tokenizer,
                f"/tmp/{job_id}_adapter",
                {
                    "job_id": job_id,
                    "base_model_id": model_config.get("base_model_id"),
                    "use_unsloth": False,
                    "hf_repo_id": hf_repo_id,
                },
            )
        except Exception as e:
            logging.error(f"Failed to save model: {str(e)}", exc_info=True)
            raise e

        # Mark job as completed with optional evaluation metrics
        job_tracker.completed(artifact.remote_path, artifact.base_model_id, metrics)

        return {
            "adapter_path": artifact.remote_path,
            "base_model_id": artifact.base_model_id,
        }

    def _run_training_hf_vision_core(
        self,
        processed_dataset_id: str,
        model_config: Dict[str, Any],
        job_id: str,
        export: str,
        job_tracker: JobTracker,
        hf_repo_id: Optional[str] = None,
        wandb_config: Optional[WandbConfig] = None,
    ) -> Dict[str, Any]:
        """
        Core training logic for HuggingFace vision fine-tuning.
        Uses pre-processed datasets from our preprocessing pipeline, similar to text training.
        Source: https://ai.google.dev/gemma/docs/core/huggingface_vision_finetune_qlora
        https://huggingface.co/docs/trl/en/training_vlm_sft
        """
        from transformers import AutoProcessor
        from PIL import Image

        job_tracker.preparing()

        # 1. Download processed dataset (same as text training)
        train_dataset, eval_dataset = storage_service.download_processed_dataset(
            processed_dataset_id
        )

        # 2. Setup model and processor for vision
        base_model_id = model_config.get("base_model_id", "google/gemma-3-4b-it")

        if base_model_id == "google/gemma-3-1b-it":
            raise ValueError(
                "Gemma 3.1B does not support vision fine-tuning. Use Gemma 3.4B or larger."
            )

        model_kwargs = {
            "torch_dtype": self.torch_dtype,
            "device_map": "auto",
            "attn_implementation": "eager",
        }

        # Add quantization for QLoRA
        if model_config.get("method", "LoRA") == "QLoRA":
            model_kwargs["quantization_config"] = self.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.torch_dtype,
            )

        processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)
        processor.tokenizer.padding_side = "right"

        model = self.AutoModelForImageTextToText.from_pretrained(
            base_model_id, **model_kwargs
        )

        # 3. Define collate function for vision training (handles pre-processed format)
        def vision_collate_fn(examples):
            """Collate function for vision datasets that are already in ChatML format with images."""
            texts = [
                processor.apply_chat_template(
                    example["messages"], tokenize=False, add_generation_prompt=False
                ).strip()
                for example in examples
            ]

            # Extract images from pre-processed ChatML messages
            def extract_images_from_messages(messages):
                images = []
                for msg in messages:
                    content = msg.get("content", [])
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "image":
                                img_data = item.get("image")
                                if img_data:
                                    # Handle PIL Image objects
                                    if isinstance(img_data, Image.Image):
                                        images.append(img_data.convert("RGB"))
                                    # Handle HuggingFace dataset format: {"bytes": ..., "path": null}
                                    elif (
                                        isinstance(img_data, dict)
                                        and "bytes" in img_data
                                    ):
                                        image_bytes = img_data["bytes"]
                                        if isinstance(image_bytes, (bytes, bytearray)):
                                            pil_image = Image.open(
                                                io.BytesIO(image_bytes)
                                            )
                                            images.append(pil_image.convert("RGB"))
                return images

            images = [
                extract_images_from_messages(example["messages"])
                for example in examples
            ]

            # Tokenize texts and process images
            batch = processor(
                text=texts, images=images, return_tensors="pt", padding=True
            )

            # Setup labels (mask padding and special tokens)
            labels = batch["input_ids"].clone()

            # Mask image tokens if they exist
            try:
                # Mask image tokens
                image_token_id = [
                    processor.tokenizer.convert_tokens_to_ids(
                        processor.tokenizer.special_tokens_map["boi_token"]
                    )
                ]
                labels[labels == image_token_id] = -100
            except KeyError:
                logging.warning(
                    "Trying to run vision training but no image token found"
                )
                pass  # Skip if no image token

            # Mask other tokens not used in loss computation
            labels[labels == processor.tokenizer.pad_token_id] = -100
            labels[labels == 262144] = -100

            batch["labels"] = labels
            return batch

        # 4. Setup PEFT with LoRA if not full fine-tuning
        if model_config.get("method", "LoRA") != "Full":
            model = self._setup_peft_with_lora(model, model_config)
            logging.info(
                f"PEFT model trainable parameters: {model.print_trainable_parameters()}"
            )
        else:
            logging.info(
                "Full fine-tuning selected for vision model. This will be slow and might result in GPU OOM"
            )

        # 5. Setup WandB and training args
        report_to, wandb_url = self._setup_wandb(wandb_config, job_id)

        training_args = self.SFTConfig(
            output_dir=f"/tmp/{job_id}",
            per_device_train_batch_size=model_config.get("batch_size", 1),
            gradient_accumulation_steps=model_config.get(
                "gradient_accumulation_steps", 4
            ),
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            remove_unused_columns=False,
            dataset_kwargs={"skip_prepare_dataset": True},  # important for collator
            warmup_steps=5,
            num_train_epochs=model_config.get("epochs", 3),
            max_steps=model_config.get("max_steps", -1),
            learning_rate=model_config.get("learning_rate", 2e-4),
            fp16=self.torch_dtype == torch.float16,
            bf16=self.torch_dtype == torch.bfloat16,
            optim="adamw_torch_fused",
            lr_scheduler_type="linear",
            weight_decay=0.01,
            save_strategy="epoch",
            # use alternatively use steps for eval_strategy and set eval_steps
            # NOTE: We chose epoch because evaluation takes a lot of memory lol
            logging_steps=10,
            report_to=report_to,
            dataset_text_field="",  # need a dummy field for collator
        )

        training_args.remove_unused_columns = False  # important for collator

        if eval_dataset is not None:
            # run evaluation each epoch
            training_args.evaluation_strategy = "epoch"

        # 6. Create trainer
        trainer = self.SFTTrainer(
            model=model,
            args=training_args,
            data_collator=vision_collate_fn,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processor.tokenizer,
            compute_metrics=compute_metrics if eval_dataset is not None else None,
        )

        # 7. Train
        job_tracker.training(wandb_url)
        trainer.train()

        metrics = None
        if eval_dataset is not None:
            eval_results = trainer.evaluate()
            logging.info(f"Initial evaluation results: {eval_results}")
            metrics = eval_results

        # 8. Save model
        try:
            storage_strategy = StorageStrategyFactory.create_strategy(export)
            artifact = storage_strategy.save_model(
                model,
                processor.tokenizer,
                f"/tmp/{job_id}_hf_vision",
                {
                    "job_id": job_id,
                    "base_model_id": base_model_id,
                    "use_unsloth": False,
                    "modality": "vision",
                    "hf_repo_id": hf_repo_id,
                },
            )

            # 9. Cleanup
            storage_strategy.cleanup(artifact)
            del model, trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logging.error(f"Failed to save vision model: {str(e)}", exc_info=True)
            raise e

        job_tracker.completed(artifact.remote_path, artifact.base_model_id, metrics)

        return {
            "adapter_path": artifact.remote_path,
            "base_model_id": artifact.base_model_id,
        }

    def _setup_model(
        self, base_model_id: str, method: str, use_fa2: bool
    ) -> Tuple[Any, Any]:
        """
        Setup and return the base model and tokenizer for training.
        Applies quantization and attention optimizations as needed.

        Args:
            base_model_id: ID of the base model to use
            method: Fine-tuning method (e.g., LoRA, QLoRA, Full)
            use_fa2: Whether to use FlashAttention 2.0

        Returns:
            tuple: (model, tokenizer) - the initialized model and tokenizer
        """

        if base_model_id not in self.supported_models:
            raise ValueError(
                f"Unsupported model {base_model_id}. Supported models are: {', '.join(self.supported_models)}"
            )

        model_kwargs = {
            "torch_dtype": self.torch_dtype,
            "attn_implementation": "flash_attention_2" if use_fa2 else "eager",
            "device_map": "auto",
        }

        # Add quantization for QLoRA
        if method == "QLoRA":
            model_kwargs["quantization_config"] = self.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.torch_dtype,
            )

        if base_model_id == "google/gemma-3-1b-it":
            model = self.AutoModelForCausalLM.from_pretrained(
                base_model_id, **model_kwargs
            )
        else:
            # For multimodal models, use AutoModelForImageTextToText
            model = self.AutoModelForImageTextToText.from_pretrained(
                base_model_id, **model_kwargs
            )
        tokenizer = self.AutoTokenizer.from_pretrained(base_model_id)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer

    def _setup_peft_with_lora(self, model: Any, model_config: Dict[str, Any]) -> Any:
        """
        Setup PEFT with LoRA configuration for the model.
        Returns a PEFT-wrapped model ready for training.

        Args:
            model: The base model to wrap with LoRA
            model_config: Model configuration parameters

        Returns:
            The PEFT-wrapped model with LoRA applied
        """
        lora_config = self.LoraConfig(
            lora_alpha=model_config.get("lora_alpha", 16),
            lora_dropout=model_config.get("lora_dropout", 0.05),
            r=model_config.get("lora_rank", 16),
            bias="none",
            target_modules="all-linear",
            task_type="CAUSAL_LM",
            modules_to_save=["lm_head", "embed_tokens"],
        )

        return self.get_peft_model(model, lora_config)

    def _setup_training_args(
        self, model_config: Dict[str, Any], job_id: str, report_to: str = "none"
    ) -> Any:
        """
        Setup training arguments for SFTTrainer.
        Returns a SFTConfig object with all training hyperparameters.

        Args:
            model_config: Model configuration parameters
            job_id: Unique identifier for the training job
            report_to: Reporting destination for logs (e.g., "none", "wandb")

        Returns:
            SFTConfig: Configured training arguments
        """
        if self.torch_dtype == torch.bfloat16:
            bf16 = True
            fp16 = False
        else:
            bf16 = False
            fp16 = True

        # TODO: Migrate this note to the documentation to guide the user:
        # Trainer calculates steps per epoch as num_samples / batch_size / gradient_accumulation_steps
        # Each step is defined by a weights update
        # If max_steps is set, it will override num_train_epochs.
        # Source: https://stackoverflow.com/questions/76002567/how-is-the-number-of-steps-calculated-in-huggingface-trainer

        return self.SFTConfig(
            output_dir=f"/tmp/{job_id}",
            # Batch size on each GPU if multiple GPUs are used, for simplicity we use same batch size for train and eval
            per_device_train_batch_size=model_config.get("batch_size", 4),
            per_device_eval_batch_size=model_config.get("batch_size", 4),
            # Do forward pass multiple times before updating weights
            gradient_accumulation_steps=model_config.get(
                "gradient_accumulation_steps", 4
            ),
            warmup_steps=5,
            # This is the max seq length of the prompt
            num_train_epochs=model_config.get("epochs", 3),
            # using max_steps will directly override num_train_epochs
            max_steps=model_config.get("max_steps", -1),  # -1 means no max steps
            learning_rate=model_config.get("learning_rate", 2e-4),
            # Packing packs the tokenized sequences to max_len
            packing=model_config.get("packing", True),
            fp16=fp16,
            bf16=bf16,
            optim="adamw_torch_fused",
            lr_scheduler_type="linear",
            weight_decay=0.01,
            save_strategy="epoch",
            push_to_hub=False,
            logging_steps=10,
            report_to=report_to,
            # NOTE: This is not available because gemma3 chat template does not contain {% generate %} tag
            # assistant_only_loss=True,  # for conversational dataset
        )

    def _prepare_hf_dataset(self, dataset: Any, tokenizer: Any) -> Any:
        """
        Prepare dataset for HuggingFace SFTTrainer format.
        Ensures the dataset is in the correct conversational format for training.

        Args:
            dataset: The dataset to prepare
            tokenizer: The tokenizer to use for encoding

        Returns:
            The prepared dataset ready for SFTTrainer
        """

        # HuggingFace SFTTrainer expects datasets with "messages" field
        def verify_format(examples):
            # Verify that all examples have the expected "messages" field
            messages_batch = examples.get("messages", [])
            if not messages_batch:
                raise ValueError(
                    "Dataset is missing 'messages' field - incompatible with preprocessing output"
                )
            return examples

        # Apply verification
        dataset = dataset.map(verify_format, batched=True)
        return dataset


class UnslothTrainingService(BaseTrainingService):
    """
    Training service using the Unsloth framework for efficient LLM fine-tuning.
    Supports 4-bit quantization and custom chat templates for Gemma models.
    Handles model setup, dataset formatting, training, and artifact export.
    """

    def __init__(self) -> None:
        # Import Unsloth libraries only when this service is instantiated
        from unsloth import FastModel, FastVisionModel
        from unsloth.chat_templates import (
            get_chat_template,
            standardize_data_formats,
            train_on_responses_only,
        )
        from unsloth.trainer import UnslothVisionDataCollator
        from trl import SFTTrainer, SFTConfig

        # Store imports as instance variables
        self.FastModel = FastModel
        self.FastVisionModel = FastVisionModel
        self.get_chat_template = get_chat_template
        self.standardize_data_formats = standardize_data_formats
        self.train_on_responses_only = train_on_responses_only
        self.UnslothVisionDataCollator = UnslothVisionDataCollator
        self.SFTTrainer = SFTTrainer
        self.SFTConfig = SFTConfig

        # Available 4-bit models for frontend selection
        self.fourbit_models = [
            "unsloth/gemma-3-1b-it-unsloth-bnb-4bit",
            "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
            "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
            # "unsloth/gemma-3-27b-it-unsloth-bnb-4bit",
            "unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit",
            "unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit",
        ]

        self.fourbit_vision_models = [
            "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
            "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
            # "unsloth/gemma-3-27b-it-unsloth-bnb-4bit",
            "unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit",
            "unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit",
        ]

        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            self.torch_dtype = torch.bfloat16
        else:
            self.torch_dtype = torch.float16

    def run_training(
        self, req: TrainRequest, job_tracker: JobTracker
    ) -> Dict[str, Any]:
        """
        Orchestrate full Unsloth training workflow and return result.
        Handles all steps from dataset download to model export for Unsloth models.

        Args:
            req: `TrainRequest` containing model config and dataset info
            job_tracker: Optional JobTracker instance for granular status updates

        Returns:
            dict: Training result with job_id, adapter_path, and base_model_id
        """
        try:
            processed_dataset_id = req.processed_dataset_id
            model_cfg = req.training_config.model_dump()
            base_model_id = model_cfg.get(
                "base_model_id", "unsloth/gemma-3-1b-it-unsloth-bnb-4bit"
            )
            job_id = job_tracker.job_id

            logging.info(
                f"Starting Unsloth training job {job_id} with model {base_model_id}"
            )
            logging.info(f"Training config: {model_cfg}")
            logging.info(f"Export type: {req.export}")

            # Call core training logic with job tracker based on modality
            if model_cfg.get("modality", "text") == "vision":
                result = self._run_training_vision_core(
                    processed_dataset_id,
                    model_cfg,
                    job_id,
                    req.export,
                    job_tracker,
                    req.hf_repo_id,
                    req.wandb_config,
                )
            else:
                result = self._run_training_text_core(
                    processed_dataset_id,
                    model_cfg,
                    job_id,
                    req.export,
                    job_tracker,
                    req.hf_repo_id,
                    req.wandb_config,
                )

            logging.info(f"Unsloth training completed successfully for job {job_id}")
            return {
                "job_id": job_id,
                "adapter_path": result["adapter_path"],
                "base_model_id": result["base_model_id"],
            }
        except Exception as e:
            logging.error(f"Unsloth training failed: {str(e)}", exc_info=True)
            raise e

    def _run_training_text_core(
        self,
        processed_dataset_id: str,
        model_config: Dict[str, Any],
        job_id: str,
        export: str,
        job_tracker: JobTracker,
        hf_repo_id: Optional[str] = None,
        wandb_config: Optional[WandbConfig] = None,
    ) -> Dict[str, Any]:
        """
        Core training logic for Unsloth text models.
        """
        job_tracker.preparing()

        # 1. Download processed dataset
        train_dataset, eval_dataset = storage_service.download_processed_dataset(
            processed_dataset_id
        )

        # 2. Setup model and tokenizer with Unsloth
        model, tokenizer = self._setup_unsloth_model(
            model_config.get("base_model_id", "unsloth/gemma-3-1b-it-unsloth-bnb-4bit"),
            model_config,
        )

        # 3. Prepare dataset for Unsloth format
        train_dataset = self._prepare_unsloth_dataset(train_dataset, tokenizer)

        # Prepare eval dataset if available
        prepared_eval_dataset = None
        if eval_dataset is not None:
            prepared_eval_dataset = self._prepare_unsloth_dataset(
                eval_dataset, tokenizer
            )

        # 4. Setup WandB and get URL
        report_to, wandb_url = self._setup_wandb(wandb_config, job_id)

        # 5. Setup training args
        training_args = self._setup_unsloth_training_args(
            model_config, job_id, report_to
        )

        if prepared_eval_dataset is not None:
            # TODO: Make this user configuration step or epoch in the future
            # If step we need to set eval_steps
            training_args["eval_strategy"] = "epoch"

        # 6. Train with Unsloth
        torch.cuda.empty_cache()
        trainer = self.SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=prepared_eval_dataset,
            args=training_args,
            compute_metrics=compute_metrics
            if prepared_eval_dataset is not None
            else None,
        )

        # Apply response-only training for text models
        trainer = self.train_on_responses_only(
            trainer,
            instruction_part="<start_of_turn>user\n",
            response_part="<start_of_turn>model\n",
        )

        # Progress tracking: start actual training with WandB URL
        job_tracker.training(wandb_url)

        trainer.train()

        metrics = None
        if prepared_eval_dataset is not None:
            eval_results = trainer.evaluate()
            logging.info(f"Evaluation results: {eval_results}")
            metrics = eval_results

        # Use storage strategy to save model
        try:
            storage_strategy = StorageStrategyFactory.create_strategy(export)
            artifact = storage_strategy.save_model(
                model,
                tokenizer,
                f"/tmp/{job_id}_adapter",
                {
                    "job_id": job_id,
                    "base_model_id": model_config.get("base_model_id"),
                    "use_unsloth": True,
                    "hf_repo_id": hf_repo_id,
                },
            )
        except Exception as e:
            logging.error(f"Failed to save model: {str(e)}", exc_info=True)
            raise e

        # 7. Cleanup
        storage_strategy.cleanup(artifact)
        del model, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Mark job as completed with evaluation metrics if available
        job_tracker.completed(artifact.remote_path, artifact.base_model_id, metrics)

        return {
            "adapter_path": artifact.remote_path,
            "base_model_id": artifact.base_model_id,
        }

    def _run_training_vision_core(
        self,
        processed_dataset_id: str,
        model_config: Dict[str, Any],
        job_id: str,
        export: str,
        job_tracker: JobTracker,
        hf_repo_id: Optional[str] = None,
        wandb_config: Optional[WandbConfig] = None,
    ) -> Dict[str, Any]:
        """
        Core training logic for Unsloth vision models.
        NOTE: For now we don't have preprocess_vision_dataset unlike text I cannot find a source indicating that this is needed
        """
        job_tracker.preparing()

        # 1. Download processed dataset
        train_dataset, eval_dataset = storage_service.download_processed_dataset(
            processed_dataset_id
        )

        # 2. Setup vision model and processor with Unsloth
        model, processor = self._setup_unsloth_vision_model(
            model_config.get("base_model_id", "unsloth/gemma-3-4b-it-unsloth-bnb-4bit"),
            model_config,
        )

        # 3. Setup WandB and get URL
        report_to, wandb_url = self._setup_wandb(wandb_config, job_id)

        # 4. Setup training args for vision
        training_args = self._setup_unsloth_vision_training_args(
            model_config, job_id, report_to
        )

        if eval_dataset is not None:
            training_args["eval_strategy"] = "epoch"

        # 5. Train with Unsloth Vision
        torch.cuda.empty_cache()

        # Enable training mode for vision model
        self.FastVisionModel.for_training(model)

        # For vision training, use UnslothVisionDataCollator
        trainer = self.SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processor.tokenizer,
            data_collator=self.UnslothVisionDataCollator(model, processor),
            args=training_args,
        )

        # Progress tracking: start actual training with WandB URL
        job_tracker.training(wandb_url)

        trainer.train()

        metrics = None
        if eval_dataset is not None:
            eval_results = trainer.evaluate()
            logging.info(f"Evaluation results: {eval_results}")
            metrics = eval_results

        # Use storage strategy to save model
        try:
            storage_strategy = StorageStrategyFactory.create_strategy(export)
            artifact = storage_strategy.save_model(
                model,
                processor,
                f"/tmp/{job_id}_adapter",
                {
                    "job_id": job_id,
                    "base_model_id": model_config.get("base_model_id"),
                    "use_unsloth": True,
                    "hf_repo_id": hf_repo_id,
                },
            )
        except Exception as e:
            logging.error(f"Failed to save model: {str(e)}", exc_info=True)
            raise e

        # 6. Cleanup
        storage_strategy.cleanup(artifact)
        del model, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        job_tracker.completed(artifact.remote_path, artifact.base_model_id, metrics)

        return {
            "adapter_path": artifact.remote_path,
            "base_model_id": artifact.base_model_id,
        }

    def _setup_unsloth_model(
        self, base_model_id: str, model_config: Dict[str, Any]
    ) -> Tuple[Any, Any]:
        """
        Setup Unsloth model and tokenizer for text training.
        Applies quantization and chat template configuration.

        Args:
            base_model_id: ID of the base model to use
            model_config: Model configuration parameters

        Returns:
            tuple: (model, tokenizer) - the initialized model and tokenizer
        """
        # Check supported models
        if base_model_id not in self.fourbit_models:
            raise ValueError(
                f"Model {base_model_id} is not supported. Supported models: {self.fourbit_models}"
            )

        # Load base model with Unsloth
        model, tokenizer = self.FastModel.from_pretrained(
            model_name=base_model_id,
            max_seq_length=model_config.get("max_seq_length", 2048),
            load_in_4bit=True,
            load_in_8bit=False,
            full_finetuning=True
            if model_config.get("method", "LoRA") == "Full"
            else False,
        )

        if model_config.get("method", "LoRA") != "Full":
            # Apply PEFT with Unsloth
            model = self.FastModel.get_peft_model(
                model,
                # target_modules=[
                #     "q_proj",
                #     "k_proj",
                #     "v_proj",
                #     "o_proj",
                #     "gate_proj",
                #     "up_proj",
                #     "down_proj",
                # ],
                target_modules="all-linear",
                r=model_config.get("lora_rank", 16),
                lora_alpha=model_config.get("lora_alpha", 16),
                lora_dropout=model_config.get("lora_dropout", 0.05),
                bias="none",
                random_state=3407,
            )
        else:
            logging.info(
                "Full fine-tuning selected, not applying LoRA. This will be slow and might result in GPU OOM"
            )

        # Setup chat template
        tokenizer = self.get_chat_template(
            tokenizer,
            chat_template="gemma-3",
        )

        return model, tokenizer

    def _setup_unsloth_vision_model(
        self, base_model_id: str, model_config: Dict[str, Any]
    ) -> Tuple[Any, Any]:
        """
        Setup Unsloth vision model and processor for training.
        Applies quantization and chat template configuration.

        Args:
            base_model_id: ID of the base model to use
            model_config: Model configuration parameters

        Returns:
            tuple: (model, processor) - the initialized model and processor
        """
        # Check if model is supported for vision
        if base_model_id not in self.fourbit_vision_models:
            raise ValueError(
                f"Model {base_model_id} is not supported for vision training. "
                f"Supported vision models: {self.fourbit_vision_models}"
            )

        # Load vision model with Unsloth
        model, processor = self.FastVisionModel.from_pretrained(
            base_model_id,
            load_in_4bit=True,
            use_gradient_checkpointing="unsloth",
        )

        # Apply PEFT for vision model
        if model_config.get("method", "LoRA") != "Full":
            model = self.FastVisionModel.get_peft_model(
                model,
                finetune_vision_layers=True,
                finetune_language_layers=True,
                finetune_attention_modules=True,
                finetune_mlp_modules=True,
                r=model_config.get("lora_rank", 16),
                lora_alpha=model_config.get("lora_alpha", 16),
                lora_dropout=model_config.get("lora_dropout", 0.05),
                bias="none",
                random_state=3407,
                use_rslora=False,
                loftq_config=None,
                target_modules="all-linear",
                modules_to_save=["lm_head", "embed_tokens"],
            )

        # Setup chat template for vision
        processor = self.get_chat_template(processor, "gemma-3")

        return model, processor

    def _prepare_unsloth_dataset(self, dataset: Any, tokenizer: Any) -> Any:
        """
        Prepare dataset for Unsloth training format.
        Standardizes and formats the dataset for Unsloth SFTTrainer.
        NOTE: This does not handle vision datasets because you should never use this on it

        Args:
            dataset: The dataset to prepare
            tokenizer: The tokenizer to use for encoding

        Returns:
            The prepared dataset ready for Unsloth SFTTrainer
        """
        # Standardize format
        dataset = self.standardize_data_formats(dataset)

        def formatting_prompts_func(examples):
            # examples["messages"] is a batch of message arrays -> we expect this to be well formatted in preprocessing
            # each convo is a list of message dicts: [{"role": "user", "content": "..."}, ...]
            # Handle both "messages" and "conversations" field names for compatibility
            if "messages" in examples:
                convos = examples["messages"]
            elif "conversations" in examples:
                convos = examples["conversations"]
            else:
                raise ValueError(
                    "Dataset must contain either 'messages' or 'conversations' field"
                )

            texts = [
                tokenizer.apply_chat_template(
                    convo, tokenize=False, add_generation_prompt=False
                ).removeprefix("<bos>")
                for convo in convos
            ]
            return {"text": texts}

        dataset = dataset.map(formatting_prompts_func, batched=True)
        return dataset

    def _setup_unsloth_vision_training_args(
        self, model_config: Dict[str, Any], job_id: str, report_to: str = "none"
    ) -> Any:
        """
        Setup training arguments for Unsloth vision SFTTrainer.
        Returns a SFTConfig object with vision-specific training hyperparameters.

        Args:
            model_config: Model configuration parameters
            job_id: Unique identifier for the training job
            report_to: Reporting destination for logs (e.g., "none", "wandb")

        Returns:
            SFTConfig: Configured training arguments for vision training
        """
        if self.torch_dtype == torch.bfloat16:
            bf16 = True
            fp16 = False
        else:
            bf16 = False
            fp16 = True

        # Vision-specific training arguments
        return self.SFTConfig(
            output_dir=f"/tmp/{job_id}",
            per_device_train_batch_size=model_config.get("batch_size", 4),
            per_device_eval_batch_size=model_config.get("batch_size", 4),
            gradient_accumulation_steps=model_config.get(
                "gradient_accumulation_steps", 4
            ),
            warmup_steps=5,
            num_train_epochs=model_config.get("epochs", 3),
            max_steps=model_config.get("max_steps", -1),
            learning_rate=model_config.get("learning_rate", 2e-4),
            packing=model_config.get("packing", True),
            fp16=fp16,
            bf16=bf16,
            optim="adamw_8bit",
            lr_scheduler_type="linear",
            weight_decay=0.01,
            save_strategy="epoch",
            logging_steps=10,
            report_to=report_to,
            # Vision-specific requirements
            remove_unused_columns=False,
            dataset_text_field="",  # NOTE: The docs specify this, for text-only this can be like "text"
            dataset_kwargs={"skip_prepare_dataset": True},
            dataset_num_proc=2,
            max_seq_length=model_config.get("max_seq_length", 2048),
        )

    def _setup_unsloth_training_args(
        self, model_config: Dict[str, Any], job_id: str, report_to: str = "none"
    ) -> Any:
        """
        Setup training arguments for Unsloth SFTTrainer.
        Returns a SFTConfig object with all training hyperparameters.

        Args:
            model_config: Model configuration parameters
            job_id: Unique identifier for the training job
            report_to: Reporting destination for logs (e.g., "none", "wandb")

        Returns:
            SFTConfig: Configured training arguments
        """
        if self.torch_dtype == torch.bfloat16:
            bf16 = True
            fp16 = False
        else:
            bf16 = False
            fp16 = True

        # Text training arguments
        return self.SFTConfig(
            dataset_text_field="text",  # this matches with the field created in _prepare_unsloth_dataset
            output_dir=f"/tmp/{job_id}",
            per_device_train_batch_size=model_config.get("batch_size", 4),
            per_device_eval_batch_size=model_config.get("batch_size", 4),
            gradient_accumulation_steps=model_config.get(
                "gradient_accumulation_steps", 4
            ),
            warmup_steps=5,
            num_train_epochs=model_config.get("epochs", 3),
            max_steps=model_config.get("max_steps", -1),
            learning_rate=model_config.get("learning_rate", 2e-4),
            packing=model_config.get("packing", True),
            fp16=fp16,
            bf16=bf16,
            optim="adamw_8bit",
            lr_scheduler_type="linear",
            weight_decay=0.01,
            save_strategy="epoch",
            logging_steps=10,
            report_to=report_to,
        )


class TrainingService:
    """
    Training service with factory method design pattern to support multiple providers.
    Allows easy extension to add new training providers in the future.
    Provides unified interface for running training jobs with different frameworks.
    """

    _providers: Dict[str, Any] = {
        "huggingface": HuggingFaceTrainingService,
        "unsloth": UnslothTrainingService,
    }

    @classmethod
    def from_provider(cls, provider: str) -> BaseTrainingService:
        """
        Create a training service instance for the specified provider.

        Args:
            provider: Name of the training provider ("huggingface" or "unsloth")

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
    def list_providers(cls) -> List[str]:
        """
        List all supported training providers.

        Returns:
            list: Names of supported providers
        """
        return list(cls._providers.keys())
