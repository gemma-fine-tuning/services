import logging
import torch
import io
from typing import Any, Tuple

from storage import storage_service
from base import BaseTrainingService
from utils import create_compute_metrics, preprocess_logits_for_metrics
from schema import TrainingConfig


class HuggingFaceTrainingService(BaseTrainingService):
    def __init__(self) -> None:
        # Import HF libraries only when service is instantiated
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
            BitsAndBytesConfig,
            AutoModelForImageTextToText,
            AutoProcessor,
        )
        from peft import LoraConfig, get_peft_model
        from trl import SFTTrainer, SFTConfig

        self.AutoTokenizer = AutoTokenizer
        self.AutoModelForCausalLM = AutoModelForCausalLM
        self.BitsAndBytesConfig = BitsAndBytesConfig
        self.AutoModelForImageTextToText = AutoModelForImageTextToText
        self.AutoProcessor = AutoProcessor
        self.LoraConfig = LoraConfig
        self.get_peft_model = get_peft_model
        self.SFTTrainer = SFTTrainer
        self.SFTConfig = SFTConfig

        self.supported_models = [
            "google/gemma-3-1b-it",
            "google/gemma-3-4b-it",
            "google/gemma-3-12b-it",
            "google/gemma-3n-E2B-it",
            "google/gemma-3n-E4B-it",
        ]

        # dtype based on GPU
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            self.torch_dtype = torch.bfloat16
        else:
            self.torch_dtype = torch.float16

    def _download_dataset(self, dataset_id: str) -> Tuple[Any, Any]:
        return storage_service.download_processed_dataset(dataset_id)

    def _setup_model(self, cfg: TrainingConfig) -> Tuple[Any, Any]:
        """
        Two criterias for model setup:

        1. If base_model_id is not 1b then use AutoModelForImageTextToText, else use AutoModelForCausalLM
        2. If modality is vision, use AutoProcessor otherwise use AutoTokenizer
        """
        base_model_id = cfg.base_model_id or "google/gemma-3-1b-it"
        if base_model_id not in self.supported_models:
            raise ValueError(
                f"Unsupported base model {cfg.base_model_id}. "
                f"Supported models: {self.supported_models}"
            )

        # Model kwargs for both text and vision
        model_kwargs = {
            "torch_dtype": self.torch_dtype,
            "device_map": "auto",
            "attn_implementation": "flash_attention_2" if cfg.use_fa2 else "eager",
        }

        # Add quantization for QLoRA
        if cfg.method == "QLoRA":
            model_kwargs["quantization_config"] = self.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.torch_dtype,
            )

        # Use AutoModelForImageTextToText or AutoModelForCausalLM based on model id, NOT modality!!
        if base_model_id == "google/gemma-3-1b-it":
            model = self.AutoModelForCausalLM.from_pretrained(
                base_model_id, **model_kwargs
            )
        else:
            model = self.AutoModelForImageTextToText.from_pretrained(
                base_model_id, **model_kwargs
            )

        # Setup tokenizer or preprocessor based on modality
        if cfg.modality == "vision":
            if base_model_id == "google/gemma-3-1b-it":
                raise ValueError(
                    "Gemma 3.1B does not support vision fine-tuning. Use Gemma 3.4B or larger."
                )

            # Vision models use AutoProcessor
            processor = self.AutoProcessor.from_pretrained(
                base_model_id, trust_remote_code=True
            )
            processor.tokenizer.padding_side = "right"

            return model, processor
        else:
            tokenizer = self.AutoTokenizer.from_pretrained(base_model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            return model, tokenizer

    def _prepare_dataset(
        self, train_ds: Any, eval_ds: Any, tokenizer: Any, cfg: TrainingConfig
    ) -> Tuple[Any, Any]:
        # Format datasets for Trainer
        train = self._prepare_hf_dataset(train_ds, tokenizer)
        eval = (
            self._prepare_hf_dataset(eval_ds, tokenizer)
            if eval_ds is not None
            else None
        )
        return train, eval

    def _apply_peft_if_needed(self, model: Any, cfg: TrainingConfig) -> Any:
        if cfg.method != "Full":
            lora_config = self.LoraConfig(
                lora_alpha=cfg.lora_alpha or 16,
                lora_dropout=cfg.lora_dropout or 0.05,
                r=cfg.lora_rank or 16,
                bias="none",
                target_modules="all-linear",
                task_type="CAUSAL_LM",
                modules_to_save=["lm_head", "embed_tokens"],
            )

            return self.get_peft_model(model, lora_config)

        logging.warning("No PEFT applied, using full model")
        return model

    def _build_training_args(
        self, cfg: TrainingConfig, job_id: str, report_to: str
    ) -> Any:
        # Mirror original SFTConfig builder from training_service
        # Determine fp16/bf16
        if self.torch_dtype == torch.bfloat16:
            bf16 = True
            fp16 = False
        else:
            bf16 = False
            fp16 = True

        args = self.SFTConfig(
            output_dir=f"/tmp/{job_id}",
            per_device_train_batch_size=cfg.batch_size,
            per_device_eval_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            warmup_steps=5,
            num_train_epochs=cfg.epochs,
            max_steps=cfg.max_steps or -1,
            learning_rate=cfg.learning_rate,
            packing=cfg.packing,
            fp16=fp16,
            bf16=bf16,
            optim="adamw_torch_fused",
            lr_scheduler_type=cfg.lr_scheduler_type or "linear",
            weight_decay=0.01,
            save_strategy=cfg.save_strategy or "epoch",
            eval_strategy=cfg.eval_strategy or "no",
            push_to_hub=False,
            logging_steps=cfg.logging_steps or 10,
            report_to=report_to,
            batch_eval_metrics=cfg.batch_eval_metrics or False,
        )

        # Add eval_steps if using steps strategy
        if cfg.eval_strategy == "steps":
            args.eval_steps = cfg.eval_steps or 50

        # Vision-specific arguments
        if cfg.modality == "vision":
            args.remove_unused_columns = False
            args.dataset_kwargs = {"skip_prepare_dataset": True}
            args.gradient_checkpointing = True
            args.gradient_checkpointing_kwargs = {"use_reentrant": False}
            args.dataset_text_field = ""  # dummy field for collator

        return args

    def _create_trainer(
        self,
        model: Any,
        tokenizer: Any,
        train_ds: Any,
        eval_ds: Any,
        args: Any,
        cfg: TrainingConfig,
    ) -> Any:
        return self.SFTTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=tokenizer,
            data_collator=self._create_vision_collate_fn(tokenizer)
            if cfg.modality == "vision"
            else None,
            compute_metrics=create_compute_metrics(
                cfg.compute_eval_metrics, cfg.batch_eval_metrics
            ),
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
            if cfg.compute_eval_metrics
            else None,
        )

    def _create_vision_collate_fn(self, processor):
        """Create vision collate function for HuggingFace vision training."""
        from PIL import Image

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

        return vision_collate_fn

    def _prepare_hf_dataset(self, dataset: Any, tokenizer: Any) -> Any:
        """Prepare dataset for HuggingFace SFTTrainer format."""

        def verify_format(examples):
            messages_batch = examples.get("messages", [])
            if not messages_batch:
                raise ValueError("Dataset is missing 'messages' field")
            return examples

        dataset = dataset.map(verify_format, batched=True)
        return dataset


class UnslothTrainingService(BaseTrainingService):
    def __init__(self) -> None:
        # importing unsloth is unused but necessary for optimization
        import unsloth  # noqa: F401
        from unsloth import FastModel, FastVisionModel, is_bfloat16_supported
        from unsloth.trainer import UnslothVisionDataCollator
        from unsloth.chat_templates import (
            get_chat_template,
            standardize_data_formats,
            train_on_responses_only,
        )
        from trl import SFTConfig, SFTTrainer

        self.FastModel = FastModel
        self.FastVisionModel = FastVisionModel
        self.is_bfloat16_supported = is_bfloat16_supported
        self.get_chat_template = get_chat_template
        self.standardize_data_formats = standardize_data_formats
        self.train_on_responses_only = train_on_responses_only
        self.UnslothVisionDataCollator = UnslothVisionDataCollator
        self.SFTConfig = SFTConfig
        self.SFTTrainer = SFTTrainer

        self.fourbit_models = [
            "unsloth/gemma-3-1b-it-unsloth-bnb-4bit",
            "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
            "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
            # "unsloth/gemma-3-27b-it-unsloth-bnb-4bit",
            "unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit",
            "unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit",
        ]

        self.torch_dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )

    # Hooks for Template Method:
    def _download_dataset(self, dataset_id: str) -> Tuple[Any, Any]:
        return storage_service.download_processed_dataset(dataset_id)

    def _setup_model(self, cfg: TrainingConfig) -> Tuple[Any, Any]:
        base_model_id = cfg.base_model_id or "unsloth/gemma-3-1b-it-unsloth-bnb-4bit"
        if base_model_id not in self.fourbit_models:
            raise ValueError(
                f"Unsupported base model {base_model_id}. "
                f"Supported models: {self.fourbit_models}"
            )

        # Choose Unsloth model based on modality
        if cfg.modality == "vision":
            if base_model_id == "unsloth/gemma-3-1b-it-unsloth-bnb-4bit":
                raise ValueError(
                    "Gemma 3.1B does not support vision fine-tuning. Use Gemma 3.4B or larger."
                )

            model, processor = self.FastVisionModel.from_pretrained(
                # Load in 4-bit for consistency with HuggingFace
                # NOTE: if you use unsloth you default to QLoRA lol
                base_model_id,
                load_in_4bit=True,
                max_seq_length=2048,  # From docs
                full_finetuning=True if cfg.method == "Full" else False,
            )
            # Setup chat template for vision models
            processor = self.get_chat_template(processor, "gemma-3")
            return model, processor
        else:
            model, tokenizer = self.FastModel.from_pretrained(
                base_model_id,
                load_in_4bit=True,
                max_seq_length=cfg.max_seq_length or 1024,
                full_finetuning=True if cfg.method == "Full" else False,
            )
            # model = self.prepare_model_for_kbit_training(model)
            # Setup chat template for text models
            tokenizer = self.get_chat_template(tokenizer, "gemma-3")
            return model, tokenizer

    def _prepare_dataset(
        self, train_ds: Any, eval_ds: Any, tokenizer: Any, cfg: TrainingConfig
    ) -> Tuple[Any, Any]:
        # Unsloth standardization for text; vision uses raw datasets
        if cfg.modality != "vision":
            train = self._prepare_unsloth_text_dataset(train_ds, tokenizer)
            eval = (
                self._prepare_unsloth_text_dataset(eval_ds, tokenizer)
                if eval_ds is not None
                else None
            )
            return train, eval
        else:
            return train_ds, eval_ds  # No standardization for vision datasets

    def _apply_peft_if_needed(self, model: Any, cfg: TrainingConfig) -> Any:
        # Method is either full or PEFT (LoRA or QLoRA)
        if cfg.method == "Full":
            logging.warning("No PEFT applied, using full model")
            return model

        if cfg.modality == "vision":
            model = self.FastVisionModel.get_peft_model(
                model,
                finetune_vision_layers=True,
                finetune_language_layers=True,
                finetune_attention_modules=True,
                finetune_mlp_modules=True,
                r=cfg.lora_rank,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                bias="none",
                random_state=3407,
                use_rslora=False,
                loftq_config=None,
                target_modules="all-linear",
                modules_to_save=["lm_head", "embed_tokens"],
            )
        else:
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
                r=cfg.lora_rank,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                bias="none",
                random_state=3407,
                use_rslora=False,
                loftq_config=None,
                target_modules="all-linear",
            )

        return model

    def _build_training_args(
        self, cfg: TrainingConfig, job_id: str, report_to: str
    ) -> Any:
        # Build Unsloth training args
        args = self.SFTConfig(
            output_dir=f"/tmp/{job_id}",
            dataset_text_field="text" if cfg.modality == "text" else "",
            per_device_train_batch_size=cfg.batch_size,
            per_device_eval_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            warmup_steps=5,
            num_train_epochs=cfg.epochs,
            max_steps=cfg.max_steps or -1,
            learning_rate=cfg.learning_rate,
            packing=cfg.packing,
            fp16=not self.is_bfloat16_supported(),
            bf16=self.is_bfloat16_supported(),
            optim="adamw_8bit",  # Unsloth default
            lr_scheduler_type=cfg.lr_scheduler_type or "linear",
            weight_decay=0.01,
            save_strategy=cfg.save_strategy or "epoch",
            eval_strategy=cfg.eval_strategy or "no",
            logging_steps=cfg.logging_steps or 10,
            report_to=report_to,
            batch_eval_metrics=cfg.batch_eval_metrics or False,
        )

        # Add eval_steps if using steps strategy
        if cfg.eval_strategy == "steps":
            args.eval_steps = cfg.eval_steps or 50

        # Add vision specific config
        if cfg.modality == "vision":
            args.remove_unused_columns = False
            args.dataset_kwargs = {}
            args.dataset_num_proc = 1
            args.max_length = 2048

        return args

    def _create_trainer(
        self,
        model: Any,
        tokenizer_or_processor: Any,
        train_ds: Any,
        eval_ds: Any,
        args: Any,
        cfg: TrainingConfig,
    ) -> Any:
        trainer = self.SFTTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=tokenizer_or_processor
            if cfg.modality == "text"
            else tokenizer_or_processor.tokenizer,  # for AutoProcessor
            data_collator=(
                # if modality is vision this is processor
                self.UnslothVisionDataCollator(model, tokenizer_or_processor)
                if cfg.modality == "vision"
                else None
            ),
            compute_metrics=create_compute_metrics(
                cfg.compute_eval_metrics, cfg.batch_eval_metrics
            ),
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
            if cfg.compute_eval_metrics
            else None,
        )

        # Apply response-only for text
        if cfg.modality == "text":
            trainer = self.train_on_responses_only(
                trainer,
                instruction_part="<start_of_turn>user\n",
                response_part="<start_of_turn>model\n",
            )

        return trainer

    def _prepare_unsloth_text_dataset(self, dataset: Any, tokenizer: Any) -> Any:
        """
        Prepare dataset for Unsloth text training format.
        Standardizes and formats the dataset with text field for Unsloth SFTTrainer.

        NOTE: Adds the "text" field only to text datasets because it is specified, otherwise we need to pass in a formatting func
        This is not required for vision.
        """
        # Standardize format first
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
                    "Dataset must contain 'messages' or 'conversations' field"
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


class TrainingService:
    _providers = {
        "huggingface": HuggingFaceTrainingService,
        "unsloth": UnslothTrainingService,
    }

    @classmethod
    def from_provider(cls, provider: str) -> BaseTrainingService:
        provider = provider.lower()
        if provider not in cls._providers:
            raise ValueError(f"Unsupported provider {provider}")
        return cls._providers[provider]()

    @classmethod
    def list_providers(cls):
        return list(cls._providers.keys())
