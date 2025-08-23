import logging
import torch
import io
from typing import Any, Tuple
from datasets import Dataset

from storage import storage_service
from base import BaseTrainingService
from rewards import load_reward_functions_from_config

# from utils import create_compute_metrics, preprocess_logits_for_metrics
from utils import create_compute_metrics
from schema import TrainingConfig


def __build_shared_training_args(
    trainer_type: str,
    cfg: TrainingConfig,
    job_id: str,
    report_to: str,
    provider_specific_args: dict,
    config_classes: dict,
) -> Any:
    """
    Shared logic for building complete training configuration across providers.

    This function handles all common logic including:
    - Base training arguments
    - Trainer-specific configuration
    - Evaluation setup
    - Vision-specific configuration

    Args:
        trainer_type: Type of trainer (sft, grpo, dpo)
        cfg: Training configuration
        job_id: Job identifier
        report_to: Reporting destination
        provider_specific_args: Provider-specific base arguments
        config_classes: Dict with trainer config classes (SFTConfig, GRPOConfig, DPOConfig)

    Returns:
        Configured training arguments object ready for trainer
    """
    hyperparam = cfg.hyperparameters
    evaluation = cfg.eval_config

    # Common base arguments
    base_args = {
        "output_dir": f"/tmp/{job_id}",
        "per_device_train_batch_size": hyperparam.batch_size,
        "gradient_accumulation_steps": hyperparam.gradient_accumulation_steps,
        "warmup_steps": 5,
        "learning_rate": hyperparam.learning_rate,
        "lr_scheduler_type": hyperparam.lr_scheduler_type or "linear",
        "weight_decay": 0.01,
        "save_strategy": hyperparam.save_strategy or "epoch",
        "push_to_hub": False,
        "logging_steps": hyperparam.logging_steps or 10,
        "report_to": report_to,
    }

    # Merge with provider-specific args
    base_args.update(provider_specific_args)

    # Trainer-specific configuration
    if trainer_type == "sft":
        trainer_args = {
            **base_args,
            "num_train_epochs": hyperparam.epochs,
            "max_steps": hyperparam.max_steps or -1,
            "packing": hyperparam.packing,
        }
        args = config_classes["sft"](**trainer_args)

    elif trainer_type == "grpo":
        # GRPO-specific parameters
        max_prompt_length = hyperparam.max_prompt_length
        max_length = hyperparam.max_seq_length
        max_completion_length = max_length - max_prompt_length

        # This makes sure the entire image token is sent for vision
        if cfg.modality == "vision":
            max_prompt_length = None
            max_completion_length = max_length

        trainer_args = {
            **base_args,
            "max_steps": hyperparam.max_steps or 50,
            "num_generations": hyperparam.num_generations or 4,
            "max_prompt_length": max_prompt_length,
            "max_completion_length": max_completion_length,
            "max_grad_norm": hyperparam.max_grad_norm or 0.1,
            "adam_beta1": hyperparam.adam_beta1 or 0.9,
            "adam_beta2": hyperparam.adam_beta2 or 0.99,
            "warmup_ratio": hyperparam.warmup_ratio or 0.1,
            "remove_unused_columns": False,  # NOTE: MUST HAVE THIS to access the additional columns
            # Being worked on by unsloth right now add: "vllm>=0.10.0" later
            # GRPO is online method and vLLM is much faster at inference
            # "use_vllm": True,
            # NOTE: We cannot user "server" because there's only one GPU on cloud run for now...
            # Otherwise we will start another vLLM inference server
            # "vllm_mode": "colocate",
        }
        args = config_classes["grpo"](**trainer_args)

    elif trainer_type == "dpo":
        trainer_args = {
            **base_args,
            "num_train_epochs": hyperparam.epochs,
            "max_steps": hyperparam.max_steps or -1,
            "beta": hyperparam.beta or 0.1,
            "max_prompt_length": hyperparam.max_prompt_length
            if cfg.modality != "vision"
            else None,
            # max_length is max_completion_length + max_prompt_length
            "max_length": hyperparam.max_length,
        }
        args = config_classes["dpo"](**trainer_args)

    else:
        raise ValueError(f"Unsupported trainer type: {trainer_type}")

    # Set eval related fields if present (common to all trainers)
    if evaluation:
        args.eval_strategy = evaluation.eval_strategy or "no"
        if evaluation.eval_strategy == "steps":
            args.eval_steps = evaluation.eval_steps
        args.per_device_eval_batch_size = cfg.hyperparameters.batch_size
        args.eval_accumulation_steps = cfg.hyperparameters.gradient_accumulation_steps

        # Set eval precision based on provider-specific args
        if "fp16_full_eval" in provider_specific_args:
            args.fp16_full_eval = provider_specific_args["fp16"]
        if "bf16_full_eval" in provider_specific_args:
            args.bf16_full_eval = provider_specific_args["bf16"]

        # Set batch eval metrics if supported by config type
        if hasattr(args, "batch_eval_metrics"):
            args.batch_eval_metrics = evaluation.batch_eval_metrics

    # Vision-specific arguments (applies to all trainers when modality is vision)
    if cfg.modality == "vision":
        args.remove_unused_columns = False
        args.gradient_checkpointing = True
        args.gradient_checkpointing_kwargs = {"use_reentrant": False}

        # SFT-specific vision settings
        if hasattr(args, "dataset_kwargs"):
            if trainer_type == "sft":
                # HuggingFace needs skip_prepare_dataset for vision
                args.dataset_kwargs = {"skip_prepare_dataset": True}
            else:
                # Unsloth uses empty dict
                args.dataset_kwargs = {}

        if hasattr(args, "dataset_text_field"):
            args.dataset_text_field = ""  # dummy field for collator

        # Unsloth-specific vision settings
        if hasattr(args, "dataset_num_proc"):
            args.dataset_num_proc = 1
        if (
            hasattr(args, "max_length") and trainer_type != "dpo"
        ):  # DPO already sets max_length
            args.max_length = 2048

    return args


def __reformat_vision_dataset(example, processor):
    """
    Convert vision datasets from complete ChatML format to the format expected by GRPO/DPO trainers.
    This extracts images to a separate column and applies chat template to create prompt strings.
    This can also be done by formatting dataset differently in preprocessing, but post-processing is what we've chosen.

    NOTE: We do NOT recommend using SFTTrainer with this YET due to multi-image support requiring data collator!!!

    Automatically detects format:
    - Prompt-only format (for GRPO): {"prompt": [...], "answer": "...", "reasoning": "...", ...}
      Output: {"prompt": "templated_string", "image": PIL.Image, "answer": "...", "reasoning": "...", ...}

    - Preference format (for DPO): {"prompt": [...], "chosen": [...], "rejected": [...]}
      Output: {"prompt": "templated_string", "chosen": "templated_string", "rejected": "templated_string", "image": PIL.Image}
    """
    from PIL import Image

    # Use the tokenizer directly for Unsloth (processor.tokenizer for AutoProcessor)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    def extract_images_and_create_text_only_messages(messages):
        """Extract images and create text-only version of messages for chat template"""
        images = []
        text_only_messages = []

        for msg in messages:
            content = msg.get("content", [])
            if isinstance(content, list):
                text_items = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "image":
                            img_data = item.get("image")
                            if img_data:
                                # Handle PIL Image objects
                                if isinstance(img_data, Image.Image):
                                    images.append(img_data.convert("RGB"))
                                # Handle HuggingFace dataset format: {"bytes": ..., "path": null}
                                elif isinstance(img_data, dict) and "bytes" in img_data:
                                    image_bytes = img_data["bytes"]
                                    if isinstance(image_bytes, (bytes, bytearray)):
                                        pil_image = Image.open(io.BytesIO(image_bytes))
                                        images.append(pil_image.convert("RGB"))
                        elif item.get("type") == "text":
                            text_items.append(item.get("text", ""))

                # Combine all text items for this message
                if text_items:
                    text_only_messages.append(
                        {"role": msg["role"], "content": " ".join(text_items)}
                    )
            else:
                # Handle backward compatibility - content is a string
                text_only_messages.append({"role": msg["role"], "content": content})

        return images, text_only_messages

    result = {}

    # Auto-detect format based on available fields
    if "prompt" in example and "chosen" in example and "rejected" in example:
        # Preference format (DPO)
        prompt_messages = example["prompt"]
        images, text_only_prompt = extract_images_and_create_text_only_messages(
            prompt_messages
        )

        # Create full conversations for chosen and rejected
        chosen_conversation = text_only_prompt + example["chosen"]
        rejected_conversation = text_only_prompt + example["rejected"]

        # Apply chat template to each
        result["prompt"] = tokenizer.apply_chat_template(
            text_only_prompt, add_generation_prompt=True, tokenize=False
        )
        result["chosen"] = tokenizer.apply_chat_template(
            chosen_conversation, add_generation_prompt=False, tokenize=False
        )
        result["rejected"] = tokenizer.apply_chat_template(
            rejected_conversation, add_generation_prompt=False, tokenize=False
        )

        if images:
            result["image"] = images[0]  # DPO typically uses single image

    elif "prompt" in example:
        # Prompt-only format (GRPO)
        prompt_messages = example["prompt"]
        images, text_only_prompt = extract_images_and_create_text_only_messages(
            prompt_messages
        )

        # Apply chat template to create prompt string
        prompt_text = tokenizer.apply_chat_template(
            text_only_prompt, add_generation_prompt=True, tokenize=False
        )

        result["prompt"] = prompt_text
        if images:
            result["image"] = images[0]  # GRPO typically uses single image

        # Copy over all other fields (answer, reasoning, etc.)
        for key, value in example.items():
            if key != "prompt":
                result[key] = value

    else:
        raise ValueError(
            "Example must contain 'prompt' field (for GRPO) or 'prompt', 'chosen', and 'rejected' fields (for DPO)"
        )

    return result


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
        from trl import (
            SFTTrainer,
            SFTConfig,
            GRPOTrainer,
            GRPOConfig,
            DPOTrainer,
            DPOConfig,
        )

        self.AutoTokenizer = AutoTokenizer
        self.AutoModelForCausalLM = AutoModelForCausalLM
        self.BitsAndBytesConfig = BitsAndBytesConfig
        self.AutoModelForImageTextToText = AutoModelForImageTextToText
        self.AutoProcessor = AutoProcessor
        self.LoraConfig = LoraConfig
        self.get_peft_model = get_peft_model
        self.SFTTrainer = SFTTrainer
        self.SFTConfig = SFTConfig
        self.GRPOTrainer = GRPOTrainer
        self.GRPOConfig = GRPOConfig
        self.DPOTrainer = DPOTrainer
        self.DPOConfig = DPOConfig

        # Support both IT and PT models
        # no official quantised so we apply them later with bnb
        self.supported_models = [
            "google/gemma-3-1b-pt",
            "google/gemma-3-1b-it",
            "google/gemma-3-4b-pt",
            "google/gemma-3-4b-it",
            "google/gemma-3-12b-pt",
            "google/gemma-3-12b-it",
            "google/gemma-3n-E2B",
            "google/gemma-3n-E2B-it",
            "google/gemma-3n-E4B",
            "google/gemma-3n-E4B-it",
            "google/gemma-3-270m",
            "google/gemma-3-270m-it",
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

        # Load the model with proper quantisation if required
        # NOTE: This can be easily extended to support other quantization methods
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

    def _prepare_dataset_if_needed(
        self,
        train_ds: Dataset,
        eval_ds: Dataset,
        tokenizer_or_processor: Any,
        modality: str,
        trainer_type: str,
    ) -> Tuple[Dataset, Dataset]:
        """
        Optionally format dataset depending on modality and trainer type.
        Otherwise you should provide a data collator that is used by the trainer to do reformatting internally.
        Or you can provide a formatting_func in the trainer.
        """
        if modality == "vision" and trainer_type in ["dpo", "grpo"]:
            # Convert vision datasets to the format expected by GRPO/DPO trainers and they will handle processing
            # NOTE: do not use batching here since it will make each column a list and break the format function
            train_ds = train_ds.map(
                lambda example: __reformat_vision_dataset(
                    example, tokenizer_or_processor
                )
            )
            if eval_ds is not None:
                eval_ds = eval_ds.map(
                    lambda example: __reformat_vision_dataset(
                        example, tokenizer_or_processor
                    )
                )

        return train_ds, eval_ds

    def _apply_peft_if_needed(self, model: Any, cfg: TrainingConfig) -> Any:
        if cfg.method != "Full":
            lora_config = self.LoraConfig(
                lora_alpha=cfg.hyperparameters.lora_alpha,
                lora_dropout=cfg.hyperparameters.lora_dropout,
                r=cfg.hyperparameters.lora_rank,
                bias="none",
                target_modules="all-linear",
                task_type="CAUSAL_LM",
                modules_to_save=["lm_head", "embed_tokens"],
            )

            # NOTE: This can be easily extended to support PEFT other than LoRA

            return self.get_peft_model(model, lora_config)

        logging.warning("No PEFT applied, using full model")
        return model

    def _build_training_args(
        self, trainer_type: str, cfg: TrainingConfig, job_id: str, report_to: str
    ) -> Any:
        # Provider-specific arguments for HuggingFace
        # Determine fp16/bf16
        if self.torch_dtype == torch.bfloat16:
            bf16 = True
            fp16 = False
        else:
            bf16 = False
            fp16 = True

        provider_specific_args = {
            "fp16": fp16,  # shared with fp16_full_eval
            "bf16": bf16,  # shared with bf16_full_eval
            "optim": "adamw_torch_fused",
        }

        # Config classes for HuggingFace
        config_classes = {
            "sft": self.SFTConfig,
            "grpo": self.GRPOConfig,
            "dpo": self.DPOConfig,
        }

        # Get complete configured training arguments
        return __build_shared_training_args(
            trainer_type, cfg, job_id, report_to, provider_specific_args, config_classes
        )

    def _create_trainer(
        self,
        model: Any,
        tokenizer_or_processor: Any,  # either AutoTokenizer or AutoProcessor
        train_ds: Any,
        eval_ds: Any,
        args: Any,
        trainer_type: str,
        cfg: TrainingConfig,
    ) -> Any:
        # Common trainer arguments
        base_trainer_args = {
            "model": model,
            "args": args,
            "train_dataset": train_ds,
            "eval_dataset": eval_ds,
            "processing_class": tokenizer_or_processor,
        }

        if trainer_type == "sft":
            # For SFT Trainer we use custom collator to support multiple images and support computing metrics
            return self.SFTTrainer(
                **base_trainer_args,
                data_collator=self._create_vision_collate_fn(tokenizer_or_processor)
                if cfg.modality == "vision"
                else None,
                compute_metrics=create_compute_metrics(
                    cfg.eval_config.compute_eval_metrics,
                    cfg.eval_config.batch_eval_metrics,
                )
                if cfg.eval_config
                else None,
            )
        elif trainer_type == "grpo":
            # GRPO requires reward functions
            reward_funcs = load_reward_functions_from_config(cfg.reward_config)
            # NOTE: GRPO Trainer doesn't allow for data collator so we need to reformat the dataset manually above
            return self.GRPOTrainer(
                **base_trainer_args,
                reward_funcs=reward_funcs,
            )
        elif trainer_type == "dpo":
            # NOTE: DPO Trainer doesn't need collator for vision, we reformat the dataset manually
            return self.DPOTrainer(**base_trainer_args)
        else:
            raise ValueError(f"Unsupported trainer type: {trainer_type}")

    def _create_vision_collate_fn(self, processor):
        """
        Create vision collate function for HuggingFace vision training.
        NOTE: There is no built in hugging face vision collator unlike Unsloth
        """
        from PIL import Image

        def vision_collate_fn(examples):
            """
            Collate function for vision datasets that are already in ChatML format with images.
            1. First apply chat template to all examples in this batch (converts from conversational to standard format)
            2. Extract images from type:image fields and apply them to processors in the correct order
            3. Do some postprocessing on the tokens and labels and return the batch with text and images
            """

            # Handle different dataset types
            # NOTE: This technically works but GRPOTrainer does not support data collator
            def get_messages_for_template(example):
                if "messages" in example:
                    # Language modeling format
                    return example["messages"]
                elif "prompt" in example:
                    if "chosen" in example and "rejected" in example:
                        # Preference format - for DPO we need prompt + chosen + rejected
                        return (
                            example["prompt"] + example["chosen"] + example["rejected"]
                        )
                    else:
                        # Prompt-only format
                        return example["prompt"]
                else:
                    raise ValueError(
                        "Example must contain 'messages' or 'prompt' field"
                    )

            texts = [
                processor.apply_chat_template(
                    get_messages_for_template(example),
                    tokenize=True,
                    add_generation_prompt=False,
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
                extract_images_from_messages(get_messages_for_template(example))
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
        from trl import (
            SFTConfig,
            SFTTrainer,
            GRPOConfig,
            GRPOTrainer,
            DPOConfig,
            DPOTrainer,
        )

        self.FastModel = FastModel
        self.FastVisionModel = FastVisionModel
        self.is_bfloat16_supported = is_bfloat16_supported
        self.get_chat_template = get_chat_template
        self.standardize_data_formats = standardize_data_formats
        self.train_on_responses_only = train_on_responses_only
        self.UnslothVisionDataCollator = UnslothVisionDataCollator
        self.SFTConfig = SFTConfig
        self.SFTTrainer = SFTTrainer
        self.GRPOConfig = GRPOConfig
        self.GRPOTrainer = GRPOTrainer
        self.DPOConfig = DPOConfig
        self.DPOTrainer = DPOTrainer

        # Haven't tested 270M will add that later
        self.supported_models = [
            "unsloth/gemma-3-1b-pt",
            "unsloth/gemma-3-1b-it",
            "unsloth/gemma-3-4b-pt",
            "unsloth/gemma-3-4b-it",
            "unsloth/gemma-3-12b-pt",
            "unsloth/gemma-3-12b-it",
            "unsloth/gemma-3n-E4B",
            "unsloth/gemma-3n-E4B-it",
            "unsloth/gemma-3n-E2B",
            "unsloth/gemma-3n-E2B-it",
            # Can't find 240m pt with unsloth
            "unsloth/gemma-3-270m-it",
            "unsloth/gemma-3-1b-pt-unsloth-bnb-4bit",
            "unsloth/gemma-3-1b-it-unsloth-bnb-4bit",
            "unsloth/gemma-3-4b-pt-unsloth-bnb-4bit",
            "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
            "unsloth/gemma-3-12b-pt-unsloth-bnb-4bit",
            "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
            # "unsloth/gemma-3-27b-pt-unsloth-bnb-4bit",
            # "unsloth/gemma-3-27b-it-unsloth-bnb-4bit",
            "unsloth/gemma-3n-E4B-unsloth-bnb-4bit",
            "unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit",
            "unsloth/gemma-3n-E2B-unsloth-bnb-4bit",
            "unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit",
            # "unsloth/gemma-3-240m-pt-unsloth-bnb-4bit",  # No PT version
            "unsloth/gemma-3-270m-it-unsloth-bnb-4bit",
        ]

    # Hooks for Template Method:
    def _download_dataset(self, dataset_id: str) -> Tuple[Any, Any]:
        return storage_service.download_processed_dataset(dataset_id)

    def _setup_model(self, cfg: TrainingConfig) -> Tuple[Any, Any]:
        base_model_id = cfg.base_model_id or "unsloth/gemma-3-1b-it-unsloth-bnb-4bit"
        if base_model_id not in self.supported_models:
            raise ValueError(
                f"Unsupported base model {base_model_id}. "
                f"Supported models: {self.supported_models}"
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
                max_seq_length=cfg.max_seq_length,
                full_finetuning=True if cfg.method == "Full" else False,
            )
            # Setup chat template for text models
            tokenizer = self.get_chat_template(tokenizer, "gemma-3")
            return model, tokenizer

    def _prepare_dataset_if_needed(
        self,
        train_ds: Dataset,
        eval_ds: Dataset,
        tokenizer_or_processor: Any,
        modality: str,
        trainer_type: str,
    ) -> Tuple[Dataset, Dataset]:
        # Unsloth standardization for text; vision uses raw datasets
        if modality == "text":
            train_ds = self._prepare_unsloth_text_dataset(
                train_ds, tokenizer_or_processor
            )
            eval_ds = (
                self._prepare_unsloth_text_dataset(eval_ds, tokenizer_or_processor)
                if eval_ds is not None
                else None
            )

        elif modality == "vision" and trainer_type in ["dpo", "grpo"]:
            # Convert vision datasets to the format expected by GRPO/DPO trainers
            train_ds = train_ds.map(
                lambda example: __reformat_vision_dataset(
                    example, tokenizer_or_processor
                )
            )
            if eval_ds is not None:
                eval_ds = eval_ds.map(
                    lambda example: __reformat_vision_dataset(
                        example, tokenizer_or_processor
                    )
                )

        return train_ds, eval_ds

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
                r=cfg.hyperparameters.lora_rank,
                lora_alpha=cfg.hyperparameters.lora_alpha,
                lora_dropout=cfg.hyperparameters.lora_dropout,
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
                r=cfg.hyperparameters.lora_rank,
                lora_alpha=cfg.hyperparameters.lora_alpha,
                lora_dropout=cfg.hyperparameters.lora_dropout,
                bias="none",
                random_state=3407,
                use_rslora=False,
                loftq_config=None,
                target_modules="all-linear",
            )

        return model

    def _build_training_args(
        self, trainer_type: str, cfg: TrainingConfig, job_id: str, report_to: str
    ) -> Any:
        # Provider-specific arguments for Unsloth
        provider_specific_args = {
            "fp16": not self.is_bfloat16_supported(),
            "bf16": self.is_bfloat16_supported(),
            "optim": "adamw_8bit",  # Unsloth default
        }

        # Config classes for Unsloth
        config_classes = {
            "sft": self.SFTConfig,
            "grpo": self.GRPOConfig,
            "dpo": self.DPOConfig,
        }

        # Get complete configured training arguments
        return __build_shared_training_args(
            trainer_type, cfg, job_id, report_to, provider_specific_args, config_classes
        )

    def _create_trainer(
        self,
        model: Any,
        tokenizer_or_processor: Any,
        train_ds: Any,
        eval_ds: Any,
        args: Any,
        trainer_type: str,
        cfg: TrainingConfig,
    ) -> Any:
        # Common trainer arguments
        base_trainer_args = {
            "model": model,
            "args": args,
            "train_dataset": train_ds,
            "eval_dataset": eval_ds,
            "processing_class": tokenizer_or_processor
            if cfg.modality == "text"
            else tokenizer_or_processor.tokenizer,  # for AutoProcessor
        }

        if trainer_type == "sft":
            # Same as hugging face except we use their built-in collator
            trainer = self.SFTTrainer(
                **base_trainer_args,
                data_collator=(
                    # if modality is vision this is processor
                    self.UnslothVisionDataCollator(model, tokenizer_or_processor)
                    if cfg.modality == "vision"
                    else None
                ),
                compute_metrics=create_compute_metrics(
                    cfg.eval_config.compute_eval_metrics,
                    cfg.eval_config.batch_eval_metrics,
                )
                if cfg.eval_config
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

        elif trainer_type == "grpo":
            # GRPO requires reward functions and processing_class
            reward_funcs = load_reward_functions_from_config(cfg.reward_config)
            return self.GRPOTrainer(
                **base_trainer_args,
                reward_funcs=reward_funcs,
            )
        elif trainer_type == "dpo":
            return self.DPOTrainer(**base_trainer_args)
        else:
            raise ValueError(f"Unsupported trainer type: {trainer_type}")

    def _prepare_unsloth_text_dataset(self, dataset: Any, tokenizer: Any) -> Any:
        """
        Prepare dataset for Unsloth text training format.
        Standardizes and formats the dataset with text field for Unsloth SFTTrainer.

        NOTE: Adds the "text" field only to text datasets because it is specified,
        otherwise we need to pass in a formatting func
        This is not required for vision because we have data collator for vision.

        Somehow unsloth requires this otherwise it breaks for random reasons,
        the original SFTTrainer doesn't care if there's a formatting func etc but the UnslothSFTTrainer does...
        """
        # Standardize format first
        dataset = self.standardize_data_formats(dataset)

        def formatting_prompts_func(examples):
            # Handle different dataset types
            if "messages" in examples:
                # Language modeling format
                convos = examples["messages"]
                texts = [
                    tokenizer.apply_chat_template(
                        convo, tokenize=False, add_generation_prompt=False
                    ).removeprefix("<bos>")
                    for convo in convos
                ]
                return {"text": texts}

            elif "prompt" in examples:
                # Check if it's preference format (has chosen/rejected)
                if "chosen" in examples and "rejected" in examples:
                    # Preference format - combine prompt + chosen + rejected for DPO
                    prompts = examples["prompt"]
                    chosen = examples["chosen"]
                    rejected = examples["rejected"]

                    texts = []
                    for prompt, chosen_response, rejected_response in zip(
                        prompts, chosen, rejected
                    ):
                        # Combine prompt + chosen + rejected into one conversation
                        combined_messages = prompt + chosen_response + rejected_response
                        text = tokenizer.apply_chat_template(
                            combined_messages,
                            tokenize=False,
                            add_generation_prompt=False,
                        ).removeprefix("<bos>")
                        texts.append(text)

                    return {"text": texts}
                else:
                    # Prompt-only format - just use the prompt
                    prompts = examples["prompt"]
                    texts = [
                        tokenizer.apply_chat_template(
                            prompt, tokenize=False, add_generation_prompt=False
                        ).removeprefix("<bos>")
                        for prompt in prompts
                    ]
                    return {"text": texts}

            else:
                raise ValueError("Dataset must contain 'messages' or 'prompt' field")

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
