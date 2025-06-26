import os
import shutil
import logging
from transformers import pipeline
from peft import PeftModel
import torch
from services.model_storage import storage_service, CloudStoredModelMetadata
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template


class InferenceService:
    """
    Service for running inference with trained adapters
    Supports both standard Transformers and Unsloth models.
    For both case, this pipeline fetches the adapter and combines it with the base model,
    then runs text generation based on the provided prompt.

    NOTE: We need to implement a way to tell if the model is a standard Transformers model
    or an Unsloth model. This can be done by storing it in the metadata and read it when
    loading from GCS. The InferenceService will then decide which private method to call.

    TODO: We might want to adopt the factory design pattern here to align with training service
    """

    def __init__(self):
        self.export_bucket = os.environ.get(
            "GCS_EXPORT_BUCKET_NAME", "gemma-export-dev"
        )

    def run_inference(self, job_id: str, prompt: str) -> str:
        """
        Fetch adapter artifacts, run generation, and return output text.

        Args:
            job_id (str): Unique identifier for the training job
            prompt (str): Input text to generate a response for

        Returns:
            str: Generated text response from the model

        Raises:
            FileNotFoundError: If adapter artifacts or config are missing
            ValueError: If base model ID is not found in adapter config
        """
        # Download metadata and artifacts via storage manager
        meta: CloudStoredModelMetadata = storage_service.download_model(job_id)
        adapter_path = meta.local_dir
        model_id = meta.model_id
        use_unsloth = meta.use_unsloth

        if not model_id:
            raise ValueError("Base model ID not found in adapter config")

        if use_unsloth:
            output = self._run_inference_unsloth(model_id, adapter_path, prompt)
        else:
            output = self._run_inference_transformers(model_id, adapter_path, prompt)

        # Cleanup local artifacts
        shutil.rmtree(adapter_path, ignore_errors=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logging.info(f"Inference completed for job {job_id}")
        return output

    def _run_inference_transformers(
        self, model_id: str, adapter_path: str, prompt: str
    ) -> str:
        """Run inference using Transformers with trained adapter"""

        # Load model with adapter
        model = PeftModel.from_pretrained(model_id, adapter_path)
        tokenizer = None  # TODO: load matching transformer tokenizer

        # Create generation pipeline
        pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer, device_map="auto"
        )

        # Prepare prompt
        chat_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        stop_tokens = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<end_of_turn>"),
        ]

        # Generate
        out = pipe(
            chat_prompt,
            max_new_tokens=256,
            do_sample=False,
            top_k=50,
            eos_token_id=stop_tokens,
        )
        text = out[0]["generated_text"][len(chat_prompt) :].strip()

        return text

    def _run_inference_unsloth(
        self, model_id: str, adapter_path: str, prompt: str
    ) -> str:
        """Run inference using Unsloth with trained adapter"""

        # Load model and tokenizer via Unsloth
        model, tokenizer = FastModel.from_pretrained(
            model_name=model_id,
            max_seq_length=2048,
            load_in_4bit=True,
        )

        # Prepare chat template
        tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

        # Generate output
        outputs = model.generate(
            **tokenizer([text], return_tensors="pt").to("cuda"),
            max_new_tokens=256,
            temperature=1.0,
            top_p=0.95,
            top_k=64,
        )
        return tokenizer.batch_decode(outputs)[0].strip()


# default service instance
inference_service = InferenceService()
run_inference = inference_service.run_inference
