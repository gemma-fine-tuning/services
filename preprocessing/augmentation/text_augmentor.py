import logging
from typing import List, Dict, Any
from abc import ABC, abstractmethod
from .eda import eda
import random

logger = logging.getLogger(__name__)


class BaseAugmentor(ABC):
    """
    Base class for all text augmentation techniques
    Whenever the augment metho is called, augmentation is applied 100% of the time.
    We do not use probabilities here because the upstream usage should ensures that
    the final desired dataset size is achieved by sampling random data from the dataset
    and applying a few variations of augmentation to it and selecting the proper variation.
    """

    def __init__(self):
        pass

    @abstractmethod
    def augment(self, text: str) -> str:
        """Apply augmentation to text"""
        pass


class EDAugmentor(BaseAugmentor):
    """
    Easy Data Augmentation using the original EDA implementation.
    This is the fastest and most lightweight augmentation technique.

    Source: https://arxiv.org/pdf/1901.11196
    """

    def __init__(
        self,
        alpha_sr=0.1,
        alpha_ri=0.1,
        alpha_rs=0.1,
        p_rd=0.1,
        num_aug=1,
    ):
        super().__init__()
        self.alpha_sr = alpha_sr  # synonym replacement
        self.alpha_ri = alpha_ri  # random insertion
        self.alpha_rs = alpha_rs  # random swap
        self.p_rd = p_rd  # random deletion
        self.num_aug = num_aug  # number of augmented sentences per original

    def augment(self, text: str) -> str:
        """Apply EDA augmentation"""
        if not text.strip():
            return text

        try:
            # Use the original EDA function
            augmented_sentences = eda(
                text,
                alpha_sr=self.alpha_sr,
                alpha_ri=self.alpha_ri,
                alpha_rs=self.alpha_rs,
                p_rd=self.p_rd,
                num_aug=self.num_aug,
            )

            # Return the first augmented sentence (excluding original which is last)
            if len(augmented_sentences) > 1:
                return augmented_sentences[0]
            return text

        except Exception as e:
            logger.warning(f"EDA augmentation failed: {e}")
            return text


class BackTranslationAugmentor(BaseAugmentor):
    """
    Back translation using Helsinki-NLP models via transformers
    Source: https://huggingface.co/docs/transformers/en/model_doc/marian#transformers.MarianMTModel

    The model class is quite lightweight (just 300MB each model) and supports 1000+ language pairs.
    This woul require at least 600MB of disk space for the two models. The models are loaded lazily only if use.

    The backtranslation technique is proposed in https://arxiv.org/pdf/2106.04681

    NOTE: Back translation does not guarantee that the output will be different from the input!
    """

    def __init__(self, intermediate_lang: str = "fr"):
        super().__init__()
        self.intermediate_lang = intermediate_lang
        self._forward_model = None
        self._backward_model = None
        self._forward_tokenizer = None
        self._backward_tokenizer = None

    def _load_models(self):
        """Lazy load translation models"""
        if self._forward_model is None:
            try:
                # import done here to avoid unnecessary dependencies
                from transformers import MarianMTModel, MarianTokenizer

                # English to intermediate language
                forward_model_name = f"Helsinki-NLP/opus-mt-en-{self.intermediate_lang}"
                self._forward_tokenizer = MarianTokenizer.from_pretrained(
                    forward_model_name
                )
                self._forward_model = MarianMTModel.from_pretrained(forward_model_name)

                # Intermediate language back to English
                backward_model_name = (
                    f"Helsinki-NLP/opus-mt-{self.intermediate_lang}-en"
                )
                self._backward_tokenizer = MarianTokenizer.from_pretrained(
                    backward_model_name
                )
                self._backward_model = MarianMTModel.from_pretrained(
                    backward_model_name
                )

                logger.info(
                    f"Loaded back-translation models for {self.intermediate_lang}"
                )

            except ImportError:
                logger.error(
                    "transformers library not installed. Install with: pip install transformers torch"
                )
                raise
            except Exception as e:
                logger.error(f"Failed to load translation models: {e}")
                raise

    def augment(self, text: str) -> str:
        """Apply back translation augmentation"""
        if not text.strip():
            return text

        try:
            self._load_models()

            # Translate to intermediate language
            forward_inputs = self._forward_tokenizer(
                text, return_tensors="pt", padding=True, truncation=True, max_length=512
            )
            forward_outputs = self._forward_model.generate(
                **forward_inputs, max_length=512, num_beams=4, early_stopping=True
            )
            intermediate_text = self._forward_tokenizer.decode(
                forward_outputs[0], skip_special_tokens=True
            )

            # Translate back to English
            backward_inputs = self._backward_tokenizer(
                intermediate_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            backward_outputs = self._backward_model.generate(
                **backward_inputs, max_length=512, num_beams=4, early_stopping=True
            )
            back_translated = self._backward_tokenizer.decode(
                backward_outputs[0], skip_special_tokens=True
            )

            if back_translated.strip():
                return back_translated.strip()
            else:
                logger.warning(
                    "Back translation returned empty result, returning original text."
                )
                return text.strip()

        except Exception as e:
            logger.warning(f"Back translation failed: {e}")
            return text


class ParaphraseAugmentor(BaseAugmentor):
    """
    Paraphrasing using T5 models specialized for paraphrasing tasks.
    Source: https://huggingface.co/humarin/chatgpt_paraphraser_on_T5_base
    This model requires about 1GB of disk space and is loaded lazily.
    """

    def __init__(self, model_name: str = "humarin/chatgpt_paraphraser_on_T5_base"):
        super().__init__()
        self.model_name = model_name
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """Lazy load paraphrasing model"""
        if self._model is None:
            try:
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

                # Load tokenizer and model
                # NOTE: If we do have GPU for this service send them .to(device) but now I assume we do preprocessing on CPU
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
                logger.info(f"Loaded paraphrasing model: {self.model_name}")

            except ImportError:
                logger.error(
                    "transformers library not installed. Install with: pip install transformers torch"
                )
                raise
            except Exception as e:
                logger.error(f"Failed to load paraphrasing model: {e}")
                raise

    def _paraphrase(
        self,
        question: str,
        num_beams=5,
        num_beam_groups=5,
        num_return_sequences=5,
        repetition_penalty=10.0,
        diversity_penalty=3.0,
        no_repeat_ngram_size=2,
        temperature=0.7,
        max_length=128,
    ) -> List[str]:
        """
        This method references the example usage from https://huggingface.co/humarin/chatgpt_paraphraser_on_T5_base
        Try not to modify the parameters since they are providers by the original author and is most likely to work best.
        """
        input_ids = self._tokenizer(
            f"paraphrase: {question}",
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
            truncation=True,
        ).input_ids

        outputs = self._model.generate(
            input_ids,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences,
            no_repeat_ngram_size=no_repeat_ngram_size,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            max_length=max_length,
            diversity_penalty=diversity_penalty,
        )

        res = self._tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return res

    def augment(self, text: str) -> str:
        """Apply paraphrasing augmentation"""
        if not text.strip():
            return text

        try:
            self._load_model()

            # Generate paraphrases -- we only need one paraphrase
            paraphrases = self._paraphrase(text, num_return_sequences=1)

            # Return the first paraphrase if available, otherwise original text
            if paraphrases:
                return paraphrases[0].strip()
            else:
                logger.warning("No paraphrase generated, returning original text.")
                return text.strip()

        except Exception as e:
            logger.warning(f"Paraphrasing failed: {e}")
            return text


class TextAugmentationPipeline:
    """
    This pipeline uses user settings to run the pipeline. It does not handle creating these config or augmentors.
    This must be instantiated by the AugmentationManager. Think of this as an intern for that manager that never gets paid.
    """

    def __init__(
        self,
        augmentors: Dict[str, Any],
        synthesizer=None,
        synthesis_settings: Dict[str, Any] = None,
    ):
        self.augmentors = augmentors
        self.synthesizer = synthesizer
        self.synthesis_settings = synthesis_settings or {}

    def augment_dataset(
        self, dataset: List[Dict[str, Any]], augmentation_factor: float
    ) -> List[Dict[str, Any]]:
        """Augment entire dataset"""
        if augmentation_factor <= 1.0:
            return dataset

        augmented_dataset = list(dataset)
        target_size = int(len(dataset) * augmentation_factor)
        additional_needed = target_size - len(dataset)

        # Use synthesis if available
        if self.synthesizer and self.synthesizer.is_available():
            synthesis_ratio = self.synthesis_settings.get("synthesis_ratio", 0.5)
            synthesis_count = int(additional_needed * synthesis_ratio)
            augmentation_count = additional_needed - synthesis_count

            if synthesis_count > 0:
                try:
                    system_message = self.synthesis_settings.get("system_message", "")
                    max_batch_size = self.synthesis_settings.get("max_batch_size", 10)
                    custom_prompt = self.synthesis_settings.get("custom_prompt", "")

                    synthetic_samples = self.synthesizer.synthesize_conversations(
                        dataset,
                        num_samples=synthesis_count,
                        system_message=system_message,
                        max_batch_size=max_batch_size,
                        user_custom_prompt=custom_prompt,
                    )
                    augmented_dataset.extend(synthetic_samples)
                    logger.info(f"Generated {len(synthetic_samples)} synthetic samples")
                except Exception as e:
                    logger.error(f"Synthesis failed: {e}, falling back to augmentation")
                    augmentation_count = additional_needed

            additional_needed = augmentation_count

        # Use traditional augmentation for remaining samples
        if additional_needed > 0 and self.augmentors:
            samples_to_augment = random.choices(dataset, k=additional_needed)

            for sample in samples_to_augment:
                try:
                    augmented_sample = self._augment_conversation(sample)
                    augmented_dataset.append(augmented_sample)
                except Exception as e:
                    logger.warning(f"Failed to augment sample: {e}")
                    augmented_dataset.append(sample)  # Use original as fallback

        logger.info(
            f"Dataset expanded from {len(dataset)} to {len(augmented_dataset)} samples"
        )
        return augmented_dataset

    def _augment_conversation(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Augment a single conversation sample"""
        if "messages" not in sample:
            return sample

        augmented_sample = sample.copy()
        augmented_messages = []

        for message in sample["messages"]:
            role = message.get("role", "")
            content = message.get("content", "")

            if role in ["user", "assistant"]:
                augmented_content = self._augment_text(content)
            else:
                augmented_content = content

            augmented_messages.append({"role": role, "content": augmented_content})

        augmented_sample["messages"] = augmented_messages
        return augmented_sample

    def _augment_text(self, text: str) -> str:
        """Augment text using available methods"""
        if not text.strip() or not self.augmentors:
            return text

        available_methods = list(self.augmentors.keys())
        random.shuffle(available_methods)

        for method in available_methods:
            try:
                augmented = self.augmentors[method].augment(text)
                if augmented != text and augmented.strip():
                    return augmented
            except Exception as e:
                logger.warning(f"Augmentation method {method} failed: {e}")
                continue

        return text
