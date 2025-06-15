import random
import logging
from typing import List, Dict, Any
from abc import ABC, abstractmethod
from eda import eda

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
                self._tokenizer = AutoTokenizer.from_pretrained(
                    "humarin/chatgpt_paraphraser_on_T5_base"
                )
                self._model = AutoModelForSeq2SeqLM.from_pretrained(
                    "humarin/chatgpt_paraphraser_on_T5_base"
                )
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
    """Comprehensive augmentation pipeline with Back Translation, Paraphrasing, EDA, and Conversational"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

        # Initialize all augmentors based on config
        self.augmentors = {}

        # Back translation
        if self.config.get("enable_back_translation", False):
            self.augmentors["back_translation"] = BackTranslationAugmentor(
                intermediate_lang=self.config.get("intermediate_lang", "fr"),
            )

        # Paraphrasing
        if self.config.get("enable_paraphrasing", False):
            self.augmentors["paraphrasing"] = ParaphraseAugmentor(
                model_name=self.config.get(
                    "paraphrase_model", "Vamsi/T5_Paraphrase_Paws"
                ),
            )

        # EDA
        if self.config.get("enable_eda", False):
            self.augmentors["eda"] = EDAugmentor(
                alpha_sr=self.config.get("eda_alpha_sr", 0.1),
                alpha_ri=self.config.get("eda_alpha_ri", 0.1),
                alpha_rs=self.config.get("eda_alpha_rs", 0.1),
                p_rd=self.config.get("eda_p_rd", 0.1),
                num_aug=1,
            )

        # Control which augmentors to use
        self.enabled_augmentors = self.config.get(
            "enabled_augmentors", list(self.augmentors.keys())
        )

        logger.info(
            f"Initialized comprehensive augmentation pipeline with: {self.enabled_augmentors}"
        )

    def augment_text(self, text: str, methods: List[str] = None) -> str:
        """
        Apply augmentation and return one variation

        Args:
            text: Input text to augment
            methods: List of methods to try
        """
        if not text.strip():
            return text

        methods = methods or self.enabled_augmentors
        available_methods = [m for m in methods if m in self.augmentors]
        if not available_methods:
            # this is a safeguard to ensure we have methods to apply otherwise we just generate duplicate data
            raise ValueError(
                "You are trying to augment text, but no methods are enabled or available."
            )

        # Randomly shuffle methods for each attempt
        random.shuffle(available_methods)

        for method in available_methods:
            try:
                augmented = self.augmentors[method].augment(text)
                # Check if augmentation result is different from original to avoid duplicates
                if augmented != text and augmented.strip():
                    return augmented
            except Exception as e:
                logger.warning(f"Augmentation method {method} failed: {e}")
                continue

        return text  # Return original if all attempts fail

    def augment_conversation(
        self, sample: Dict[str, Any], num_variations: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Augment a conversation sample and return multiple variations
        Variations will not include the original sample.
        """
        if "messages" not in sample:
            return [sample]

        variations = []

        for _ in range(num_variations):
            try:
                augmented_sample = sample.copy()
                augmented_messages = []

                for message in sample["messages"]:
                    role = message.get("role", "")
                    content = message.get("content", "")

                    if role in ["user", "assistant"]:
                        # Apply text augmentation
                        augmented_content = self.augment_text(content)
                    else:
                        # Keep system messages unchanged
                        augmented_content = content

                    augmented_messages.append(
                        {"role": role, "content": augmented_content}
                    )

                augmented_sample["messages"] = augmented_messages
                variations.append(augmented_sample)

            except Exception as e:
                logger.warning(f"Failed to create variation: {e}")
                continue

        return variations

    def augment_dataset(
        self,
        dataset: List[Dict[str, Any]],
        augmentation_factor: float = 1.5,
        num_variations: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Augment entire dataset by expanding it with variations to a desired size.
        This is the main method used by preprocessor on TextAugmentationPipeline.

        We will select (1 - augmentation_factor) * len(dataset) samples randomly
        and for each selected sample generate num_variations variations, and
        then select one of them randomly to add to the dataset.
        """
        if augmentation_factor <= 1.0:
            return dataset

        augmented_dataset = list(dataset)  # Include original samples

        # Calculate how many additional samples to generate
        target_size = int(len(dataset) * augmentation_factor)
        additional_needed = target_size - len(dataset)

        # Randomly select samples to augment
        samples_to_augment = random.choices(dataset, k=additional_needed)

        for sample in samples_to_augment:
            try:
                # Generate variations for that conversation sample, will not contain original sample
                variations = self.augment_conversation(
                    sample, num_variations=num_variations
                )
                # if we have any variations, randomly select one
                if len(variations) > 0:
                    augmented_dataset.append(random.choice(variations))
                else:
                    augmented_dataset.append(sample)
            except Exception as e:
                logger.warning(f"Failed to augment sample: {e}")
                continue

        logger.info(
            f"Augmented dataset from {len(dataset)} to {len(augmented_dataset)} samples"
        )
        return augmented_dataset

    def get_stats(self) -> Dict[str, Any]:
        """Get augmentation statistics"""
        return {
            "enabled_augmentors": self.enabled_augmentors,
            "available_methods": list(self.augmentors.keys()),
            "config": self.config,
            "methods": {
                "back_translation": "Helsinki-NLP translation models",
                "paraphrasing": "T5-based paraphrasing",
                "eda": "Easy Data Augmentation (synonym, insertion, swap, deletion)",
            },
        }


# Factory functions for easy instantiation
def create_augmentation_pipeline(
    eda: bool = False,
    back_translation: bool = False,
    paraphrasing: bool = False,
    **kwargs: Any,
) -> TextAugmentationPipeline:
    """
    Create augmentation pipeline with preset configurations.
    For each augmentation type, you can enable or disable it.
    """

    if eda:
        # EDA-only config
        config = {
            "enable_eda": True,
            "eda_alpha_sr": kwargs.get("eda_alpha_sr", 0.1),
            "eda_alpha_ri": kwargs.get("eda_alpha_ri", 0.1),
            "eda_alpha_rs": kwargs.get("eda_alpha_rs", 0.1),
            "eda_p_rd": kwargs.get("eda_p_rd", 0.1),
            **kwargs,
        }

    elif back_translation:
        config = {
            "enable_back_translation": True,
            "intermediate_lang": kwargs.get("intermediate_lang", "fr"),
            **kwargs,
        }
    elif paraphrasing:
        config = {
            "enable_paraphrasing": True,
            "paraphrase_model": kwargs.get(
                "paraphrase_model", "Vamsi/T5_Paraphrase_Paws"
            ),
            **kwargs,
        }

    return TextAugmentationPipeline(config)


# Example usage
if __name__ == "__main__":
    # Test all three methods
    text = "This is a simple example sentence for testing comprehensive augmentation."

    print("Original:", text)
    print()

    # Test lightweight pipeline (EDA only)
    # NOTE: To test anything else just set the corresponding flags to True
    print("1. Testing EDA-only pipeline...")
    eda_pipeline = create_augmentation_pipeline(eda=True)
    eda_variations = eda_pipeline.augment_text(text)

    for i, variation in enumerate(eda_variations):
        print(f"EDA variation {i}: {variation}")

    print()

    # Test dataset augmentation
    print("\nTesting dataset augmentation on full pipeline...")
    full_pipeline = create_augmentation_pipeline(
        eda=True, back_translation=True, paraphrasing=True
    )
    sample_dataset = [
        {
            "messages": [
                {"role": "user", "content": "What is artificial intelligence?"},
                {
                    "role": "assistant",
                    "content": "AI is the simulation of human intelligence in machines.",
                },
            ]
        },
        {
            "messages": [
                {
                    "role": "user",
                    "content": "How does deep learning differ from traditional machine learning?",
                },
                {
                    "role": "assistant",
                    "content": "Deep learning uses neural networks with many layers to analyze data.",
                },
            ]
        },
    ]

    print(f"Original dataset size: {len(sample_dataset)}")
    augmented_dataset = full_pipeline.augment_dataset(
        sample_dataset, augmentation_factor=1.5
    )

    print(f"Augmented dataset size: {len(augmented_dataset)}")
    for i, sample in enumerate(augmented_dataset):
        print(f"Sample {i + 1}:")
        for msg in sample["messages"]:
            print(f"  {msg['role']}: {msg['content']}")
        print()
    print(f"\nPipeline stats: {full_pipeline.get_stats()}")
