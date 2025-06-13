import random
import logging
from typing import List, Dict, Any
from abc import ABC, abstractmethod
from eda import eda

logger = logging.getLogger(__name__)


class BaseAugmentor(ABC):
    """Base class for all text augmentation techniques"""

    def __init__(self, probability: float = 0.5):
        self.probability = probability

    @abstractmethod
    def augment(self, text: str) -> str:
        """Apply augmentation to text"""
        pass

    def should_augment(self) -> bool:
        """Determine if augmentation should be applied based on probability"""
        return random.random() < self.probability


class BackTranslationAugmentor(BaseAugmentor):
    """Back translation using Helsinki-NLP models via transformers"""

    def __init__(self, probability: float = 0.3, intermediate_lang: str = "fr"):
        super().__init__(probability)
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
        if not self.should_augment() or not text.strip():
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

            return back_translated if back_translated.strip() else text

        except Exception as e:
            logger.warning(f"Back translation failed: {e}")
            return text


class ParaphraseAugmentor(BaseAugmentor):
    """Paraphrasing using T5 or similar models"""

    def __init__(
        self, probability: float = 0.4, model_name: str = "Vamsi/T5_Paraphrase_Paws"
    ):
        super().__init__(probability)
        self.model_name = model_name
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """Lazy load paraphrasing model"""
        if self._model is None:
            try:
                from transformers import T5ForConditionalGeneration, T5Tokenizer

                self._tokenizer = T5Tokenizer.from_pretrained(self.model_name)
                self._model = T5ForConditionalGeneration.from_pretrained(
                    self.model_name
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

    def augment(self, text: str) -> str:
        """Apply paraphrasing augmentation"""
        if not self.should_augment() or not text.strip():
            return text

        try:
            self._load_model()

            # Prepare input for T5 (add task prefix)
            input_text = f"paraphrase: {text}"

            # Tokenize and generate
            inputs = self._tokenizer.encode(
                input_text, return_tensors="pt", max_length=512, truncation=True
            )
            outputs = self._model.generate(
                inputs,
                max_length=512,
                num_beams=5,
                num_return_sequences=1,
                temperature=0.7,
                early_stopping=True,
            )

            paraphrased = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            return paraphrased if paraphrased.strip() else text

        except Exception as e:
            logger.warning(f"Paraphrasing failed: {e}")
            return text


class EDAugmentor(BaseAugmentor):
    """Easy Data Augmentation using the original EDA implementation"""

    def __init__(
        self,
        probability: float = 0.5,
        alpha_sr=0.1,
        alpha_ri=0.1,
        alpha_rs=0.1,
        p_rd=0.1,
        num_aug=1,
    ):
        super().__init__(probability)
        self.alpha_sr = alpha_sr  # synonym replacement
        self.alpha_ri = alpha_ri  # random insertion
        self.alpha_rs = alpha_rs  # random swap
        self.p_rd = p_rd  # random deletion
        self.num_aug = num_aug  # number of augmented sentences per original

        # Try to download WordNet if not available
        try:
            import nltk
            from nltk.corpus import wordnet

            # Test if wordnet is available
            wordnet.synsets("test")
        except:
            try:
                import nltk

                nltk.download("wordnet", quiet=True)
                logger.info("Downloaded NLTK WordNet data")
            except Exception as e:
                logger.warning(
                    f"Could not download WordNet: {e}. EDA will use basic augmentation only."
                )

    def augment(self, text: str) -> str:
        """Apply EDA augmentation"""
        if not self.should_augment() or not text.strip():
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


class TextAugmentationPipeline:
    """Comprehensive augmentation pipeline with Back Translation, Paraphrasing, EDA, and Conversational"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

        # Initialize all augmentors based on config
        self.augmentors = {}

        # Back translation
        if self.config.get("enable_back_translation", True):
            self.augmentors["back_translation"] = BackTranslationAugmentor(
                probability=self.config.get("back_translation_probability", 0.2),
                intermediate_lang=self.config.get("intermediate_lang", "fr"),
            )

        # Paraphrasing
        if self.config.get("enable_paraphrasing", True):
            self.augmentors["paraphrasing"] = ParaphraseAugmentor(
                probability=self.config.get("paraphrasing_probability", 0.3),
                model_name=self.config.get(
                    "paraphrase_model", "Vamsi/T5_Paraphrase_Paws"
                ),
            )

        # EDA
        if self.config.get("enable_eda", True):
            self.augmentors["eda"] = EDAugmentor(
                probability=self.config.get("eda_probability", 0.4),
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

    def augment_text(self, text: str, methods: List[str] = None) -> List[str]:
        """Apply augmentation and return multiple variations"""
        if not text.strip():
            return [text]

        methods = methods or self.enabled_augmentors
        variations = [text]  # Include original

        for method in methods:
            if method in self.augmentors:
                try:
                    augmented = self.augmentors[method].augment(text)
                    if augmented != text and augmented.strip():
                        variations.append(augmented)
                except Exception as e:
                    logger.warning(f"Augmentation method {method} failed: {e}")
                    continue

        return variations

    def augment_conversation_sample(
        self, sample: Dict[str, Any], num_variations: int = 2
    ) -> List[Dict[str, Any]]:
        """Augment a conversation sample and return multiple variations"""
        if "messages" not in sample:
            return [sample]

        variations = [sample]  # Include original

        for _ in range(num_variations):
            try:
                augmented_sample = sample.copy()
                augmented_messages = []

                for message in sample["messages"]:
                    role = message.get("role", "")
                    content = message.get("content", "")

                    if role == "user":
                        # Augment user questions
                        augmented_content = (
                            self.conversation_augmentor.vary_question_style(content)
                        )

                        # Apply text augmentation with lower probability for questions
                        text_variations = self.augment_text(augmented_content)
                        if len(text_variations) > 1 and random.random() < 0.3:
                            augmented_content = random.choice(text_variations[1:])

                    elif role == "assistant":
                        # Apply text augmentation to assistant responses
                        text_variations = self.augment_text(content)
                        if len(text_variations) > 1:
                            augmented_content = random.choice(text_variations[1:])
                        else:
                            augmented_content = content

                        # Add conversational elements
                        augmented_content = (
                            self.conversation_augmentor.augment_response(
                                augmented_content
                            )
                        )

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
        self, dataset: List[Dict[str, Any]], augmentation_factor: float = 1.5
    ) -> List[Dict[str, Any]]:
        """Augment entire dataset"""
        if augmentation_factor <= 1.0:
            return dataset

        augmented_dataset = list(dataset)  # Include original samples

        # Calculate how many additional samples to generate
        target_size = int(len(dataset) * augmentation_factor)
        additional_needed = target_size - len(dataset)

        if additional_needed <= 0:
            return dataset

        # Randomly select samples to augment
        samples_to_augment = random.choices(dataset, k=additional_needed)

        for sample in samples_to_augment:
            try:
                variations = self.augment_conversation_sample(sample, num_variations=1)
                if len(variations) > 1:
                    augmented_dataset.append(variations[1])  # Add the augmented version
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
                "conversational": "Natural conversation variations",
            },
        }


# Factory functions for easy instantiation
def create_augmentation_pipeline(
    lightweight: bool = False, **kwargs
) -> TextAugmentationPipeline:
    """Create augmentation pipeline with different complexity levels"""

    if lightweight:
        # Lightweight config - only EDA and conversational augmentation
        config = {
            "enable_back_translation": False,
            "enable_paraphrasing": False,
            "enable_eda": True,
            "eda_probability": kwargs.get("eda_probability", 0.5),
            "eda_alpha_sr": kwargs.get("eda_alpha_sr", 0.1),
            "eda_alpha_ri": kwargs.get("eda_alpha_ri", 0.1),
            "eda_alpha_rs": kwargs.get("eda_alpha_rs", 0.1),
            "eda_p_rd": kwargs.get("eda_p_rd", 0.1),
            **kwargs,
        }
    else:
        # Full config with all augmentation methods
        config = {
            "enable_back_translation": kwargs.get("enable_back_translation", True),
            "enable_paraphrasing": kwargs.get("enable_paraphrasing", True),
            "enable_eda": kwargs.get("enable_eda", True),
            "back_translation_probability": kwargs.get(
                "back_translation_probability", 0.2
            ),
            "paraphrasing_probability": kwargs.get("paraphrasing_probability", 0.3),
            "eda_probability": kwargs.get("eda_probability", 0.4),
            "intermediate_lang": kwargs.get("intermediate_lang", "fr"),
            "eda_alpha_sr": kwargs.get("eda_alpha_sr", 0.1),
            "eda_alpha_ri": kwargs.get("eda_alpha_ri", 0.1),
            "eda_alpha_rs": kwargs.get("eda_alpha_rs", 0.1),
            "eda_p_rd": kwargs.get("eda_p_rd", 0.1),
            **kwargs,
        }

    return TextAugmentationPipeline(config)


def create_eda_only_pipeline(**kwargs) -> TextAugmentationPipeline:
    """Create pipeline with only EDA augmentation"""
    config = {
        "enable_back_translation": False,
        "enable_paraphrasing": False,
        "enable_eda": True,
        "eda_probability": kwargs.get("eda_probability", 0.8),
        "eda_alpha_sr": kwargs.get("eda_alpha_sr", 0.1),
        "eda_alpha_ri": kwargs.get("eda_alpha_ri", 0.1),
        "eda_alpha_rs": kwargs.get("eda_alpha_rs", 0.1),
        "eda_p_rd": kwargs.get("eda_p_rd", 0.1),
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
    print("1. Testing EDA-only pipeline...")
    eda_pipeline = create_eda_only_pipeline()
    eda_variations = eda_pipeline.augment_text(text)

    for i, variation in enumerate(eda_variations):
        print(f"EDA variation {i}: {variation}")

    print()

    # Test conversation sample with all methods
    print("2. Testing full pipeline with conversation...")
    full_pipeline = create_augmentation_pipeline(lightweight=False)

    conversation_sample = {
        "messages": [
            {
                "role": "user",
                "content": "What is machine learning and how does it work?",
            },
            {
                "role": "assistant",
                "content": "Machine learning is a method of data analysis that automates analytical model building using algorithms.",
            },
        ]
    }

    print("Original conversation:")
    for msg in conversation_sample["messages"]:
        print(f"  {msg['role']}: {msg['content']}")

    print("\nAugmented conversation (full pipeline):")
    variations = full_pipeline.augment_conversation_sample(
        conversation_sample, num_variations=1
    )
    if len(variations) > 1:
        for msg in variations[1]["messages"]:
            print(f"  {msg['role']}: {msg['content']}")

    print(f"\nPipeline stats: {full_pipeline.get_stats()}")
