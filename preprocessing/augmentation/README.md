# Text Augmentation Module

This module provides a unified, robust interface for text augmentation, supporting EDA, Back Translation, Paraphrasing, and LLM Synthesis (Gemini). It is optimized for LLM fine-tuning and dataset expansion.

---

## Quick Start

```python
from preprocessing.augmentation import (
    AugmentationManager,
    create_lightweight_config,
    create_full_config,
    run_augment_pipeline,
)

# Lightweight (EDA only)
manager = AugmentationManager()
config = create_lightweight_config(augmentation_factor=1.5)
manager.configure(config)
augmented_dataset, result = manager.augment_dataset(dataset)

# Full config (all methods)
config = create_full_config(
    augmentation_factor=2.0,
    enable_synthesis=True,
    gemini_api_key="your-api-key",
    custom_prompt="Focus on educational content with clear explanations.",
    synthesis_ratio=0.3,
    system_message="You are a helpful AI assistant.",
    max_batch_size=10,
)
manager.configure(config)
augmented_dataset, result = manager.augment_dataset(dataset)
```

---

## Configuration

- **Lightweight**: EDA only, fast, no dependencies.
- **Full**: All methods, including synthesis and transformer-based methods.

```python
from preprocessing.augmentation import AugmentationConfig, AugmentationMethod

config = AugmentationConfig(
    enabled_methods=[
        AugmentationMethod.EDA,
        AugmentationMethod.BACK_TRANSLATION,
        AugmentationMethod.PARAPHRASING,
        AugmentationMethod.SYNTHESIS,
    ],
    augmentation_factor=1.8,
    lightweight_mode=False,
    eda_settings={"eda_alpha_sr": 0.15},
    back_translation_settings={"intermediate_lang": "es"},
    paraphrasing_settings={"paraphrase_model": "humarin/chatgpt_paraphraser_on_T5_base"},
    synthesis_settings={
        "enable_synthesis": True,
        "gemini_api_key": "your-api-key",
        "custom_prompt": "Focus on educational content.",
        "synthesis_ratio": 0.7,
        "system_message": "You are a helpful educational assistant.",
        "max_batch_size": 5,
    }
)
```

---

## Augmentation Methods

### 1. EDA (Easy Data Augmentation)

- **Description**: Fast lexical transformations using WordNet synonyms and random operations.
- **Techniques**: Synonym replacement, random insertion, random swap, random deletion.
- **Dependencies**: None (requires NLTK).
- **Best for**: Fast, lightweight augmentation.

### 2. Back Translation

- **Description**: Translate text to an intermediate language and back to English for semantic variation.
- **Dependencies**: `transformers`, `torch`.
- **Best for**: Semantic diversity, cross-lingual robustness.

### 3. Paraphrasing

- **Description**: T5-based paraphrasing for high-quality semantic variations.
- **Dependencies**: `transformers`, `torch`.
- **Best for**: Meaning preservation, conversational data.

### 4. LLM Synthesis (Gemini)

- **Description**: Generate new conversations using Gemini API, guided by custom prompts.
- **Dependencies**: `google-generativeai`.
- **Best for**: Creating diverse, high-quality synthetic data.

#### Custom Prompt Example

```python
config = create_full_config(
    enable_synthesis=True,
    gemini_api_key="your-api-key",
    custom_prompt="""
You are generating training data for an educational AI assistant.

When synthesizing conversations, please ensure:
- Focus on educational topics like programming, data science, mathematics
- Questions should be at beginner to intermediate level
- Answers should be informative but accessible
- Include practical examples when possible
- Avoid controversial or sensitive topics
""",
    synthesis_ratio=0.7,
    system_message="You are a helpful educational assistant."
)
```

---

## API Integration Example

```python
import requests

config = {
    "augmentation_config": {
        "enabled": True,
        "lightweight": False,
        "augmentation_factor": 2.0,
        "pipeline_config": {
            "enable_synthesis": True,
            "gemini_api_key": "your-api-key",
            "custom_prompt": "Focus on educational content.",
            "synthesis_ratio": 0.7,
            "system_message": "You are a helpful educational assistant."
        }
    }
}

response = requests.post("/preprocess", json={
    "dataset_source": "upload",
    "dataset_id": "your-dataset-id",
    "options": config
})
```

---

## Configuration Reference

| Field                       | Type                       | Description                    | Default  |
| --------------------------- | -------------------------- | ------------------------------ | -------- |
| `enabled_methods`           | `List[AugmentationMethod]` | Methods to enable              | Required |
| `augmentation_factor`       | `float`                    | Dataset size multiplier        | `1.5`    |
| `lightweight_mode`          | `bool`                     | Use only fast methods          | `True`   |
| `eda_settings`              | `Dict`                     | EDA parameters                 | `{}`     |
| `back_translation_settings` | `Dict`                     | Back translation parameters    | `{}`     |
| `paraphrasing_settings`     | `Dict`                     | Paraphrasing parameters        | `{}`     |
| `synthesis_settings`        | `Dict`                     | Synthesis parameters           | `{}`     |

---

## Best Practices

- **Be specific**: Define domain, topics, and quality criteria in custom prompts.
- **Set boundaries**: Clearly state what to avoid.
- **Review outputs**: Always validate augmented samples for quality.

---

## Troubleshooting

- **Synthesis not applied**: Check `enable_synthesis`, API key, and `custom_prompt`.
- **Low quality**: Make prompts more specific, add examples, adjust `synthesis_ratio`.
- **Dependency issues**: Ensure required libraries are installed.

---

## Dependencies

- `nltk>=3.8` (always)
- `transformers`, `torch`, `sentencepiece` (for transformer methods)
- `google-generativeai` (for synthesis)

---

## References

- [EDA Paper](https://arxiv.org/abs/1901.11196)
- [Back Translation](https://arxiv.org/abs/1511.06709)
- [T5 Paraphrasing](https://arxiv.org/abs/1910.10683)
- [Google Gemini API](https://ai.google.dev/)

---

This system provides production-ready text augmentation optimized for LLM fine-tuning with research-backed methods and flexible configuration.
