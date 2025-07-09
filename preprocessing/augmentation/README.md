# Text Augmentation Module

This module provides comprehensive text augmentation capabilities including:

- EDA (Easy Data Augmentation)
- Back Translation
- Paraphrasing
- Conversational augmentation

---

## Quick Start

### Simplified pipeline interface

```python
from preprocessing.augmentation import run_augment_pipeline

augmented_dataset, result = run_augment_pipeline(
    dataset,
    {
        "use_eda": True,
        "eda_alpha_sr": 0.1,
        # Other settings as needed
    }
)
```

### Creating custom configurations

```python
from preprocessing.augmentation import (
    AugmentationManager,
    AugmentationConfig,
    AugmentationMethod,
    run_augment_pipeline,
)

# High-level interface: use AugmentationManager and AugmentationConfig
manager = AugmentationManager()
config = AugmentationConfig(
    enabled_methods=[AugmentationMethod.EDA, AugmentationMethod.BACK_TRANSLATION],
    augmentation_factor=1.5,
)
manager.configure(config)
augmented_dataset, result = manager.augment_dataset(dataset)

# Or use the simplified pipeline interface
augmented_dataset, result = run_augment_pipeline(
    dataset,
    {
        "use_eda": True,
        "eda_alpha_sr": 0.1,
        # Other settings as needed
    }
)
```

---

## Low-level configurations

- **Custom**: Use `AugmentationConfig` directly for full control.

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
    eda_settings={"eda_alpha_sr": 0.15},
    back_translation_settings={"intermediate_lang": "es"},
    paraphrasing_settings={"paraphrase_model": "humarin/chatgpt_paraphraser_on_T5_base"},
    synthesis_settings={
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
config = AugmentationConfig(
    enabled_methods=[AugmentationMethod.EDA, AugmentationMethod.SYNTHESIS],
    synthesis_settings={
        "gemini_api_key": "your-api-key",
        "custom_prompt": "Focus on educational content.",
        "synthesis_ratio": 0.7,
        "system_message": "You are a helpful educational assistant."
    }
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

| Field                       | Type                       | Description                 | Default  |
| --------------------------- | -------------------------- | --------------------------- | -------- |
| `enabled_methods`           | `List[AugmentationMethod]` | Methods to enable           | Required |
| `augmentation_factor`       | `float`                    | Dataset size multiplier     | `1.5`    |
| `eda_settings`              | `EDASettings`              | EDA parameters              | `{}`     |
| `back_translation_settings` | `BackTranslationSettings`  | Back translation parameters | `{}`     |
| `paraphrasing_settings`     | `ParaphrasingSettings`     | Paraphrasing parameters     | `{}`     |
| `synthesis_settings`        | `SynthesisSettings`        | Synthesis parameters        | `{}`     |

The internal structures of augmentation settings and synthesis settings are defined in `text_augmentor.py` but are not needed for the user, since currently we do not support customising parameters for augmentation methods.

---

## Best Practices

- **Be specific**: Define domain, topics, and quality criteria in custom prompts.
- **Set boundaries**: Clearly state what to avoid.
- **Review outputs**: Always validate augmented samples for quality.

---

## Troubleshooting

- **Synthesis not applied**: Check API key and `custom_prompt`.
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
