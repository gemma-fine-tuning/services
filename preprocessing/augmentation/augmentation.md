# Comprehensive Text Augmentation System

## Overview

This system provides **three powerful text augmentation methods** for fine-tuning dataset enhancement:

1. **🔄 Back Translation** - Using Helsinki-NLP translation models
2. **✍️ Paraphrasing** - Using T5-based paraphrasing models
3. **📊 EDA (Easy Data Augmentation)** - Using the original research implementation

## Quick Start

### 1. EDA Only (Fast, No Downloads)

```python
from text_augmentor import create_eda_only_pipeline

pipeline = create_eda_only_pipeline()
variations = pipeline.augment_text("Your text here")
```

### 2. All Methods Combined

```python
from text_augmentor import create_augmentation_pipeline

pipeline = create_augmentation_pipeline(lightweight=False)
augmented_dataset = pipeline.augment_dataset(your_dataset, augmentation_factor=1.5)
```

### 3. Transformer Methods Only

```python
from text_augmentor import create_transformer_only_pipeline

pipeline = create_transformer_only_pipeline()
variations = pipeline.augment_text("Your text here")
```

## Testing

Run the comprehensive test to see all methods in action:

```bash
cd preprocessing
python test_comprehensive_augmentation.py
```

This will demonstrate:

- EDA augmentation (immediate)
- Conversation-style augmentation
- Dataset scaling
- Optional transformer methods (requires model downloads)

## Method Comparison

| Method               | Speed  | Quality    | Downloads | Memory | Best For                        |
| -------------------- | ------ | ---------- | --------- | ------ | ------------------------------- |
| **EDA**              | ⚡⚡⚡ | ⭐⭐⭐     | None      | 50MB   | Development, Fast iteration     |
| **Back Translation** | ⚡     | ⭐⭐⭐⭐⭐ | ~500MB    | 2GB    | High-quality semantic variation |
| **Paraphrasing**     | ⚡⚡   | ⭐⭐⭐⭐⭐ | ~500MB    | 1GB    | Meaning preservation            |
| **All Combined**     | ⚡     | ⭐⭐⭐⭐⭐ | ~1GB      | 3GB    | Production datasets             |

## Configuration Examples

### Conservative Augmentation (Development)

```python
config = {
    'enable_eda': True,
    'enable_back_translation': False,
    'enable_paraphrasing': False,
    'eda_probability': 0.3,
    'eda_alpha_sr': 0.05,  # Low synonym replacement
    'eda_alpha_ri': 0.05,  # Low random insertion
    'eda_alpha_rs': 0.05,  # Low random swap
    'eda_p_rd': 0.05       # Low random deletion
}
```

### Aggressive Augmentation (Production)

```python
config = {
    'enable_eda': True,
    'enable_back_translation': True,
    'enable_paraphrasing': True,
    'eda_probability': 0.5,
    'back_translation_probability': 0.3,
    'paraphrasing_probability': 0.4,
    'eda_alpha_sr': 0.15,
    'eda_alpha_ri': 0.15,
    'eda_alpha_rs': 0.15,
    'eda_p_rd': 0.15,
    'intermediate_lang': 'fr'
}
```

## API Integration

Use with the preprocessing API:

```json
{
  "dataset_source": "upload",
  "dataset_id": "your-dataset",
  "options": {
    "augmentation_config": {
      "enabled": true,
      "lightweight": false,
      "augmentation_factor": 1.5,
      "pipeline_config": {
        "enable_eda": true,
        "enable_paraphrasing": true,
        "enable_back_translation": true,
        "eda_probability": 0.4,
        "paraphrasing_probability": 0.3,
        "back_translation_probability": 0.2
      }
    }
  }
}
```

## Preset Configurations

The system includes built-in presets:

- **`medical`**: Full augmentation optimized for medical domain
- **`question_answering`**: All methods for Q&A tasks
- **`default`**: No augmentation (original behavior)

## Dependencies

### Required (Always)

```
nltk>=3.8  # For EDA
```

### Optional (For Transformer Methods)

```
transformers>=4.30.0
torch>=2.0.0
sentencepiece>=0.1.99
```

## Performance Guidelines

### For Development/Testing

- Use EDA only: `create_eda_only_pipeline()`
- Fast iteration, good quality
- No model downloads needed

### For Small Datasets (<1K samples)

- Use all methods: `create_augmentation_pipeline(lightweight=False)`
- Maximum diversity and quality
- Worth the model download time

### For Large Datasets (>10K samples)

- Use EDA + one transformer method
- Balance speed and quality
- Consider computational costs

## Troubleshooting

### EDA Issues

```
Problem: No synonyms found
Solution: Ensure NLTK WordNet is downloaded
Fix: python -c "import nltk; nltk.download('wordnet')"
```

### Transformer Issues

```
Problem: Model download fails
Solution: Check internet connection and disk space
Requirements: ~2GB free space, stable internet
```

### Memory Issues

```
Problem: Out of memory during augmentation
Solution: Use lightweight=True or reduce batch sizes
Alternative: Process dataset in smaller chunks
```

## Examples

### Domain Adaptation (Medical)

```python
pipeline = create_augmentation_pipeline(
    lightweight=False,
    enable_eda=True,
    enable_paraphrasing=True,
    enable_back_translation=False,  # Skip for medical accuracy
    eda_alpha_sr=0.05,  # Conservative synonym replacement
    paraphrasing_probability=0.3
)

medical_dataset = [
    {'messages': [
        {'role': 'user', 'content': 'What are the symptoms of diabetes?'},
        {'role': 'assistant', 'content': 'Common symptoms include frequent urination, excessive thirst, and unexplained weight loss.'}
    ]}
]

augmented = pipeline.augment_dataset(medical_dataset, augmentation_factor=1.3)
```

### Task Adaptation (Q&A)

```python
pipeline = create_augmentation_pipeline(
    lightweight=False,
    enable_eda=True,
    enable_paraphrasing=True,
    enable_back_translation=True,
    eda_probability=0.5,
    paraphrasing_probability=0.3,
    back_translation_probability=0.2,
    intermediate_lang='fr'
)

qa_dataset = [
    {'messages': [
        {'role': 'user', 'content': 'How does machine learning work?'},
        {'role': 'assistant', 'content': 'Machine learning uses algorithms to find patterns in data and make predictions.'}
    ]}
]

augmented = pipeline.augment_dataset(qa_dataset, augmentation_factor=1.4)
```

## Contributing

To add new augmentation methods:

1. Create a new class inheriting from `BaseAugmentor`
2. Implement the `augment(text)` method
3. Add to `TextAugmentationPipeline.augmentors`
4. Update configuration schemas
5. Add tests and documentation

## References

- **EDA Paper**: "EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks" (Wei & Zou, 2019)
- **Back Translation**: Helsinki-NLP OPUS-MT models
- **Paraphrasing**: T5-based paraphrasing models

This comprehensive system provides production-ready text augmentation for fine-tuning datasets with flexible configuration and proven research-backed methods.
