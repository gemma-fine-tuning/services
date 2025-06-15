# Comprehensive Text Augmentation System

## Overview

This system provides **three powerful text augmentation methods** for enhancing fine-tuning datasets. Each method applies augmentation 100% of the time when called, with no probabilistic gating at the method level. Dataset-level control is handled by the pipeline's sampling strategy.

1. **🔄 Back Translation** - Helsinki-NLP Marian translation models
2. **✍️ Paraphrasing** - T5-based specialized paraphrasing models
3. **📊 EDA (Easy Data Augmentation)** - Lightweight lexical transformations

> The overall augmentation techniques are informed by [this research overview paper](https://arxiv.org/html/2501.18845v1). We tried to select the methods that are most efficient but also provide the best quality for LLM fine-tuning tasks.

## Quick Start

### 1. EDA Only (Fast, No Downloads)

```python
from preprocessing.augmentation import create_augmentation_pipeline

pipeline = create_augmentation_pipeline(eda=True)
augmented_text = pipeline.augment_text("Your text here")
```

### 2. Back Translation

```python
pipeline = create_augmentation_pipeline(back_translation=True)
augmented_text = pipeline.augment_text("Your text here")
```

### 3. Paraphrasing

```python
pipeline = create_augmentation_pipeline(paraphrasing=True)
augmented_text = pipeline.augment_text("Your text here")
```

### 4. Dataset Augmentation

```python
# Expand dataset by 50%
pipeline = create_augmentation_pipeline(paraphrasing=True)
augmented_dataset = pipeline.augment_dataset(
    your_dataset,
    augmentation_factor=1.5
)
```

## Augmentation Methods

### 1. Easy Data Augmentation (EDA)

**Source**: [EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks](https://arxiv.org/abs/1901.11196)

**Description**: Fast lexical transformations using WordNet synonyms and random operations.

**Techniques**:

- **Synonym Replacement**: Replace words with WordNet synonyms
- **Random Insertion**: Insert random synonyms at random positions
- **Random Swap**: Swap positions of two random words
- **Random Deletion**: Randomly delete words

**Advantages**:

- ⚡ Extremely fast (no model loading)
- 📦 Lightweight (only requires NLTK)
- 🔧 Highly configurable parameters

**Best for**: Development, rapid prototyping, large-scale augmentation

### 2. Back Translation

**Source**: [Helsinki-NLP OPUS-MT Models](https://huggingface.co/docs/transformers/en/model_doc/marian) | [Technique Paper](https://arxiv.org/abs/2106.04681)

**Description**: Translate text to an intermediate language and back to English to create semantic variations.

**Process**:

1. English → Intermediate Language (e.g., French)
2. Intermediate Language → English
3. Return paraphrased result

**Advantages**:

- 🌐 Preserves semantic meaning
- 🎯 High-quality variations
- 📚 Supports 1000+ language pairs

**Requirements**: ~600MB disk space, transformers library

**Best for**: Semantic diversity, cross-lingual robustness

### 3. Paraphrasing

**Source**: [ChatGPT Paraphraser on T5 Base](https://huggingface.co/humarin/chatgpt_paraphraser_on_T5_base)

**Description**: Specialized T5 model fine-tuned specifically for paraphrasing tasks.

**Process**:

1. Prefix input with "paraphrase: "
2. Generate paraphrase using beam search
3. Return highest-quality variation

**Advantages**:

- 🎯 Purpose-built for paraphrasing
- 💎 Highest semantic preservation
- 🔧 Configurable generation parameters

**Requirements**: ~1GB disk space, transformers library

**Best for**: High-quality semantic variations, meaning preservation

## Usage Examples

### Single Text Augmentation

```python
from preprocessing.augmentation import create_augmentation_pipeline

# Create pipeline with specific method
pipeline = create_augmentation_pipeline(eda=True)

# Augment single text
original = "Machine learning is a subset of artificial intelligence."
augmented = pipeline.augment_text(original)
print(f"Original: {original}")
print(f"Augmented: {augmented}")
```

### Conversation Augmentation

```python
# Augment conversation samples
conversation = {
    "messages": [
        {"role": "user", "content": "What is deep learning?"},
        {"role": "assistant", "content": "Deep learning uses neural networks with multiple layers."}
    ]
}

variations = pipeline.augment_conversation(conversation, num_variations=2)
for i, variation in enumerate(variations):
    print(f"Variation {i+1}:")
    for msg in variation["messages"]:
        print(f"  {msg['role']}: {msg['content']}")
```

### Dataset Expansion

```python
# Expand dataset by 50% using random sampling
dataset = [
    {"messages": [...]},  # Your conversation samples
    {"messages": [...]},
]

augmented_dataset = pipeline.augment_dataset(
    dataset,
    augmentation_factor=1.5  # 50% more samples
)

print(f"Original size: {len(dataset)}")
print(f"Augmented size: {len(augmented_dataset)}")
```

## Configuration

### EDA Parameters

```python
pipeline = create_augmentation_pipeline(
    eda=True,
    eda_alpha_sr=0.1,  # Synonym replacement ratio
    eda_alpha_ri=0.1,  # Random insertion ratio
    eda_alpha_rs=0.1,  # Random swap ratio
    eda_p_rd=0.1       # Random deletion probability
)
```

### Back Translation Parameters

```python
pipeline = create_augmentation_pipeline(
    back_translation=True,
    intermediate_lang="fr"  # French, German (de), Spanish (es), etc.
)
```

### Paraphrasing Parameters

```python
pipeline = create_augmentation_pipeline(
    paraphrasing=True,
    paraphrase_model="humarin/chatgpt_paraphraser_on_T5_base"
)
```

## Method Comparison

| Method               | Speed  | Quality    | Downloads | Memory | Deterministic | Best For                    |
| -------------------- | ------ | ---------- | --------- | ------ | ------------- | --------------------------- |
| **EDA**              | ⚡⚡⚡ | ⭐⭐⭐     | None      | 50MB   | No            | Development, Fast iteration |
| **Back Translation** | ⚡     | ⭐⭐⭐⭐⭐ | ~600MB    | 2GB    | Yes           | Semantic variation          |
| **Paraphrasing**     | ⚡⚡   | ⭐⭐⭐⭐⭐ | ~1GB      | 1GB    | No            | Meaning preservation        |

## API Integration

Use with the preprocessing service:

```json
{
  "dataset_source": "upload",
  "dataset_id": "your-dataset",
  "options": {
    "augmentation_config": {
      "enabled": true,
      "method": "paraphrasing",
      "augmentation_factor": 1.5,
      "pipeline_config": {
        "enable_paraphrasing": true,
        "paraphrase_model": "humarin/chatgpt_paraphraser_on_T5_base"
      }
    }
  }
}
```

## Testing

Run the main test:

```bash
cd preprocessing/augmentation
python text_augmentor.py
```

This demonstrates:

- Single text augmentation
- Conversation augmentation
- Dataset expansion
- All three methods in action

## Dependencies

### Required (Always)

```
nltk>=3.8  # For EDA WordNet
```

### Optional (For Transformer Methods)

```
transformers>=4.30.0
torch>=2.0.0
sentencepiece>=0.1.99
```

## Performance Guidelines

### For Development/Testing

- Use **EDA only**: Fast iteration, no downloads
- Good quality for most use cases

### For Production (Small Datasets <1K)

- Use **Paraphrasing**: Best quality-to-speed ratio
- Download time is acceptable for small datasets

### For Production (Large Datasets >10K)

- Use **EDA**: Computational efficiency matters
- Consider **Back Translation** for multilingual robustness

## Best Practices

### Dataset Augmentation Strategy

1. **Conservative Expansion** (1.2x): High-quality datasets
2. **Moderate Expansion** (1.5x): Standard practice
3. **Aggressive Expansion** (2.0x): Small/imbalanced datasets

### Method Selection

- **EDA**: Lexical diversity, speed priority
- **Back Translation**: Semantic robustness, multilingual data
- **Paraphrasing**: Meaning preservation, conversational data

### Quality Assurance

- Always review augmented samples
- Validate semantic consistency
- Test on held-out evaluation data

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
Requirements: Stable internet, sufficient disk space
```

### Memory Issues

```
Problem: Out of memory during augmentation
Solution: Process dataset in smaller batches
Alternative: Use EDA for memory-constrained environments
```

## References

- **EDA Paper**: [Wei & Zou, 2019 - Easy Data Augmentation Techniques](https://arxiv.org/abs/1901.11196)
- **Back Translation Technique**: [Sennrich et al., 2016 - Improving Neural Machine Translation](https://arxiv.org/abs/1511.06709)
- **Helsinki-NLP Models**: [Tiedemann & Thottingal, 2020 - OPUS-MT](https://aclanthology.org/2020.eamt-1.61/)
- **T5 Paraphrasing**: [Raffel et al., 2020 - Exploring the Limits of Transfer Learning](https://arxiv.org/abs/1910.10683)

This system provides production-ready text augmentation optimized for LLM fine-tuning with research-backed methods and flexible configuration.
