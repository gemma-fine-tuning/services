"""
# Text Augmentation Module

This module provides comprehensive text augmentation capabilities including:
- EDA (Easy Data Augmentation)
- Back Translation
- Paraphrasing
- Conversational augmentation

Usage:
```python
from preprocessing.augmentation import TextAugmentationPipeline

# Or import specific components
from preprocessing.augmentation.text_augmentor import (
    create_augmentation_pipeline,
    create_eda_only_pipeline,
)
```
"""

# Import main classes and functions to make them available at package level
from .text_augmentor import (
    TextAugmentationPipeline,
    create_augmentation_pipeline,
    create_eda_only_pipeline,
    BaseAugmentor,
    BackTranslationAugmentor,
    ParaphraseAugmentor,
    EDAugmentor,
)

# Import EDA functions for direct use
from .eda import eda

__all__ = [
    "TextAugmentationPipeline",
    "create_augmentation_pipeline",
    "create_eda_only_pipeline",
    "BaseAugmentor",
    "BackTranslationAugmentor",
    "ParaphraseAugmentor",
    "EDAugmentor",
    "eda",
]
