"""
# Text Augmentation Module

This module provides comprehensive text augmentation capabilities including:
- EDA (Easy Data Augmentation)
- Back Translation
- Paraphrasing
- Conversational augmentation

The module provides a high level interface for configuring and running augmentations:
```python
from preprocessing.augmentation import (
    AugmentationManager,
    AugmentationConfig,
    run_augment_pipeline,
)
# Create an augmentation manager with desired configuration
manager = AugmentationManager()
config = AugmentationConfig(
    enabled_methods=[AugmentationMethod.EDA, AugmentationMethod.BACK_TRANSLATION],
    augmentation_factor=1.5,
    lightweight_mode=True,
)
manager.configure(config)
augmented_dataset, result = manager.augment_dataset(dataset)
```

However, for most use cases, you can use the simplified pipeline interface:
```python
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
"""

# Import new simplified interface
from .augmentation_manager import (
    AugmentationManager,
    AugmentationConfig,
    AugmentationMethod,
    AugmentationResult,
    run_augment_pipeline,
)

__all__ = [
    # Main interface
    "AugmentationManager",
    "AugmentationConfig",
    "AugmentationMethod",
    "AugmentationResult",
    "run_augment_pipeline",
]
