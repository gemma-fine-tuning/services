import uuid
import re
import logging
from typing import List, Dict, Any
from datasets import Dataset
from schema import ProcessingResult
from augmentation.text_augmentor import create_augmentation_pipeline

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Exposes the process_dataset method to handle all dataset processing tasks.
    Pretty much everything else is private or internal functions, do not call
    for example augmentation directly since it requires config handling.
    """

    def __init__(self, storage_manager=None):
        self.whitespace_pattern = re.compile(r"\s+")
        self.storage_manager = storage_manager
        self.augmentation_pipeline = None

    async def process_dataset(
        self, dataset: List[Dict], config: Dict[str, Any]
    ) -> ProcessingResult:
        """
        Process dataset with configuration
        Handles EVERYTHING based on your config
        """
        try:
            # Convert to HuggingFace Dataset
            hf_dataset = Dataset.from_list(dataset)

            # Apply format conversion if needed
            format_config = config.get("format_config", {})
            if format_config.get("type") != "default":
                hf_dataset = hf_dataset.map(
                    lambda x: self._format_conversation(x, format_config),
                    desc="Formatting conversations",
                    batched=True,
                )

            # Clean text fields
            if config.get("normalize_whitespace", True):
                hf_dataset = hf_dataset.map(
                    lambda x: {k: self._normalize_whitespace(v) for k, v in x.items()},
                    desc="Normalizing whitespace",
                )

            # Apply data augmentation if enabled
            augmentation_config = config.get("augmentation_config", {})
            if augmentation_config.get("enabled", False):
                hf_dataset = self._apply_augmentation(hf_dataset, augmentation_config)

            # Generate unique ID for processed dataset
            processed_id = str(uuid.uuid4())

            if not self.storage_manager:
                raise ValueError("Storage manager not initialized")

            # Handle train/test split
            if config.get("train_test_split", False):
                split = hf_dataset.train_test_split(
                    test_size=config.get("test_size", 0.2), shuffle=True, seed=42
                )
                train_dataset = split["train"]
                test_dataset = split["test"]

                # Save splits to storage
                train_blob_name = f"processed_datasets/{processed_id}_train.json"
                test_blob_name = f"processed_datasets/{processed_id}_test.json"

                train_gcs_path = await self.storage_manager.upload_data(
                    train_dataset.to_list(), train_blob_name
                )
                test_gcs_path = await self.storage_manager.upload_data(
                    test_dataset.to_list(), test_blob_name
                )

                return ProcessingResult(
                    processed_dataset_id=processed_id,
                    original_count=len(hf_dataset),
                    processed_count=len(hf_dataset),
                    train_count=len(train_dataset),
                    test_count=len(test_dataset),
                    train_gcs_path=train_gcs_path,
                    test_gcs_path=test_gcs_path,
                    sample_comparison={
                        "original": dataset[0] if len(dataset) > 0 else None,
                        "processed": train_dataset[0]
                        if len(train_dataset) > 0
                        else None,
                    },
                )
            else:
                # Save full dataset
                processed_blob_name = f"processed_datasets/{processed_id}_train.json"
                gcs_path = await self.storage_manager.upload_data(
                    hf_dataset.to_list(), processed_blob_name
                )

                return ProcessingResult(
                    processed_dataset_id=processed_id,
                    original_count=len(dataset),
                    processed_count=len(hf_dataset),
                    gcs_path=gcs_path,
                    sample_comparison={
                        "original": dataset[0] if len(dataset) > 0 else None,
                        "processed": hf_dataset[0] if len(hf_dataset) > 0 else None,
                    },
                )

        except Exception as e:
            logger.error(f"Error processing dataset: {str(e)}")
            raise

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace by removing extra spaces"""
        return (
            self.whitespace_pattern.sub(" ", text).strip()
            if isinstance(text, str)
            else text
        )

    def _format_conversation(self, example: Dict, format_config: Dict) -> Dict:
        """Format a single example into conversation format"""
        if format_config.get("type") == "default":
            return example

        # Extract config
        system_message = format_config.get("system_message", "")
        user_template = format_config.get("user_prompt_template", "")
        include_system = format_config.get("include_system", False)
        field_mappings = format_config.get("field_mappings", {})

        # Build template variables
        template_vars = {
            placeholder: example.get(field, "")
            for placeholder, field in field_mappings.items()
        }

        # Format user message
        try:
            user_content = (
                user_template.format(**template_vars)
                if user_template
                else example.get("input", "")
            )
        except KeyError as e:
            logger.warning(f"Missing field mapping for {e}")
            user_content = user_template

        # Get assistant response
        output_field = next(
            (
                field
                for placeholder, field in field_mappings.items()
                if placeholder.lower() in ["answer", "output", "response", "sql"]
            ),
            None,
        )
        assistant_content = (
            example.get(output_field, "") if output_field else example.get("output", "")
        )

        # Build messages
        messages = []
        if include_system and system_message:
            messages.append({"role": "system", "content": system_message})
        messages.extend(
            [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]
        )

        return {"messages": messages}

    def clean_text(self, dataset: List[Dict], cleaning_config: Dict):
        """Text cleaning pipeline - placeholder for future implementation"""
        pass

    def format_data(self, dataset: List[Dict], format_config: Dict):
        """Data formatting - uses existing _format_conversation method"""
        pass

    def _apply_augmentation(
        self, hf_dataset: Dataset, augmentation_config: Dict
    ) -> Dataset:
        """
        Apply data augmentation to the dataset.
        NOTE: This process can often take a while if the dataset is large.
        For production, we cannot just let the user run this because it will block the server.
        Running some of the LLM-based augmentations locally will take forever on huge datasets.
        """
        try:
            # Initialize augmentation pipeline if not already done
            if self.augmentation_pipeline is None:
                lightweight = augmentation_config.get("lightweight", True)
                if lightweight:
                    self.augmentation_pipeline = create_augmentation_pipeline(
                        eda=True,
                        back_translation=False,
                        paraphrasing=False,
                        **augmentation_config.get("pipeline_config", {}),
                    )
                else:
                    self.augmentation_pipeline = create_augmentation_pipeline(
                        **augmentation_config.get("pipeline_config", {}),
                    )

            # Get augmentation factor
            augmentation_factor = augmentation_config.get("augmentation_factor", 1.5)

            if augmentation_factor <= 1.0:
                logger.info("Augmentation factor <= 1.0, skipping augmentation")
                return hf_dataset

            # Convert to list for augmentation
            dataset_list = hf_dataset.to_list()

            # Apply augmentation
            logger.info(f"Applying augmentation with factor {augmentation_factor}")
            augmented_list = self.augmentation_pipeline.augment_dataset(
                dataset_list, augmentation_factor=augmentation_factor
            )

            # Convert back to HuggingFace Dataset
            augmented_dataset = Dataset.from_list(augmented_list)

            logger.info(
                f"Dataset augmented from {len(hf_dataset)} to {len(augmented_dataset)} samples"
            )
            return augmented_dataset

        except Exception as e:
            logger.error(f"Augmentation failed: {e}")
            logger.warning("Continuing without augmentation")
            return hf_dataset

    def validate_quality(self, original: List[Dict], processed: List[Dict]):
        """Quality validation - placeholder for future implementation"""
        pass

    def preview_processing(
        self, dataset: List[Dict], config: Dict, sample_size: int = 10
    ):
        """Preview results - placeholder for future implementation"""
        pass
