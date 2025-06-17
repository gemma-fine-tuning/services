# Configuration and presets for preprocessing

PREPROCESSING_PRESETS = {
    # Domain Adaptation Presets
    "medical": {
        "description": "Medical domain adaptation",
        "cleaning": {
            "preserve_medical_terms": True,
            "normalize_measurements": True,
            "remove_pii": True,
        },
        "formatting": {
            "type": "instruction",
            "field_mappings": {
                "system_field": {
                    "type": "template",
                    "value": "You are a helpful medical assistant that provides accurate and helpful information.",
                },
                "user_field": {"type": "column", "value": "query"},
                "assistant_field": {"type": "column", "value": "response"},
            },
        },
        "augmentation": {
            "enabled": True,
            "lightweight": False,
            "augmentation_factor": 1.3,
            "pipeline_config": {
                "enable_back_translation": True,
                "enable_paraphrasing": True,
                "enable_eda": True,
                "back_translation_probability": 0.2,
                "paraphrasing_probability": 0.3,
                "eda_probability": 0.4,
                "eda_alpha_sr": 0.1,
                "eda_alpha_ri": 0.1,
                "eda_alpha_rs": 0.1,
                "eda_p_rd": 0.1,
                "intermediate_lang": "fr",
            },
        },
    },
    "legal": {
        "description": "Legal domain adaptation",
        "cleaning": {"preserve_legal_terms": True, "normalize_citations": True},
        "formatting": {
            "type": "instruction",
            "field_mappings": {
                "system_field": {
                    "type": "template",
                    "value": "You are a legal assistant that provides accurate legal information.",
                },
                "user_field": {"type": "column", "value": "input"},
                "assistant_field": {"type": "column", "value": "output"},
            },
        },
        "augmentation": {"enabled": False},
    },
    "financial": {
        "description": "Financial domain adaptation",
        "cleaning": {"preserve_financial_terms": True, "normalize_currencies": True},
        "formatting": {
            "type": "instruction",
            "field_mappings": {
                "system_field": {
                    "type": "template",
                    "value": "You are a financial advisor that provides helpful financial guidance.",
                },
                "user_field": {"type": "column", "value": "input"},
                "assistant_field": {"type": "column", "value": "output"},
            },
        },
        "augmentation": {"enabled": False},
    },
    # Task Adaptation Presets
    "question_answering": {
        "description": "Question answering task",
        "formatting": {
            "type": "qa",
            "field_mappings": {
                "user_field": {
                    "type": "template",
                    "value": "Context: {context}\nQuestion: {question}",
                },
                "assistant_field": {"type": "column", "value": "answer"},
            },
        },
        "augmentation": {
            "enabled": True,
            "lightweight": False,
            "augmentation_factor": 1.4,
            "pipeline_config": {
                "enable_back_translation": True,
                "enable_paraphrasing": True,
                "enable_eda": True,
                "back_translation_probability": 0.2,
                "paraphrasing_probability": 0.3,
                "eda_probability": 0.5,
                "eda_alpha_sr": 0.1,
                "eda_alpha_ri": 0.1,
                "eda_alpha_rs": 0.1,
                "eda_p_rd": 0.1,
                "intermediate_lang": "fr",
            },
        },
    },
    "text_classification": {
        "description": "Text classification task",
        "formatting": {
            "type": "classification",
            "field_mappings": {
                "user_field": {
                    "type": "template",
                    "value": "Classify the following text: {text}",
                },
                "assistant_field": {"type": "column", "value": "output"},
            },
        },
        "augmentation": {"enabled": False},
    },
    "code_generation": {
        "description": "Code generation task",
        "cleaning": {"preserve_code_structure": True, "normalize_indentation": True},
        "formatting": {
            "type": "instruction",
            "field_mappings": {
                "system_field": {
                    "type": "template",
                    "value": "You are a coding assistant that helps with programming tasks.",
                },
                "user_field": {"type": "column", "value": "input"},
                "assistant_field": {"type": "column", "value": "output"},
            },
        },
        "augmentation": {"enabled": False},
    },
    "summarization": {
        "description": "Text summarization task",
        "formatting": {
            "type": "instruction",
            "field_mappings": {
                "user_field": {
                    "type": "template",
                    "value": "Summarize the following document:\n{document}",
                },
                "assistant_field": {"type": "column", "value": "output"},
            },
        },
        "augmentation": {"enabled": False},
    },
    # Basic preset (current behavior)
    "default": {
        "description": "Keep original format",
        "formatting": {"type": "default"},
        "augmentation": {"enabled": False},
    },
}


class ConfigManager:
    """Manages preprocessing configurations and presets"""

    def __init__(self):
        self.presets = PREPROCESSING_PRESETS

    def get_preset(self, preset_name: str) -> dict:
        """Get predefined configuration"""
        if preset_name not in self.presets:
            raise ValueError(f"Unknown preset: {preset_name}")
        return self.presets[preset_name]

    def create_config(self, base_preset: str, overrides: dict) -> dict:
        """Create custom configuration from preset"""
        if base_preset not in self.presets:
            raise ValueError(f"Unknown base preset: {base_preset}")

        config = self.presets[base_preset].copy()

        # Apply overrides
        for key, value in overrides.items():
            if isinstance(value, dict) and key in config:
                config[key].update(value)
            else:
                config[key] = value

        return config

    def validate_config(self, config: dict, dataset_schema: dict = None) -> dict:
        """Validate configuration - placeholder for future implementation"""
        return {"valid": True, "errors": []}

    def get_available_presets(self) -> dict:
        """Get list of available presets with descriptions"""
        return {
            name: preset.get("description", "") for name, preset in self.presets.items()
        }
