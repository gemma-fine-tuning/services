# Configuration and presets for preprocessing

PREPROCESSING_PRESETS = {
    # Domain Adaptation Presets
    'medical': {
        'description': 'Medical domain adaptation',
        'cleaning': {
            'preserve_medical_terms': True,
            'normalize_measurements': True,
            'remove_pii': True
        },
        'formatting': {
            'type': 'instruction',
            'system_message': 'You are a helpful medical assistant that provides accurate and helpful information.',
            'field_mappings': {'question': 'query', 'answer': 'response'}
        },
        'augmentation': {
            'enabled': False  # Start simple
        }
    },
    
    'legal': {
        'description': 'Legal domain adaptation',
        'cleaning': {
            'preserve_legal_terms': True,
            'normalize_citations': True
        },
        'formatting': {
            'type': 'instruction',
            'system_message': 'You are a legal assistant that provides accurate legal information.',
            'field_mappings': {'case': 'input', 'ruling': 'output'}
        },
        'augmentation': {
            'enabled': False
        }
    },
    
    'financial': {
        'description': 'Financial domain adaptation',
        'cleaning': {
            'preserve_financial_terms': True,
            'normalize_currencies': True
        },
        'formatting': {
            'type': 'instruction',
            'system_message': 'You are a financial advisor that provides helpful financial guidance.',
            'field_mappings': {'query': 'input', 'advice': 'output'}
        },
        'augmentation': {
            'enabled': False
        }
    },
    
    # Task Adaptation Presets
    'question_answering': {
        'description': 'Question answering task',
        'formatting': {
            'type': 'qa',
            'field_mappings': {'question': 'question', 'answer': 'answer', 'context': 'context'},
            'user_prompt_template': 'Context: {context}\nQuestion: {question}'
        },
        'augmentation': {
            'enabled': False
        }
    },
    
    'text_classification': {
        'description': 'Text classification task',
        'formatting': {
            'type': 'classification',
            'field_mappings': {'text': 'input', 'label': 'output'},
            'user_prompt_template': 'Classify the following text: {text}'
        },
        'augmentation': {
            'enabled': False
        }
    },
    
    'code_generation': {
        'description': 'Code generation task',
        'cleaning': {
            'preserve_code_structure': True,
            'normalize_indentation': True
        },
        'formatting': {
            'type': 'instruction',
            'system_message': 'You are a coding assistant that helps with programming tasks.',
            'field_mappings': {'instruction': 'input', 'code': 'output'}
        },
        'augmentation': {
            'enabled': False
        }
    },
    
    'summarization': {
        'description': 'Text summarization task',
        'formatting': {
            'type': 'instruction',
            'field_mappings': {'document': 'input', 'summary': 'output'},
            'user_prompt_template': 'Summarize the following document:\n{document}'
        },
        'augmentation': {
            'enabled': False
        }
    },
    
    # Basic preset (current behavior)
    'default': {
        'description': 'Keep original format',
        'formatting': {'type': 'default'},
        'augmentation': {
            'enabled': False
        }
    }
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
            name: preset.get('description', '')
            for name, preset in self.presets.items()
        }