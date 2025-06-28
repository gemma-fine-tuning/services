# Cloud Infrastructure for Gemma Fine-Tuning

> To be written and documented...

This project sets up cloud infra for runing fine tuning jobs on Gemma models using the Hugging Face ecosystem. It includes two main services: a training service and a preprocessing service, both deployed on Google Cloud Run with GPU support.

**Supports:**

- Data preprocessing using huggingface and custom uploaded datasets
- Data augmentation using NLP techniques and synthetic generation using LLM
- Fine-tuning using Huggingface or Unsloth frameworks
- Fine-tuning with PEFT (LoRA, QLoRA) and RL (GRPO, PPO)
- Logging integration with Weights & Biases or TensorBoard
- Evaluation, inference, and export of trained models

Jet Chiang & Adarsh Dubey -- Google Summer of Code 2025 @ Google DeepMind
