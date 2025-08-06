# Running Fine-tuned Models in GGUF Format

This provides a guide to run fine tuned models locally. For on the cloud, we have provided direct deployment options already using various backend providers :)

This assumes that you have the model downloaded locally (we will add docs for on hub soon)

## llama.cpp

1. Install `llama.cpp` following the [llama.cpp Installation Guide](https://github.com/ggerganov/llama.cpp#installation)

2. Run the model:

   ```bash
   llama-cli -m path-to-gguf-file -sys "system-prompt-if-any"
   ```

## Ollama

1. Install Ollama: [Ollama Installation Guide](https://ollama.com/docs/installation)

2. Start Ollama server:

   ```bash
   ollama serve
   ```

3. Create `Modelfile`

   ```txt
   FROM path-to-gguf-file

   SYSTEM """
   system-prompt-if-any
   """

   PARAMETER...
   ```

4. Create a model using the GGUF file:

   ```bash
   ollama create <model_name> -f Modelfile
   ```

5. Run the model:

   ```bash
    ollama run <model_name>
   ```

6. If the model is on HF Hub:

   ```bash
   ollama run hf.co/{username}/{repository}:{quantization}
   ```

   [references](https://huggingface.co/docs/hub/en/ollama)
