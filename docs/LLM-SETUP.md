# Setting Up Real AI Models for Seren

This guide will help you set up Ollama and configure the real AI models required by Seren to function with true AI capabilities instead of pre-defined responses.

## Installing Ollama

Ollama is a lightweight, local LLM server that allows you to run Large Language Models on your own hardware.

### On Linux

```bash
# Download and install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start the Ollama service in the background
ollama serve
```

### On macOS

1. Download Ollama from the [official website](https://ollama.com/download/mac)
2. Install and run the application

### On Windows

1. Download Ollama from the [official website](https://ollama.com/download/windows)
2. Install and run the application

## Installing the Required Models

Once Ollama is running, you need to pull the models used by Seren:

```bash
# For Qwen model (used for planning and architecture)
ollama pull qwen2:7b

# For CodeLlama model (used for coding and implementation)
ollama pull codellama:7b
```

These downloads might take some time depending on your internet connection, as the models are several gigabytes in size.

## Verifying Your Setup

To verify that the models are correctly installed:

```bash
# List all installed models
ollama list
```

You should see both `qwen2:7b` and `codellama:7b` in the list.

## Configuration

By default, Seren will look for Ollama at the default address `http://127.0.0.1:11434`. If you're running Ollama on a different host or port, you can set the environment variable:

```bash
export OLLAMA_HOST=http://your-custom-host:port
```

## Checking Ollama Status

You can manually verify that Ollama is running properly by making a request to its API:

```bash
curl http://127.0.0.1:11434/api/version
```

If Ollama is running, you should see a JSON response with version information.

## Hardware Requirements

For optimal performance, we recommend:
- At least 16GB of RAM
- A CPU with AVX2 instructions
- At least 20GB of free disk space

While a GPU is not strictly required, having a CUDA-compatible NVIDIA GPU will significantly improve performance.

## Starting Seren with Real LLMs

Once you have Ollama running with the required models, simply restart the Seren application. It will automatically detect Ollama and use the real models instead of pre-defined responses.

## Troubleshooting

If you encounter issues:

1. **Ollama Not Starting**: Check if you have sufficient permissions or if another process is using port 11434.
2. **Models Not Found**: Verify the model names in your Ollama installation match what Seren expects (`qwen2:7b` and `codellama:7b`).
3. **Out of Memory Errors**: You may need to reduce model contexts or use smaller models if you're on limited hardware.
4. **Connection Refused**: Ensure Ollama is running and accessible on the expected host and port.

For more detailed troubleshooting, check Ollama's logs and the Seren server logs.