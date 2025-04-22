#!/bin/bash
# Script to install required packages for Seren AI system

echo "Installing required packages for Seren AI system..."

# Core ML packages
pip install transformers torch accelerate --extra-index-url https://download.pytorch.org/whl/cpu

# Model handling
pip install bitsandbytes sentencepiece protobuf ctransformers huggingface-hub einops

# Utilities
pip install scipy tqdm safetensors numpy pandas requests tenacity

# API and backend
pip install fastapi uvicorn pydantic python-dotenv httpx aiohttp websockets

# Memory and storage
pip install faiss-cpu chromadb 

# Security
pip install cryptography pyjwt pyotp 

# System utilities
pip install psutil loguru

echo "Done installing required packages."
echo "To download models, run: python -m ai_core.model_downloader"