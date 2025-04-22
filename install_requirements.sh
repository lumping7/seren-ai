#!/bin/bash

# Install requirements for Seren AI System
echo "Installing requirements for Seren AI System..."

# Check Python version
python_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Detected Python version: $python_version"

# Basic requirements
pip install -q fastapi uvicorn pydantic numpy requests beautifulsoup4 nltk

# For vector embeddings
pip install -q faiss-cpu

# For machine learning and model hosting
pip install -q torch transformers

# For knowledge library functionality
pip install -q scikit-learn pdf2image pytesseract rouge-score

# For security
pip install -q cryptography pyca

# Download NLTK data
python3 -c "import nltk; nltk.download('punkt', quiet=True)"

echo "Installation complete!"
echo "To download the required models, run:"
echo "python -m ai_core.model_downloader --model all --quantized"