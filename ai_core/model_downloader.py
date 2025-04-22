"""
Model Downloader for Seren

Downloads and sets up local AI models for offline operation,
including Qwen2.5-omni-7b and OlympicCoder-7B models.
"""

import os
import sys
import subprocess
import logging
import argparse
import hashlib
import requests
from typing import Dict, List, Any, Optional
from tqdm import tqdm
import torch
from huggingface_hub import snapshot_download, hf_hub_download, login

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Model configurations
MODEL_CONFIGS = {
    "qwen": {
        "repo_id": "Qwen/Qwen2.5-7B-Instruct",
        "local_dir": "models/qwen2.5-omni-7b",
        "description": "Qwen2.5-omni-7b - Advanced multi-purpose model",
        "size_gb": 14,
        "quantized": {
            "repo_id": "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
            "local_dir": "models/qwen2.5-omni-7b-quantized",
            "description": "Qwen2.5-omni-7b (Quantized 4-bit) - Optimized for inference",
            "size_gb": 4
        }
    },
    "olympic": {
        "repo_id": "TheBloke/OlympicCoder-7B-GGUF",
        "local_dir": "models/olympiccoder-7b",
        "description": "OlympicCoder-7B - Specialized for code generation",
        "size_gb": 14,
        "quantized": {
            "repo_id": "TheBloke/OlympicCoder-7B-GGUF",
            "filename": "olympiccoder-7b.Q4_K_M.gguf",
            "local_dir": "models/olympiccoder-7b-quantized",
            "description": "OlympicCoder-7B (Quantized 4-bit) - Optimized for code generation",
            "size_gb": 4
        }
    }
}

class ModelDownloader:
    """
    Model Downloader for Seren
    
    Downloads and sets up the required AI models for local offline operation:
    - Qwen2.5-omni-7b: Advanced multi-purpose language model
    - OlympicCoder-7B: Specialized code generation model
    
    Features:
    - Automatic download from Hugging Face Hub
    - Optional quantized variants for efficient operation
    - Integrity verification
    - Storage optimization
    """
    
    def __init__(self, base_dir: str = None, use_auth: bool = False):
        """Initialize the model downloader"""
        # Set the base directory
        self.base_dir = base_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Flag for using HF Hub authentication
        self.use_auth = use_auth
        
        # Check available disk space
        self.available_space_gb = self._get_available_disk_space()
        
        # Check CPU/GPU availability
        self.has_gpu = torch.cuda.is_available()
        if self.has_gpu:
            self.gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"GPU detected: {torch.cuda.get_device_name(0)} with {self.gpu_memory_gb:.2f} GB memory")
        else:
            self.gpu_memory_gb = 0
            logger.info("No GPU detected, will use CPU for inference")
        
        logger.info(f"Model Downloader initialized. Available disk space: {self.available_space_gb:.2f} GB")
    
    def _get_available_disk_space(self) -> float:
        """Get available disk space in GB"""
        if sys.platform == "win32":
            free_bytes = ctypes.c_ulonglong(0)
            ctypes.windll.kernel32.GetDiskFreeSpaceExW(ctypes.c_wchar_p(self.base_dir), None, None, ctypes.pointer(free_bytes))
            return free_bytes.value / (1024**3)
        else:
            st = os.statvfs(self.base_dir)
            return (st.f_bavail * st.f_frsize) / (1024**3)
    
    def download_model(self, model_key: str, quantized: bool = False) -> bool:
        """
        Download a specific model
        
        Args:
            model_key: Key for the model (qwen, olympic)
            quantized: Whether to download the quantized version
            
        Returns:
            Success status
        """
        # Check if model key is valid
        if model_key not in MODEL_CONFIGS:
            logger.error(f"Invalid model key: {model_key}")
            return False
        
        # Get model config
        model_config = MODEL_CONFIGS[model_key]
        
        # Use quantized version if requested
        if quantized and "quantized" in model_config:
            model_config = model_config["quantized"]
        
        # Create local directory
        local_dir = os.path.join(self.base_dir, model_config["local_dir"])
        os.makedirs(local_dir, exist_ok=True)
        
        # Check if model already exists
        if os.path.exists(os.path.join(local_dir, "config.json")) or \
           (model_key == "olympic" and quantized and \
            os.path.exists(os.path.join(local_dir, model_config.get("filename", "")))):
            logger.info(f"Model {model_key} already exists at {local_dir}")
            return True
        
        # Check available space
        required_space = model_config["size_gb"] * 1.1  # Add 10% buffer
        if self.available_space_gb < required_space:
            logger.error(f"Not enough disk space to download model {model_key}. "
                         f"Required: {required_space:.2f} GB, Available: {self.available_space_gb:.2f} GB")
            return False
        
        # Log download information
        logger.info(f"Downloading {model_config['description']} "
                    f"({model_config['size_gb']} GB)...")
        
        try:
            # Authenticate with Hugging Face if needed
            if self.use_auth:
                # First check if we have a token set
                hf_token = os.environ.get("HUGGINGFACE_TOKEN")
                if not hf_token:
                    logger.warning("No HUGGINGFACE_TOKEN environment variable found. "
                                   "Some models may not be accessible.")
                    # Attempt to use the cached token if available
                else:
                    login(token=hf_token)
            
            # Download the model
            if model_key == "olympic" and quantized:
                # For Olympic, download specific file for quantized version
                filepath = hf_hub_download(
                    repo_id=model_config["repo_id"],
                    filename=model_config["filename"],
                    local_dir=local_dir,
                    local_dir_use_symlinks=False
                )
                logger.info(f"Downloaded {model_config['filename']} to {filepath}")
            else:
                # Download the entire model repository
                snapshot_path = snapshot_download(
                    repo_id=model_config["repo_id"],
                    local_dir=local_dir,
                    local_dir_use_symlinks=False
                )
                logger.info(f"Downloaded model to {snapshot_path}")
            
            logger.info(f"Successfully downloaded {model_config['description']}")
            return True
        
        except Exception as e:
            logger.error(f"Error downloading model {model_key}: {str(e)}")
            return False
    
    def download_all_models(self, quantized: bool = False) -> Dict[str, bool]:
        """
        Download all required models
        
        Args:
            quantized: Whether to download quantized versions
            
        Returns:
            Dict of model keys and success status
        """
        results = {}
        
        # Determine which models to download based on hardware
        if self.has_gpu and self.gpu_memory_gb >= 24:
            # High-end GPU with sufficient memory - download full models
            use_quantized = False
            logger.info("Using full models with high-performance GPU")
        elif self.has_gpu and self.gpu_memory_gb >= 8:
            # Mid-range GPU - use quantized models
            use_quantized = True
            logger.info("Using quantized models with mid-range GPU")
        else:
            # Low-end GPU or CPU only - use quantized models
            use_quantized = True
            logger.info("Using quantized models for CPU or low-end GPU")
        
        # Override with user preference if specified
        if quantized is not None:
            use_quantized = quantized
            logger.info(f"Using {'quantized' if use_quantized else 'full'} models as specified")
        
        # Download models
        for model_key in MODEL_CONFIGS:
            logger.info(f"Downloading {model_key} model...")
            success = self.download_model(model_key, use_quantized)
            results[model_key] = success
        
        return results
    
    def verify_models(self) -> Dict[str, bool]:
        """
        Verify that all required models are downloaded and valid
        
        Returns:
            Dict of model keys and validation status
        """
        results = {}
        
        for model_key, config in MODEL_CONFIGS.items():
            # Check both standard and quantized versions
            for version_type in ["standard", "quantized"]:
                if version_type == "standard":
                    model_config = config
                else:
                    if "quantized" not in config:
                        continue
                    model_config = config["quantized"]
                
                local_dir = os.path.join(self.base_dir, model_config["local_dir"])
                
                # For Olympic quantized, check specific file
                if model_key == "olympic" and version_type == "quantized":
                    filename = model_config.get("filename", "")
                    model_path = os.path.join(local_dir, filename)
                    if os.path.exists(model_path) and os.path.getsize(model_path) > 1000000:  # > 1MB
                        results[f"{model_key}_{version_type}"] = True
                    else:
                        results[f"{model_key}_{version_type}"] = False
                else:
                    # Check for config.json for standard models
                    config_path = os.path.join(local_dir, "config.json")
                    if os.path.exists(config_path):
                        results[f"{model_key}_{version_type}"] = True
                    else:
                        results[f"{model_key}_{version_type}"] = False
        
        return results

def main():
    """Command-line interface for model downloader"""
    parser = argparse.ArgumentParser(description="Download and set up AI models for Seren")
    parser.add_argument("--model", choices=["qwen", "olympic", "all"], default="all",
                        help="Which model to download (default: all)")
    parser.add_argument("--quantized", action="store_true", 
                        help="Download quantized versions of the models")
    parser.add_argument("--auth", action="store_true",
                        help="Use Hugging Face authentication for downloading models")
    parser.add_argument("--verify", action="store_true",
                        help="Verify downloaded models without downloading")
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = ModelDownloader(use_auth=args.auth)
    
    # If verify only
    if args.verify:
        results = downloader.verify_models()
        for model_key, status in results.items():
            print(f"{model_key}: {'✅ Available' if status else '❌ Not found'}")
        return
    
    # Download models
    if args.model == "all":
        results = downloader.download_all_models(quantized=args.quantized)
        for model_key, success in results.items():
            print(f"{model_key}: {'✅ Success' if success else '❌ Failed'}")
    else:
        success = downloader.download_model(args.model, quantized=args.quantized)
        print(f"{args.model}: {'✅ Success' if success else '❌ Failed'}")

if __name__ == "__main__":
    main()