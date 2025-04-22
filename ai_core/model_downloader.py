"""
Model Downloader for Seren

Downloads and sets up local AI models for offline operation,
including Qwen2.5-omni-7b and OlympicCoder-7B models.
"""

import os
import sys
import json
import logging
import argparse
import hashlib
import shutil
import time
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Tuple
import requests
from tqdm import tqdm
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Model configuration - with Hugging Face repository IDs and local paths
MODEL_CONFIGS = {
    "qwen": {
        "name": "Qwen2.5-omni-7b",
        "repo_id": "Qwen/Qwen2.5-7B-Omni",
        "local_dir": "models/qwen2.5-omni-7b",
        "description": "Advanced multi-purpose model for general queries",
        "min_disk_space": 15.0,  # GB
        "files": ["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"],
        "quantized": {
            "name": "Qwen2.5-omni-7b-GPTQ",
            "repo_id": "Qwen/Qwen2.5-7B-Omni-GPTQ",
            "local_dir": "models/qwen2.5-omni-7b-quantized",
            "min_disk_space": 4.0,  # GB
            "files": ["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"]
        }
    },
    "olympic": {
        "name": "OlympicCoder-7B",
        "repo_id": "TheBloke/OlympicCoder-7B",
        "local_dir": "models/olympic-coder-7b",
        "description": "Specialized model for code generation tasks",
        "min_disk_space": 15.0,  # GB
        "files": ["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"],
        "quantized": {
            "name": "OlympicCoder-7B-GGUF",
            "repo_id": "TheBloke/OlympicCoder-7B-GGUF",
            "local_dir": "models/olympic-coder-7b-quantized",
            "min_disk_space": 4.0,  # GB
            "filename": "olympiccoder-7b.q4_K_M.gguf",  # Specific file for GGUF
            "files": ["olympiccoder-7b.q4_K_M.gguf"]
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
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.join(self.base_dir, "models"), exist_ok=True)
        
        # HF Auth token (if provided)
        self.use_auth = use_auth
        self.auth_token = os.environ.get("HF_TOKEN", None)
        
        if use_auth and not self.auth_token:
            logger.warning("Authentication requested but HF_TOKEN not found in environment variables")
        
        # Tracking
        self.downloads = {}
        
        logger.info(f"Model Downloader initialized. Base directory: {self.base_dir}")
    
    def _get_available_disk_space(self) -> float:
        """Get available disk space in GB"""
        disk_usage = psutil.disk_usage(self.base_dir)
        free_space_gb = disk_usage.free / (1024**3)  # Convert bytes to GB
        return free_space_gb
    
    def _create_progress_bar(self, total_size: int, desc: str = "Downloading") -> tqdm:
        """Create a progress bar for downloads"""
        return tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            desc=desc,
            ascii=True
        )
    
    def _download_file(self, url: str, dest_path: str) -> bool:
        """
        Download a file with progress tracking
        
        Args:
            url: URL to download from
            dest_path: Destination file path
            
        Returns:
            Success status
        """
        try:
            # Create parent directories if they don't exist
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            
            # Set up headers
            headers = {}
            if self.use_auth and self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"
            
            # Make request
            response = requests.get(url, headers=headers, stream=True)
            response.raise_for_status()
            
            # Get total size
            total_size = int(response.headers.get('content-length', 0))
            
            # Create progress bar
            progress = self._create_progress_bar(total_size, desc=f"Downloading {os.path.basename(dest_path)}")
            
            # Download with progress
            with open(dest_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        progress.update(len(chunk))
            
            progress.close()
            
            # Verify file exists
            if not os.path.exists(dest_path):
                logger.error(f"Failed to download {dest_path}")
                return False
            
            # Verify file size
            file_size = os.path.getsize(dest_path)
            if total_size > 0 and file_size != total_size:
                logger.error(f"File size mismatch for {dest_path}. Expected {total_size}, got {file_size}")
                return False
            
            logger.info(f"Successfully downloaded {dest_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error downloading {url}: {str(e)}")
            # Clean up partial download if it exists
            if os.path.exists(dest_path):
                os.remove(dest_path)
            return False
    
    def _download_from_huggingface(self, repo_id: str, filename: str, dest_path: str) -> bool:
        """
        Download a file from Hugging Face
        
        Args:
            repo_id: Hugging Face repository ID
            filename: Filename in the repository
            dest_path: Destination file path
            
        Returns:
            Success status
        """
        # Use Hugging Face API to get file
        url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
        return self._download_file(url, dest_path)
    
    def _verify_file_integrity(self, file_path: str, expected_hash: str = None) -> bool:
        """
        Verify file integrity using hash if provided
        
        Args:
            file_path: Path to the file
            expected_hash: Expected hash (if available)
            
        Returns:
            True if verified, False otherwise
        """
        # If no hash provided, just check file exists and is not empty
        if not expected_hash:
            return os.path.exists(file_path) and os.path.getsize(file_path) > 0
        
        # If hash provided, verify it
        try:
            # Calculate SHA256 hash
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            
            # Compare with expected hash
            calculated_hash = sha256_hash.hexdigest()
            return calculated_hash == expected_hash
        
        except Exception as e:
            logger.error(f"Error verifying {file_path}: {str(e)}")
            return False
    
    def download_model(self, model_key: str, quantized: bool = False) -> bool:
        """
        Download a specific model
        
        Args:
            model_key: Key for the model (qwen, olympic)
            quantized: Whether to download the quantized version
            
        Returns:
            Success status
        """
        if model_key not in MODEL_CONFIGS:
            logger.error(f"Unknown model key: {model_key}")
            return False
        
        # Get model config
        model_config = MODEL_CONFIGS[model_key]
        
        # Use quantized version if requested
        if quantized and "quantized" in model_config:
            model_config = model_config["quantized"]
        
        # Log what we're about to do
        logger.info(f"Downloading {model_config['name']}...")
        
        # Check disk space
        free_space_gb = self._get_available_disk_space()
        required_space_gb = model_config.get("min_disk_space", 20.0)
        
        if free_space_gb < required_space_gb:
            logger.error(f"Not enough disk space. Required: {required_space_gb} GB, Available: {free_space_gb:.2f} GB")
            return False
        
        # Create model directory
        model_dir = os.path.join(self.base_dir, model_config["local_dir"])
        os.makedirs(model_dir, exist_ok=True)
        
        # Download each required file
        success = True
        for filename in model_config.get("files", []):
            # For GGUF models, the filename might be different
            if quantized and model_key == "olympic" and "filename" in model_config:
                # In this case, filename is the specific GGUF file we want
                dest_path = os.path.join(model_dir, model_config["filename"])
                source_filename = model_config["filename"]
            else:
                dest_path = os.path.join(model_dir, filename)
                source_filename = filename
            
            # Skip if file already exists and is valid
            if os.path.exists(dest_path) and self._verify_file_integrity(dest_path):
                logger.info(f"File {dest_path} already exists and is valid. Skipping.")
                continue
            
            # Download file
            file_success = self._download_from_huggingface(
                model_config["repo_id"],
                source_filename,
                dest_path
            )
            
            if not file_success:
                logger.error(f"Failed to download {source_filename}")
                success = False
                break
        
        # Final check
        if success:
            logger.info(f"Successfully downloaded {model_config['name']}")
            # Track what we've downloaded
            self.downloads[model_key] = {
                "name": model_config["name"],
                "path": model_dir,
                "quantized": quantized,
                "timestamp": time.time()
            }
        else:
            logger.error(f"Failed to download {model_config['name']}")
        
        return success
    
    def download_all_models(self, quantized: bool = False) -> Dict[str, bool]:
        """
        Download all required models
        
        Args:
            quantized: Whether to download quantized versions
            
        Returns:
            Dict of model keys and success status
        """
        results = {}
        
        for model_key in MODEL_CONFIGS:
            results[model_key] = self.download_model(model_key, quantized)
        
        # Summary
        success_count = sum(1 for result in results.values() if result)
        total_count = len(results)
        
        logger.info(f"Downloaded {success_count} of {total_count} models")
        
        return results
    
    def verify_models(self) -> Dict[str, bool]:
        """
        Verify that all required models are downloaded and valid
        
        Returns:
            Dict of model keys and validation status
        """
        results = {}
        
        for model_key, model_config in MODEL_CONFIGS.items():
            # Check standard model
            model_dir = os.path.join(self.base_dir, model_config["local_dir"])
            model_valid = os.path.exists(model_dir)
            
            # Check all required files
            if model_valid:
                for filename in model_config.get("files", []):
                    file_path = os.path.join(model_dir, filename)
                    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                        model_valid = False
                        break
            
            results[model_key] = model_valid
            
            # Check quantized model if available
            if "quantized" in model_config:
                quant_config = model_config["quantized"]
                quant_dir = os.path.join(self.base_dir, quant_config["local_dir"])
                quant_valid = os.path.exists(quant_dir)
                
                # Check all required files
                if quant_valid:
                    for filename in quant_config.get("files", []):
                        file_path = os.path.join(quant_dir, filename)
                        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                            quant_valid = False
                            break
                
                results[f"{model_key}_quantized"] = quant_valid
        
        return results

def main():
    """Command-line interface for model downloader"""
    parser = argparse.ArgumentParser(description="Download models for Seren")
    parser.add_argument("--model", type=str, help="Model to download (qwen, olympic, or all)")
    parser.add_argument("--quantized", action="store_true", help="Download quantized models")
    parser.add_argument("--verify", action="store_true", help="Verify downloaded models")
    parser.add_argument("--auth", action="store_true", help="Use HF_TOKEN for authentication")
    
    args = parser.parse_args()
    
    downloader = ModelDownloader(use_auth=args.auth)
    
    if args.verify:
        results = downloader.verify_models()
        print("Model Verification Results:")
        for model_key, valid in results.items():
            status = "✓ Valid" if valid else "✗ Missing or Invalid"
            print(f"  {model_key}: {status}")
        return
    
    if not args.model or args.model == "all":
        results = downloader.download_all_models(quantized=args.quantized)
        print("Download Results:")
        for model_key, success in results.items():
            status = "✓ Success" if success else "✗ Failed"
            print(f"  {model_key}: {status}")
    else:
        success = downloader.download_model(args.model, quantized=args.quantized)
        status = "✓ Success" if success else "✗ Failed"
        print(f"Download of {args.model}: {status}")

if __name__ == "__main__":
    main()