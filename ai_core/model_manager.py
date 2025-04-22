"""
Model Manager for Seren

Loads and manages locally hosted AI models for offline operation,
including Qwen2.5-omni-7b and OlympicCoder-7B models.
"""

import os
import sys
import logging
import json
import time
import torch
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from threading import Lock

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

# Import from the model downloader for model configs
from ai_core.model_downloader import MODEL_CONFIGS

# We'll use HuggingFace's transformers library
try:
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, TextIteratorStreamer
    from threading import Thread
except ImportError:
    logger.error("Transformers library not installed. Please install with: pip install transformers")
    raise

# For GGUF models (Olympic)
try:
    from ctransformers import AutoModelForCausalLM as CT_AutoModelForCausalLM
except ImportError:
    logger.warning("ctransformers library not installed. OlympicCoder-7B GGUF model will not be available.")

class ModelType(Enum):
    """Types of models"""
    QWEN = "qwen"           # Qwen2.5-omni-7b
    OLYMPIC = "olympic"     # OlympicCoder-7B
    HYBRID = "hybrid"       # Combined approach

class ModelMode(Enum):
    """Operating modes for models"""
    STANDARD = "standard"   # Full precision models
    QUANTIZED = "quantized" # Quantized models
    AUTO = "auto"           # Auto-select based on hardware

class ModelManager:
    """
    Model Manager for Seren
    
    Loads and manages locally hosted AI models for offline operation:
    - Automatically selects appropriate model format based on hardware
    - Efficiently unloads and switches models to manage memory
    - Provides a unified interface for all models
    - Supports streaming generation for responsive UI
    - Manages model state including conversation history
    
    Supports the following models:
    - Qwen2.5-omni-7b: Advanced multi-purpose language model
    - OlympicCoder-7B: Specialized code generation model
    """
    
    def __init__(self, base_dir: str = None, mode: ModelMode = ModelMode.AUTO):
        """Initialize the model manager"""
        # Set the base directory
        self.base_dir = base_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Set model mode
        self.mode = mode
        
        # Track loaded models
        self.loaded_models = {}
        self.loaded_tokenizers = {}
        
        # Monitor memory usage
        self.memory_usage = {}
        
        # Lock for model loading
        self.model_lock = Lock()
        
        # Check hardware capabilities
        self.device = self._get_device()
        self.gpu_memory_gb = self._get_gpu_memory() if torch.cuda.is_available() else 0
        
        # Select mode based on hardware if AUTO
        if self.mode == ModelMode.AUTO:
            if self.device == "cuda" and self.gpu_memory_gb >= 24:
                self.mode = ModelMode.STANDARD
                logger.info("Using standard models with high-performance GPU")
            else:
                self.mode = ModelMode.QUANTIZED
                logger.info("Using quantized models due to hardware constraints")
        
        logger.info(f"Model Manager initialized. Using {self.mode.value} models on {self.device}.")
    
    def _get_device(self) -> str:
        """Get the appropriate device for model inference"""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"  # Apple Silicon
        else:
            return "cpu"
    
    def _get_gpu_memory(self) -> float:
        """Get GPU memory in GB"""
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return 0
    
    def _get_model_path(self, model_type: ModelType) -> str:
        """Get the path to a model based on type and mode"""
        model_key = model_type.value
        
        if model_key not in MODEL_CONFIGS:
            raise ValueError(f"Invalid model type: {model_key}")
        
        # Get model config
        model_config = MODEL_CONFIGS[model_key]
        
        # Use quantized version if in quantized mode
        if self.mode == ModelMode.QUANTIZED and "quantized" in model_config:
            model_config = model_config["quantized"]
        
        # Get the local path
        return os.path.join(self.base_dir, model_config["local_dir"])
    
    def _verify_model_exists(self, model_type: ModelType) -> bool:
        """Verify that a model exists at the expected path"""
        model_path = self._get_model_path(model_type)
        
        # For Olympic GGUF model in quantized mode
        if model_type == ModelType.OLYMPIC and self.mode == ModelMode.QUANTIZED:
            filename = MODEL_CONFIGS["olympic"]["quantized"].get("filename", "")
            file_path = os.path.join(model_path, filename)
            return os.path.exists(file_path)
        
        # For other models, check for the model directory and config.json
        config_path = os.path.join(model_path, "config.json")
        return os.path.exists(config_path)
    
    def load_model(self, model_type: ModelType) -> bool:
        """
        Load a specific model into memory
        
        Args:
            model_type: Type of model to load
            
        Returns:
            Success status
        """
        model_key = model_type.value
        
        # Check if already loaded
        if model_key in self.loaded_models:
            logger.info(f"Model {model_key} already loaded")
            return True
        
        # Verify model exists
        if not self._verify_model_exists(model_type):
            logger.error(f"Model {model_key} not found at expected path")
            return False
        
        # Get model path
        model_path = self._get_model_path(model_type)
        
        # Acquire lock to ensure only one model is loaded at a time
        with self.model_lock:
            try:
                start_time = time.time()
                logger.info(f"Loading {model_key} model from {model_path}...")
                
                # Load model based on type
                if model_type == ModelType.QWEN:
                    # Load Qwen model
                    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                    
                    # Determine loading parameters based on mode and device
                    load_kwargs = {}
                    if self.mode == ModelMode.QUANTIZED:
                        load_kwargs["torch_dtype"] = torch.float16
                        load_kwargs["load_in_4bit"] = True
                        load_kwargs["device_map"] = "auto"
                    else:
                        if self.device == "cuda":
                            load_kwargs["torch_dtype"] = torch.float16
                            load_kwargs["device_map"] = "auto"
                        else:
                            # For CPU, we should load in 8-bit to save memory
                            load_kwargs["load_in_8bit"] = True
                            load_kwargs["device_map"] = "auto"
                    
                    # Load the model
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        trust_remote_code=True,
                        **load_kwargs
                    )
                    
                    # Save loaded model and tokenizer
                    self.loaded_models[model_key] = model
                    self.loaded_tokenizers[model_key] = tokenizer
                
                elif model_type == ModelType.OLYMPIC:
                    if self.mode == ModelMode.QUANTIZED:
                        # Load OlympicCoder GGUF model with ctransformers
                        if 'CT_AutoModelForCausalLM' not in globals():
                            logger.error("ctransformers library not installed. Cannot load OlympicCoder GGUF model.")
                            return False
                        
                        # Get the GGUF filename
                        filename = MODEL_CONFIGS["olympic"]["quantized"].get("filename", "")
                        file_path = os.path.join(model_path, filename)
                        
                        # Load the GGUF model
                        model = CT_AutoModelForCausalLM.from_pretrained(
                            file_path,
                            model_type="llama",
                            gpu_layers=99 if self.device == "cuda" else 0
                        )
                        
                        # Load tokenizer from the repo
                        tokenizer = AutoTokenizer.from_pretrained(
                            "TheBloke/OlympicCoder-7B-GGUF", 
                            use_fast=True
                        )
                        
                    else:
                        # Load OlympicCoder model with transformers
                        tokenizer = AutoTokenizer.from_pretrained(model_path)
                        
                        load_kwargs = {}
                        if self.device == "cuda":
                            load_kwargs["torch_dtype"] = torch.float16
                            load_kwargs["device_map"] = "auto"
                        else:
                            # For CPU, we should load in 8-bit to save memory
                            load_kwargs["load_in_8bit"] = True
                            load_kwargs["device_map"] = "auto"
                        
                        model = AutoModelForCausalLM.from_pretrained(
                            model_path,
                            **load_kwargs
                        )
                    
                    # Save loaded model and tokenizer
                    self.loaded_models[model_key] = model
                    self.loaded_tokenizers[model_key] = tokenizer
                
                else:
                    logger.error(f"Unknown model type: {model_key}")
                    return False
                
                # Calculate memory usage
                if model_type == ModelType.QWEN:
                    if hasattr(model, "get_memory_footprint"):
                        memory_bytes = model.get_memory_footprint()
                        self.memory_usage[model_key] = memory_bytes / (1024**3)  # Convert to GB
                
                elapsed_time = time.time() - start_time
                logger.info(f"Model {model_key} loaded in {elapsed_time:.2f} seconds")
                
                return True
            
            except Exception as e:
                logger.error(f"Error loading model {model_key}: {str(e)}")
                # Free any partially loaded components
                if model_key in self.loaded_models:
                    del self.loaded_models[model_key]
                if model_key in self.loaded_tokenizers:
                    del self.loaded_tokenizers[model_key]
                return False
    
    def unload_model(self, model_type: ModelType) -> bool:
        """
        Unload a model from memory
        
        Args:
            model_type: Type of model to unload
            
        Returns:
            Success status
        """
        model_key = model_type.value
        
        # Check if loaded
        if model_key not in self.loaded_models:
            logger.info(f"Model {model_key} not loaded")
            return True
        
        # Acquire lock
        with self.model_lock:
            try:
                logger.info(f"Unloading {model_key} model...")
                
                # Delete model and tokenizer
                del self.loaded_models[model_key]
                del self.loaded_tokenizers[model_key]
                
                # Run garbage collection to free memory
                import gc
                gc.collect()
                
                # Clear CUDA cache if using GPU
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                
                # Clear memory usage record
                if model_key in self.memory_usage:
                    del self.memory_usage[model_key]
                
                logger.info(f"Model {model_key} unloaded successfully")
                return True
            
            except Exception as e:
                logger.error(f"Error unloading model {model_key}: {str(e)}")
                return False
    
    def is_model_loaded(self, model_type: ModelType) -> bool:
        """Check if a model is currently loaded"""
        return model_type.value in self.loaded_models
    
    def generate_text(
        self,
        model_type: ModelType,
        prompt: str,
        max_length: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repetition_penalty: float = 1.1,
        streaming: bool = False,
        callback = None,
        **kwargs
    ) -> Union[str, TextIteratorStreamer]:
        """
        Generate text using a specific model
        
        Args:
            model_type: Type of model to use
            prompt: Input prompt for generation
            max_length: Maximum length of generated text
            temperature: Temperature for sampling
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repetition
            streaming: Whether to stream the generation
            callback: Callback function for streaming (receives text chunk)
            
        Returns:
            Generated text or streamer object if streaming=True
        """
        model_key = model_type.value
        
        # Load model if not loaded
        if not self.is_model_loaded(model_type):
            success = self.load_model(model_type)
            if not success:
                raise RuntimeError(f"Failed to load {model_key} model")
        
        # Get model and tokenizer
        model = self.loaded_models[model_key]
        tokenizer = self.loaded_tokenizers[model_key]
        
        # Log generation request
        logger.info(f"Generating text with {model_key} model (max_length={max_length}, temp={temperature})")
        
        # Process based on model type
        try:
            # For streaming
            if streaming:
                # Set up streamer
                if model_type == ModelType.OLYMPIC and self.mode == ModelMode.QUANTIZED:
                    # For GGUF models, we don't have built-in streaming
                    # We'll simulate streaming for consistent API
                    result = model(
                        prompt,
                        max_new_tokens=max_length,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        repetition_penalty=repetition_penalty,
                        **kwargs
                    )
                    
                    # Simulate streaming if callback provided
                    if callback:
                        # Split into chunks and call callback
                        chunk_size = 4  # characters
                        for i in range(0, len(result), chunk_size):
                            chunk = result[i:i+chunk_size]
                            callback(chunk)
                            time.sleep(0.01)  # Small delay for realistic streaming
                    
                    return result
                else:
                    # For transformers models, use TextIteratorStreamer
                    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
                    
                    # Prepare generation inputs
                    inputs = tokenizer(prompt, return_tensors="pt")
                    if self.device == "cuda":
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # Set up generation parameters
                    gen_kwargs = {
                        "max_new_tokens": max_length,
                        "temperature": temperature,
                        "top_p": top_p,
                        "top_k": top_k,
                        "repetition_penalty": repetition_penalty,
                        "streamer": streamer,
                        **kwargs
                    }
                    
                    # Start generation in a thread
                    thread = Thread(target=model.generate, kwargs={**inputs, **gen_kwargs})
                    thread.start()
                    
                    # Handle callback if provided
                    if callback:
                        for text in streamer:
                            callback(text)
                    
                    return streamer
            else:
                # For non-streaming
                if model_type == ModelType.OLYMPIC and self.mode == ModelMode.QUANTIZED:
                    # For GGUF models with ctransformers
                    result = model(
                        prompt,
                        max_new_tokens=max_length,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        repetition_penalty=repetition_penalty,
                        **kwargs
                    )
                    return result
                else:
                    # For transformers models
                    inputs = tokenizer(prompt, return_tensors="pt")
                    if self.device == "cuda":
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_length,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        repetition_penalty=repetition_penalty,
                        **kwargs
                    )
                    
                    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # Remove prompt from result if present
                    if result.startswith(prompt):
                        result = result[len(prompt):]
                    
                    return result.strip()
        
        except Exception as e:
            logger.error(f"Error generating text with {model_key} model: {str(e)}")
            raise
    
    def generate_code(
        self,
        prompt: str,
        language: str = "python",
        max_length: int = 2048,
        temperature: float = 0.3,  # Lower temperature for code
        streaming: bool = False,
        callback = None
    ) -> Union[str, TextIteratorStreamer]:
        """
        Generate code using the best model for the task
        
        Args:
            prompt: Input prompt describing the code to generate
            language: Programming language
            max_length: Maximum length of generated code
            temperature: Temperature for sampling
            streaming: Whether to stream the generation
            callback: Callback function for streaming
            
        Returns:
            Generated code or streamer object if streaming=True
        """
        # Use OlympicCoder-7B for code generation tasks
        model_type = ModelType.OLYMPIC
        
        # Create a code-specific prompt
        code_prompt = f"""
Write {language} code for the following:
{prompt}

Return only the code with no explanations, using correct {language} syntax.

```{language}
"""
        
        # Generate with OlympicCoder model
        result = self.generate_text(
            model_type=model_type,
            prompt=code_prompt,
            max_length=max_length,
            temperature=temperature,
            top_p=0.95,
            repetition_penalty=1.05,
            streaming=streaming,
            callback=callback
        )
        
        # For non-streaming, clean up the result
        if not streaming:
            # Extract code block if present
            if "```" in result:
                # Find the closing code block
                code_end = result.find("```", 3)
                if code_end != -1:
                    result = result[:code_end].strip()
            
            # Clean any remaining markers
            result = result.replace(f"```{language}", "").replace("```", "").strip()
        
        return result
    
    def chat(
        self,
        model_type: ModelType,
        messages: List[Dict[str, str]],
        max_length: int = 1024,
        temperature: float = 0.7,
        streaming: bool = False,
        callback = None
    ) -> Union[str, TextIteratorStreamer]:
        """
        Chat with a model using a list of messages
        
        Args:
            model_type: Type of model to use
            messages: List of message dicts with 'role' and 'content'
            max_length: Maximum length of generated response
            temperature: Temperature for sampling
            streaming: Whether to stream the generation
            callback: Callback function for streaming
            
        Returns:
            Generated response or streamer object if streaming=True
        """
        model_key = model_type.value
        
        # Load model if not loaded
        if not self.is_model_loaded(model_type):
            success = self.load_model(model_type)
            if not success:
                raise RuntimeError(f"Failed to load {model_key} model")
        
        # Get tokenizer
        tokenizer = self.loaded_tokenizers[model_key]
        
        # Format depends on model type
        if model_type == ModelType.QWEN:
            # Qwen uses ChatML format
            if hasattr(tokenizer, "apply_chat_template"):
                prompt = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            else:
                # Fallback for older versions
                prompt = ""
                for msg in messages:
                    role = msg["role"]
                    content = msg["content"]
                    if role == "user":
                        prompt += f"Human: {content}\n"
                    elif role == "assistant":
                        prompt += f"Assistant: {content}\n"
                    elif role == "system":
                        prompt += f"System: {content}\n"
                prompt += "Assistant: "
        
        elif model_type == ModelType.OLYMPIC:
            # Format for Olympic Coder
            prompt = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "user":
                    prompt += f"### User:\n{content}\n\n"
                elif role == "assistant":
                    prompt += f"### Assistant:\n{content}\n\n"
                elif role == "system":
                    prompt += f"### System:\n{content}\n\n"
            prompt += "### Assistant:\n"
        
        else:
            raise ValueError(f"Unknown model type: {model_key}")
        
        # Generate the response
        return self.generate_text(
            model_type=model_type,
            prompt=prompt,
            max_length=max_length,
            temperature=temperature,
            streaming=streaming,
            callback=callback
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get the status of the model manager"""
        # Count models
        loaded = list(self.loaded_models.keys())
        
        # Check all models
        all_models_status = {}
        for model_type in ModelType:
            model_key = model_type.value
            exists = self._verify_model_exists(model_type)
            all_models_status[model_key] = {
                "exists": exists,
                "loaded": model_key in loaded,
                "memory_gb": self.memory_usage.get(model_key, 0) if model_key in loaded else 0
            }
        
        return {
            "device": self.device,
            "mode": self.mode.value,
            "loaded_models": loaded,
            "models": all_models_status
        }

# Initialize model manager - the actual loading of models will happen on-demand
model_manager = ModelManager()