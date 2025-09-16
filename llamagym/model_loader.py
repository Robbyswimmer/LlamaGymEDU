"""
Smart model loading utilities for LlamaGym.

Provides intelligent model and tokenizer loading with automatic configuration,
device detection, error handling, and optimization.
"""

import os
import torch
from typing import Tuple, Optional, Dict, Any
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead
from peft import LoraConfig

from .models import model_registry, ModelConfig


def detect_device() -> str:
    """Automatically detect the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps:0"
    else:
        return "cpu"



def should_use_quantization(model_name: str, device: str) -> Dict[str, Any]:
    """Suggest quantization settings based on model and hardware."""
    model_lower = model_name.lower()
    quantization_kwargs = {}
    
    # Large models benefit from quantization
    if any(size in model_lower for size in ["70b", "405b", "mixtral"]):
        if device != "cpu":  # Quantization not supported on CPU in many cases
            quantization_kwargs["load_in_4bit"] = True
        else:
            print(f"Warning: Large model {model_name} on CPU may require significant memory")
    
    elif any(size in model_lower for size in ["7b", "8b", "13b"]):
        # Medium models - optional 8-bit quantization
        if device == "cuda":
            quantization_kwargs["load_in_8bit"] = True
    
    return quantization_kwargs


def load_model_and_tokenizer(
    model_name: str,
    device: str = "auto",
    load_in_8bit: Optional[bool] = None,
    load_in_4bit: Optional[bool] = None,
    peft_config: Optional[LoraConfig] = None,
    model_config_override: Optional[ModelConfig] = None,
    trust_remote_code: bool = False,
    token: Optional[str] = None,
    **kwargs
) -> Tuple[AutoModelForCausalLMWithValueHead, AutoTokenizer, str]:
    """
    Smart model and tokenizer loading with automatic configuration.
    
    Args:
        model_name: HuggingFace model name or local path
        device: Device to load model on ("auto", "cpu", "cuda", "mps")
        load_in_8bit: Force 8-bit quantization (overrides auto-detection)
        load_in_4bit: Force 4-bit quantization (overrides auto-detection)
        peft_config: Custom PEFT configuration (uses model default if None)
        model_config_override: Override model configuration
        trust_remote_code: Whether to trust remote code in model
        token: HuggingFace API token
        **kwargs: Additional arguments passed to model loading
        
    Returns:
        Tuple of (model, tokenizer, actual_device) ready for training
        
    Raises:
        Exception: If model loading fails with helpful error message
    """
    print(f"Loading model: {model_name}")
    
    # Auto-detect device if requested
    if device == "auto":
        device = detect_device()
        print(f"Auto-detected device: {device}")
    
    # Get or override model configuration
    if model_config_override:
        config = model_config_override
    else:
        config = model_registry.get_config(model_name)
        print(f"Using model config: {config.name}")
    
    # Get HuggingFace token from environment if not provided
    if token is None:
        token = os.environ.get("HF_TOKEN")
    
    try:
        # Load tokenizer first (faster, helps catch auth issues early)
        print("Loading tokenizer...")
        tokenizer_kwargs = {
            **config.tokenizer_kwargs,
            "token": token,
            "trust_remote_code": trust_remote_code
        }
        
        # Try fast tokenizer first, fallback to slow if it fails
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                **tokenizer_kwargs
            )
        except Exception as fast_tokenizer_error:
            print(f"Fast tokenizer failed, trying slow tokenizer: {fast_tokenizer_error}")
            try:
                tokenizer_kwargs_slow = {**tokenizer_kwargs, "use_fast": False}
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    **tokenizer_kwargs_slow
                )
            except Exception as slow_tokenizer_error:
                print(f"Slow tokenizer also failed: {slow_tokenizer_error}")
                # For Unsloth models, try using a compatible Llama tokenizer
                if "unsloth" in model_name.lower() and "llama-3.2" in model_name.lower():
                    print("Trying compatible Llama tokenizer for Unsloth model...")
                    # Try a non-gated Llama tokenizer
                    compatible_tokenizers = [
                        "meta-llama/Llama-2-7b-hf",  # Non-gated Llama tokenizer
                        "NousResearch/Llama-2-7b-hf"  # Community mirror
                    ]
                    tokenizer_loaded = False
                    for compatible_model in compatible_tokenizers:
                        try:
                            original_tokenizer_kwargs = {**tokenizer_kwargs}
                            tokenizer = AutoTokenizer.from_pretrained(
                                compatible_model,
                                **original_tokenizer_kwargs
                            )
                            print(f"Successfully loaded tokenizer from {compatible_model}")
                            tokenizer_loaded = True
                            break
                        except:
                            continue
                    if not tokenizer_loaded:
                        raise slow_tokenizer_error
                else:
                    raise slow_tokenizer_error
        
        # Handle pad token setup
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
                print("Set pad_token_id to eos_token_id")
            else:
                print("Warning: Could not set pad_token_id")
        
        # Setup PEFT configuration
        if peft_config is None and config.recommended_peft_config:
            peft_config = LoraConfig(**config.recommended_peft_config)
            print(f"Using recommended PEFT config: r={peft_config.r}")
        
        # Prepare model loading arguments
        model_kwargs = {
            **config.model_kwargs,
            "token": token,
            "trust_remote_code": trust_remote_code,
            **kwargs  # User overrides take precedence
        }
        
        # Handle MPS BFloat16 incompatibility
        if device.startswith("mps") or (device == "auto" and detect_device().startswith("mps")):
            if model_kwargs.get("torch_dtype") == "auto":
                model_kwargs["torch_dtype"] = torch.float16
                print("Using float16 for MPS compatibility (BFloat16 not supported)")
        
        # Handle quantization
        if load_in_4bit is not None:
            model_kwargs["load_in_4bit"] = load_in_4bit
        elif load_in_8bit is not None:
            model_kwargs["load_in_8bit"] = load_in_8bit
        else:
            # Auto-detect quantization needs
            quant_kwargs = should_use_quantization(model_name, device)
            model_kwargs.update(quant_kwargs)
            if quant_kwargs:
                print(f"Auto-enabled quantization: {quant_kwargs}")
        
        # Handle device mapping
        if device == "cpu":
            model_kwargs["device_map"] = "cpu"
        elif device.startswith("mps"):
            # For MPS, don't use device_map="auto" as it causes offloading issues with ValueHead
            model_kwargs.pop("device_map", None)
        elif "device_map" not in model_kwargs:
            model_kwargs["device_map"] = "auto"
        
        # Load model
        print("Loading model with value head...")
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            pretrained_model_name_or_path=model_name,
            peft_config=peft_config,
            **model_kwargs
        )
        
        # Ensure model is on correct device if not using device_map="auto"
        if model_kwargs.get("device_map") == "cpu" or device == "cpu" or device.startswith("mps"):
            model = model.to(device)
            print(f"Moved model to {device}")
        
        # Resize token embeddings if tokenizer was modified
        if hasattr(tokenizer, '_added_tokens') and tokenizer._added_tokens:
            print("Resizing token embeddings...")
            model.pretrained_model.resize_token_embeddings(len(tokenizer))
        
        # Debug: Check if PEFT is properly applied and base layers are frozen
        print(f"✅ Successfully loaded {model_name}")
        print(f"   Model device: {next(model.parameters()).device}")
        print(f"   Tokenizer vocab size: {len(tokenizer)}")
        
        # Check PEFT status
        if hasattr(model, 'pretrained_model') and hasattr(model.pretrained_model, 'peft_config'):
            print(f"   ✅ PEFT enabled with config: {model.pretrained_model.peft_config}")
            
            # Count trainable vs total parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"   Total parameters: {total_params:,}")
            print(f"   Trainable parameters: {trainable_params:,}")
            print(f"   Trainable %: {100 * trainable_params / total_params:.2f}%")
        else:
            print(f"   ⚠️  PEFT not detected - this might train the full model!")
            
        if config.notes:
            print(f"   Notes: {config.notes}")
        
        return model, tokenizer, device
        
    except Exception as e:
        error_msg = f"Failed to load model {model_name}: {str(e)}"
        
        # Provide helpful error messages for common issues
        if "rope_scaling" in str(e).lower():
            error_msg += (
                "\n💡 Tip: This looks like a Llama 3.1 compatibility issue. "
                "Make sure you have transformers>=4.43.0 installed."
            )
        elif "token" in str(e).lower() and "authentication" in str(e).lower():
            error_msg += (
                "\n💡 Tip: This model may require authentication. "
                "Set your HF_TOKEN environment variable or pass token= parameter."
            )
        elif "memory" in str(e).lower() or "cuda" in str(e).lower():
            error_msg += (
                "\n💡 Tip: Try enabling quantization: load_in_8bit=True or load_in_4bit=True"
            )
        
        raise Exception(error_msg) from e


def create_generation_config(model_config: ModelConfig, **overrides) -> Dict[str, Any]:
    """Create a generation config from model config with overrides."""
    config = {**model_config.generation_config, **overrides}
    return config