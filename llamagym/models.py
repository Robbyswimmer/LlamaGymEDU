"""
Model registry and configuration management for LlamaGym.

Provides pre-configured settings for popular models and families,
with intelligent defaults and fallback mechanisms.
"""

from typing import Dict, Any, Optional
import re


class ModelConfig:
    """Configuration for a specific model or model family."""
    
    def __init__(
        self,
        name: str,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None, 
        generation_config: Optional[Dict[str, Any]] = None,
        chat_template: Optional[str] = None,
        recommended_peft_config: Optional[Dict[str, Any]] = None,
        notes: Optional[str] = None
    ):
        self.name = name
        self.tokenizer_kwargs = tokenizer_kwargs or {}
        self.model_kwargs = model_kwargs or {}
        self.generation_config = generation_config or {}
        self.chat_template = chat_template
        self.recommended_peft_config = recommended_peft_config or {}
        self.notes = notes or ""
    
    def merge_with(self, overrides: 'ModelConfig') -> 'ModelConfig':
        """Merge this config with override values."""
        return ModelConfig(
            name=overrides.name or self.name,
            tokenizer_kwargs={**self.tokenizer_kwargs, **overrides.tokenizer_kwargs},
            model_kwargs={**self.model_kwargs, **overrides.model_kwargs},
            generation_config={**self.generation_config, **overrides.generation_config},
            chat_template=overrides.chat_template or self.chat_template,
            recommended_peft_config={**self.recommended_peft_config, **overrides.recommended_peft_config},
            notes=overrides.notes or self.notes
        )


class ModelRegistry:
    """Registry of pre-configured model settings with intelligent fallbacks."""
    
    def __init__(self):
        self._exact_configs = {}
        self._family_configs = {}
        self._setup_default_configs()
    
    def _setup_default_configs(self):
        """Setup pre-configured model profiles."""
        
        # Llama 3.1 Family
        llama31_base = ModelConfig(
            name="llama-3.1-base",
            tokenizer_kwargs={
                "add_special_tokens": True,
                "padding_side": "left",
                "trust_remote_code": False
            },
            model_kwargs={
                "torch_dtype": "auto",
                "device_map": "auto",
                "trust_remote_code": False
            },
            generation_config={
                "max_new_tokens": 16,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": False,  # Use greedy for stability
                "pad_token_id": 128001,  # Llama 3.1 specific
                "eos_token_id": 128001   # Proper EOS handling
            },
            recommended_peft_config={
                "r": 8,  # Reduced from 16 to 8 for efficiency
                "lora_alpha": 16,  # Scaled proportionally 
                "lora_dropout": 0.05,
                "bias": "none",
                "task_type": "CAUSAL_LM",
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]  # Only attention layers, removed MLP layers
            },
            notes="Llama 3.1 models with optimized settings for RL training"
        )
        
        # Specific model variants
        self._exact_configs.update({
            "meta-llama/Meta-Llama-3.1-8B": llama31_base,
            "meta-llama/Meta-Llama-3.1-8B-Instruct": llama31_base,
            "meta-llama/Meta-Llama-3.1-70B": llama31_base,
            "meta-llama/Meta-Llama-3.1-70B-Instruct": llama31_base,
            "meta-llama/Meta-Llama-3.1-405B": llama31_base,
            "meta-llama/Meta-Llama-3.1-405B-Instruct": llama31_base,
        })
        
        # Llama 3.2 Family (similar to 3.1)
        llama32_base = llama31_base.merge_with(ModelConfig(
            name="llama-3.2-base",
            generation_config={
                "max_new_tokens": 16,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": False,  # Use greedy for stability
                "pad_token_id": 128001,  # Llama 3.2 specific
                "eos_token_id": 128001   # Proper EOS handling
            },
            notes="Llama 3.2 models with optimized settings for RL training"
        ))
        
        # Simple config for Unsloth models to avoid generation conflicts
        unsloth_llama32_config = ModelConfig(
            name="unsloth-llama-3.2-base",
            tokenizer_kwargs={
                "add_special_tokens": True,
                "padding_side": "left",
                "trust_remote_code": False
            },
            model_kwargs={
                "torch_dtype": "auto",
                "device_map": "auto",
                "trust_remote_code": False
            },
            generation_config={
                "max_new_tokens": 16,
                "pad_token_id": 2  # Standard eos token
            },
            recommended_peft_config={
                "r": 8,  # Reduced from 16 to 8 for efficiency
                "lora_alpha": 16,  # Scaled proportionally
                "lora_dropout": 0.05,
                "bias": "none",
                "task_type": "CAUSAL_LM",
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]  # Only attention layers, removed MLP layers
            },
            notes="Unsloth Llama 3.2 with simplified generation config"
        )
        
        self._exact_configs.update({
            "meta-llama/Llama-3.2-1B": llama32_base,
            "meta-llama/Llama-3.2-1B-Instruct": llama32_base,
            "meta-llama/Llama-3.2-3B": llama32_base,
            "meta-llama/Llama-3.2-3B-Instruct": llama32_base,
            "unsloth/Llama-3.2-1B": unsloth_llama32_config,
            "unsloth/Llama-3.2-1B-Instruct": unsloth_llama32_config,
        })
        
        # GPT-2 Family (backward compatibility)
        gpt2_base = ModelConfig(
            name="gpt2-base",
            tokenizer_kwargs={
                "add_special_tokens": True,
                "padding_side": "left"
            },
            model_kwargs={
                "torch_dtype": "auto"
            },
            generation_config={
                "max_new_tokens": 64,
                "temperature": 0.9,
                "top_p": 0.6,
                "do_sample": True
            },
            recommended_peft_config={
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "bias": "none", 
                "task_type": "CAUSAL_LM"
            },
            notes="GPT-2 family with conservative settings"
        )
        
        self._exact_configs.update({
            "gpt2": gpt2_base,
            "gpt2-medium": gpt2_base,
            "gpt2-large": gpt2_base,
            "gpt2-xl": gpt2_base,
            "distilgpt2": gpt2_base,
            "microsoft/DialoGPT-medium": gpt2_base,
            "microsoft/DialoGPT-large": gpt2_base,
        })

        # Laptop-friendly compact models
        tinyllama_config = ModelConfig(
            name="tinyllama-1.1b-chat",
            tokenizer_kwargs={
                "add_special_tokens": True,
                "padding_side": "left",
                "trust_remote_code": False,
            },
            model_kwargs={
                "torch_dtype": "auto",
                "device_map": "auto",
                "trust_remote_code": False,
            },
            generation_config={
                "max_new_tokens": 64,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
            },
            recommended_peft_config={
                "r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "bias": "none",
                "task_type": "CAUSAL_LM",
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            },
            notes="TinyLlama 1.1B chat model—fits on CPU or Apple M-series laptops",
        )

        phi_small_config = ModelConfig(
            name="phi-small",
            tokenizer_kwargs={
                "add_special_tokens": True,
                "padding_side": "left",
                "trust_remote_code": False,
            },
            model_kwargs={
                "torch_dtype": "auto",
                "device_map": "auto",
                "trust_remote_code": False,
            },
            generation_config={
                "max_new_tokens": 80,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
            },
            recommended_peft_config={
                "r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "bias": "none",
                "task_type": "CAUSAL_LM",
            },
            notes="Microsoft Phi models tuned for laptop-scale experiments",
        )

        smollm_config = ModelConfig(
            name="smollm-1.7b-instruct",
            tokenizer_kwargs={
                "add_special_tokens": True,
                "padding_side": "left",
                "trust_remote_code": False,
            },
            model_kwargs={
                "torch_dtype": "auto",
                "device_map": "auto",
                "trust_remote_code": False,
            },
            generation_config={
                "max_new_tokens": 96,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
            },
            recommended_peft_config={
                "r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "bias": "none",
                "task_type": "CAUSAL_LM",
            },
            notes="SmolLM 1.7B instruct—portable baseline for CPU finetuning",
        )

        qwen_small_config = ModelConfig(
            name="qwen2-1.5b-instruct",
            tokenizer_kwargs={
                "add_special_tokens": True,
                "padding_side": "left",
                "trust_remote_code": False,
            },
            model_kwargs={
                "torch_dtype": "auto",
                "device_map": "auto",
                "trust_remote_code": False,
            },
            generation_config={
                "max_new_tokens": 96,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
            },
            recommended_peft_config={
                "r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "bias": "none",
                "task_type": "CAUSAL_LM",
            },
            notes="Qwen2 1.5B instruct—good balance between capability and memory",
        )

        self._exact_configs.update({
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0": tinyllama_config,
            "TinyLlama/TinyLlama-1.1B": tinyllama_config,
            "microsoft/phi-2": phi_small_config,
            "microsoft/phi-1_5": phi_small_config,
            "microsoft/phi-3-mini-4k-instruct": phi_small_config,
            "microsoft/phi-3-mini-128k-instruct": phi_small_config,
            "HuggingFaceTB/SmolLM-1.7B-Instruct": smollm_config,
            "Qwen/Qwen2-1.5B-Instruct": qwen_small_config,
        })

        # Mixtral Family
        mixtral_base = ModelConfig(
            name="mixtral-base",
            tokenizer_kwargs={
                "add_special_tokens": True,
                "padding_side": "left"
            },
            model_kwargs={
                "torch_dtype": "auto",
                "device_map": "auto",
                "load_in_4bit": True  # Mixtral is large
            },
            generation_config={
                "max_new_tokens": 256,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True
            },
            recommended_peft_config={
                "r": 64,  # Higher rank for Mixtral
                "lora_alpha": 128,
                "lora_dropout": 0.1,
                "bias": "none",
                "task_type": "CAUSAL_LM"
            },
            notes="Mixtral models with 4-bit quantization by default"
        )
        
        self._exact_configs.update({
            "mistralai/Mixtral-8x7B-v0.1": mixtral_base,
            "mistralai/Mixtral-8x7B-Instruct-v0.1": mixtral_base,
        })
        
        # Family-based patterns for fuzzy matching
        self._family_configs = {
            "tinyllama": tinyllama_config,
            "phi": phi_small_config,
            "smollm": smollm_config,
            "qwen2": qwen_small_config,
            "llama-3.1": llama31_base,
            "llama-3.2": llama32_base,
            "llama": llama31_base,  # Default to latest
            "gpt2": gpt2_base,
            "mixtral": mixtral_base,
        }
    
    def get_config(self, model_name: str) -> ModelConfig:
        """
        Get configuration for a model with intelligent fallbacks.
        
        Args:
            model_name: HuggingFace model name or path
            
        Returns:
            ModelConfig with appropriate settings
        """
        # 1. Exact match
        if model_name in self._exact_configs:
            return self._exact_configs[model_name]
        
        # 2. Family-based fuzzy matching
        model_lower = model_name.lower()
        for family_key, config in self._family_configs.items():
            if family_key in model_lower:
                return config
        
        # 3. Intelligent defaults based on patterns
        if "instruct" in model_lower or "chat" in model_lower:
            # Instruction-tuned models
            return ModelConfig(
                name=f"instruct-default-{model_name}",
                tokenizer_kwargs={"add_special_tokens": True, "padding_side": "left"},
                model_kwargs={"torch_dtype": "auto", "device_map": "auto"},
                generation_config={"max_new_tokens": 256, "temperature": 0.7, "top_p": 0.9},
                recommended_peft_config={"r": 16, "lora_alpha": 32, "task_type": "CAUSAL_LM"},
                notes="Auto-detected instruction model with conservative defaults"
            )
        
        # 4. Safe universal defaults
        return ModelConfig(
            name=f"default-{model_name}",
            tokenizer_kwargs={"add_special_tokens": True, "padding_side": "left"},
            model_kwargs={"torch_dtype": "auto"},
            generation_config={"max_new_tokens": 128, "temperature": 0.9},
            recommended_peft_config={"r": 16, "lora_alpha": 32, "task_type": "CAUSAL_LM"},
            notes="Universal fallback configuration"
        )
    
    def register_model(self, model_name: str, config: ModelConfig):
        """Register a custom model configuration."""
        self._exact_configs[model_name] = config
    
    def list_supported_models(self) -> Dict[str, str]:
        """List all supported models with their descriptions."""
        return {name: config.notes for name, config in self._exact_configs.items()}
    
    def list_families(self) -> Dict[str, str]:
        """List supported model families."""
        return {name: config.notes for name, config in self._family_configs.items()}


# Global registry instance
model_registry = ModelRegistry()