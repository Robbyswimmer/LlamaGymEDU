from typing import List, Dict, Tuple
import torch


class TokenizationUtils:
    """Utilities for robust tokenization and chat template handling."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self._setup_tokenizer()
    
    def _setup_tokenizer(self):
        """Initialize tokenizer with pad token and chat template fallback."""
        # Set pad token if not present
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Add chat template fallback for non-chat tokenizers
        if not getattr(self.tokenizer, "chat_template", None):
            self.tokenizer.chat_template = (
                "{% for m in messages %}"
                "{% if m['role']=='system' %}System: {{ m['content'] }}\n"
                "{% elif m['role']=='user' %}User: {{ m['content'] }}\n"
                "{% elif m['role']=='assistant' %}Assistant: {{ m['content'] }}\n{% endif %}"
                "{% endfor %}{% if add_generation_prompt %}Assistant: {% endif %}"
            )
    
    def qr_token_ids(self, msgs_prefix: List[Dict[str, str]], msgs_full: List[Dict[str, str]], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract query and response token IDs using length-based splitting.
        
        Args:
            msgs_prefix: Messages up to (and including) the final user turn
            msgs_full: msgs_prefix + the assistant response
            device: Device to place tensors on
            
        Returns:
            Tuple of (query_ids, response_ids) tensors
        """
        # Generate query text with generation prompt
        q_txt = self.tokenizer.apply_chat_template(
            msgs_prefix, tokenize=False, add_generation_prompt=True
        )
        # Generate full conversation text  
        full_txt = self.tokenizer.apply_chat_template(
            msgs_full, tokenize=False, add_generation_prompt=False
        )
        
        # Tokenize both
        q_ids = self.tokenizer(q_txt, return_tensors="pt").input_ids[0]
        full_ids = self.tokenizer(full_txt, return_tensors="pt").input_ids[0]
        
        # Extract response by length difference
        r_ids = full_ids[len(q_ids):]
        
        # Move to specified device
        q_ids = q_ids.to(device)
        r_ids = r_ids.to(device)
        
        return q_ids, r_ids
    
    def get_pad_token_id(self) -> int:
        """Get the pad token ID, using eos_token_id as fallback."""
        return self.tokenizer.eos_token_id if self.tokenizer.pad_token_id is None else self.tokenizer.pad_token_id
    
    def encode_with_device(self, text: str, device: torch.device, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Encode text and move tensors to specified device.
        
        Args:
            text: Text to encode
            device: Target device
            **kwargs: Additional tokenizer arguments
            
        Returns:
            Dictionary of tensors on the specified device
        """
        encoded = self.tokenizer(text, return_tensors="pt", **kwargs)
        return {k: v.to(device) for k, v in encoded.items()}