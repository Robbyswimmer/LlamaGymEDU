import torch
from typing import List, Dict, Optional


class SFTTrainer:
    """
    Supervised Fine-Tuning trainer for offline→online bridge.
    
    Handles SFT warm-start training on top-performing episodes from replay buffer
    with proper device handling and value head freezing.
    """
    
    def __init__(self, model, tokenizer, ppo_trainer, replay_buffer, 
                 sft_steps: int = 1, topk_p: float = 0.2):
        """
        Initialize SFT trainer.
        
        Args:
            model: The language model with value head
            tokenizer: Tokenizer instance
            ppo_trainer: PPO trainer instance (for optimizer access)
            replay_buffer: Replay buffer containing episodes
            sft_steps: Number of SFT gradient steps per warmstart
            topk_p: Top percentile of episodes to use for SFT
        """
        self.model = model
        self.tokenizer = tokenizer
        self.ppo_trainer = ppo_trainer
        self.replay_buffer = replay_buffer
        self.sft_steps = sft_steps
        self.topk_p = topk_p
    
    def run_sft_warmstart(self, get_system_prompt_fn) -> Optional[Dict]:
        """
        Run SFT on top-K episodes from replay buffer.
        
        Args:
            get_system_prompt_fn: Function that returns system prompt
            
        Returns:
            Dictionary with SFT training statistics, or None if no data
        """
        top_episodes = self.replay_buffer.sample_topk(self.topk_p)
        if not top_episodes:
            return None
        
        sft_data = self.replay_buffer.get_sft_data(top_episodes, max_pairs=128, dedup=True)
        if not sft_data:
            return None
        
        # Get model's actual device
        model_device = next(self.model.parameters()).device
        
        # Set model to training mode
        self.model.train()
        
        # Use existing optimizer to maintain consistency
        optimizer = self.ppo_trainer.optimizer
        
        # Freeze value head during SFT to prevent interference
        vhead_state = self._freeze_value_head()
        
        total_loss = 0.0
        num_steps = 0
        
        try:
            for _ in range(self.sft_steps):
                for obs_text, response_text in sft_data:
                    loss = self._sft_step(obs_text, response_text, get_system_prompt_fn(), 
                                        model_device, optimizer)
                    if loss is not None:
                        total_loss += loss
                        num_steps += 1
        finally:
            # Always restore value head gradients
            self._restore_value_head(vhead_state)
        
        avg_loss = total_loss / max(1, num_steps)
        return {
            "sft_loss": avg_loss,
            "sft_steps": num_steps,
            "sft_episodes_used": len(top_episodes),
            "sft_pairs_used": len(sft_data)
        }
    
    def _sft_step(self, obs_text: str, response_text: str, system_prompt: str, 
                  device: torch.device, optimizer) -> Optional[float]:
        """
        Execute a single SFT training step.
        
        Args:
            obs_text: Observation text
            response_text: Response text  
            system_prompt: System prompt
            device: Model device
            optimizer: Optimizer instance
            
        Returns:
            Loss value or None if step failed
        """
        try:
            # Build messages for prefix (without assistant response)
            pre_msgs = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": obs_text},
            ]
            
            # Build full messages (with assistant response)
            full_msgs = pre_msgs + [{"role": "assistant", "content": response_text}]
            
            # Generate text representations
            pre_txt = self.tokenizer.apply_chat_template(
                pre_msgs, tokenize=False, add_generation_prompt=False
            )
            full_txt = self.tokenizer.apply_chat_template(
                full_msgs, tokenize=False, add_generation_prompt=False
            )
            
            # Tokenize and move to device
            enc_full = self.tokenizer(full_txt, return_tensors="pt", truncation=True)
            enc_pre = self.tokenizer(pre_txt, return_tensors="pt", truncation=True)
            
            enc_full = {k: v.to(device) for k, v in enc_full.items()}
            enc_pre = {k: v.to(device) for k, v in enc_pre.items()}
            
            # Create labels - mask everything before assistant response
            labels = enc_full["input_ids"].clone()
            labels[:, :enc_pre["input_ids"].size(1)] = -100
            
            # Forward pass
            optimizer.zero_grad(set_to_none=True)
            outputs = self.model(**enc_full, labels=labels)
            
            # Extract loss (handle both tuple and object returns)
            if isinstance(outputs, tuple):
                loss = outputs[0]
            else:
                loss = outputs.loss
            
            # Ensure loss is scalar
            if loss.numel() > 1:
                loss = loss.mean()
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            
            return loss.item()
            
        except Exception as e:
            # Log error but continue training
            print(f"SFT step failed: {e}")
            return None
    
    def _freeze_value_head(self) -> List[bool]:
        """
        Freeze value head parameters during SFT.
        
        Returns:
            List of original requires_grad states for restoration
        """
        vhead = getattr(self.model, "v_head", None)
        frozen_states = []
        
        if vhead is not None:
            for param in vhead.parameters():
                frozen_states.append(param.requires_grad)
                param.requires_grad = False
        
        return frozen_states
    
    def _restore_value_head(self, frozen_states: List[bool]):
        """
        Restore value head gradient states after SFT.
        
        Args:
            frozen_states: Original requires_grad states to restore
        """
        vhead = getattr(self.model, "v_head", None)
        
        if vhead is not None and frozen_states:
            for param, original_state in zip(vhead.parameters(), frozen_states):
                param.requires_grad = original_state