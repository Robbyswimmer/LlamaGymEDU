import torch
from typing import List, Dict, Optional
from tqdm import tqdm


class SFTTrainer:
    """
    Supervised Fine-Tuning trainer for offline→online bridge.
    
    Handles SFT warm-start training on top-performing episodes from replay buffer
    with proper device handling and value head freezing.
    """
    
    def __init__(self, model, tokenizer, ppo_trainer, replay_buffer, 
                 sft_steps: int = 1, topk_p: float = 0.2, sft_lr: float = 1e-4):
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
        self.sft_lr = sft_lr

        # Configure tokenizer for SFT: keep assistant tokens, not prompt
        self.tokenizer.truncation_side = "left"
        self.tokenizer.padding_side = "right"
    
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



        # Disable KV cache during training for stability on MPS / to reduce memory
        try:
            if hasattr(self.model, 'config'):
                self.model.config.use_cache = False
        except Exception:
            pass
        
        # Build a dedicated SFT optimizer over only trainable (LoRA) params
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if len(trainable_params) == 0:
            print("SFT Warning: No trainable parameters found for SFT optimizer")
            return None
        optimizer = torch.optim.AdamW(trainable_params, lr=self.sft_lr, betas=(0.9, 0.999), weight_decay=0.01)

        # Debug: report trainable parameters and optimizer param groups
        total_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        opt_params = sum(p.numel() for g in optimizer.param_groups for p in g.get('params', [])) if hasattr(optimizer, 'param_groups') else 0
        lrs = []
        if hasattr(optimizer, 'param_groups'):
            for g in optimizer.param_groups:
                if 'lr' in g:
                    lrs.append(g['lr'])
        print(f"SFT Debug: trainable params={total_trainable:,}, optimizer params={opt_params:,}, groups={len(getattr(optimizer, 'param_groups', []))}, lrs={lrs}")
        
        # Freeze value head during SFT to prevent interference
        vhead_state = self._freeze_value_head()


        total_loss = 0.0
        num_steps = 0
        
        # Calculate total steps for progress bar
        total_sft_steps = self.sft_steps * len(sft_data)
        
        print(f"\n🔥 Starting SFT warm-start: {len(sft_data)} examples × {self.sft_steps} steps = {total_sft_steps} total steps")
        print(f"First example: obs='{sft_data[0][0]}' response='{sft_data[0][1]}'")
        
        try:
            with tqdm(total=total_sft_steps, desc="SFT Training", unit="steps") as pbar:
                for step in range(self.sft_steps):
                    for obs_text, response_text in sft_data:
                        loss = self._sft_step(obs_text, response_text, get_system_prompt_fn(), 
                                            model_device, optimizer)
                        if loss is not None:
                            total_loss += loss
                            num_steps += 1
                            avg_loss_display = total_loss / num_steps if num_steps > 0 else 0.0
                            pbar.set_postfix(loss=f"{loss:.4f}", avg_loss=f"{avg_loss_display:.4f}")
                        else:
                            pbar.set_postfix(loss="skip", avg_loss=f"{total_loss/max(1,num_steps):.4f}")
                        pbar.update(1)
        finally:
            # Always restore value head gradients
            self._restore_value_head(vhead_state)
        
        avg_loss = total_loss / max(1, num_steps)
        print(f"✅ SFT warm-start completed! Avg loss: {avg_loss:.4f}")
        return {
            "sft_loss": avg_loss,
            "sft_steps": num_steps,
            "sft_episodes_used": len(top_episodes),
            "sft_pairs_used": len(sft_data)
        }
    
    def _sft_step(self, obs_text: str, response_text: str, system_prompt: str, 
                  device: torch.device, optimizer) -> Optional[float]:
        try:
            # Keep assistant tokens by truncating from the left
            self.tokenizer.truncation_side = "left"
            self.tokenizer.padding_side = "right"
            max_len = min(getattr(self.tokenizer, "model_max_length", 1024), 1024)

            # Build texts, but first compute a response with EOS and its token length
            pre_msgs = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": obs_text},
            ]
            # ensure EOS on target
            resp = response_text
            eos = self.tokenizer.eos_token or ""
            if eos and not resp.endswith(eos):
                resp = resp + eos

            # Estimate response token budget (cap to avoid single giant answers dominating)
            max_resp_tokens_cap = max_len // 2  # avoid responses longer than half the context
            resp_ids = self.tokenizer(resp, return_tensors="pt")["input_ids"].squeeze(0)
            target_resp_tokens = min(resp_ids.size(0), max_resp_tokens_cap)
            # Also require a minimal response budget to avoid pathological prefix-only inputs
            # Use a smaller minimum to reduce prefix pressure on tight contexts
            min_resp_tokens = min(32, max_len // 4)
            resp_budget = max(target_resp_tokens, min_resp_tokens)

            # Iteratively trim obs_text until there is room: len(prefix) <= (max_len - resp_budget)
            # We approximate prefix by tokenizing the messages without the assistant.
            # Use a bounded number of iterations to avoid rare infinite loops.
            for _ in range(10):
                pre_txt_tmp = self.tokenizer.apply_chat_template(pre_msgs, tokenize=False, add_generation_prompt=False)
                enc_pre_tmp = self.tokenizer(pre_txt_tmp, return_tensors="pt", truncation=False)
                pre_len_tmp = enc_pre_tmp["input_ids"].size(1)
                allowed_prefix = max_len - resp_budget
                if pre_len_tmp <= allowed_prefix:
                    break
                # Need to shrink obs_text from the left (keep latest part)
                # Compute how many obs tokens to drop so that pre_len fits, with a small margin (8 tokens)
                obs_ids = self.tokenizer(obs_text, return_tensors="pt")["input_ids"].squeeze(0)
                # Over-limit amount we need to shave off the prefix length
                over_by = pre_len_tmp - allowed_prefix
                # Heuristic: assume ~1:1 contribution from obs tokens, drop that many plus margin
                trim_n = min(obs_ids.size(0) - 1, max(over_by + 8, 32))
                if trim_n <= 0 or obs_ids.size(0) <= 1:
                    # Cannot trim further
                    break
                if trim_n >= obs_ids.size(0):
                    # Would remove entire obs; keep last token to preserve structure
                    obs_ids = obs_ids[-1:]
                else:
                    obs_ids = obs_ids[-(obs_ids.size(0) - trim_n):]
                obs_text = self.tokenizer.decode(obs_ids, skip_special_tokens=True)
                pre_msgs = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": obs_text},
                ]

            # Now finalize texts
            full_msgs = pre_msgs + [{"role": "assistant", "content": resp}]

            pre_txt  = self.tokenizer.apply_chat_template(pre_msgs,  tokenize=False, add_generation_prompt=False)
            full_txt = self.tokenizer.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)

            enc_full = self.tokenizer(full_txt, return_tensors="pt", truncation=True,  max_length=max_len)
            enc_pre  = self.tokenizer(pre_txt,  return_tensors="pt", truncation=True,  max_length=max_len)

            enc_full = {k: v.to(device) for k, v in enc_full.items()}
            enc_pre  = {k: v.to(device) for k, v in enc_pre.items()}

            labels = enc_full["input_ids"].clone()
            label_start = enc_pre["input_ids"].size(1)
            # guard against pathological truncation
            if label_start >= labels.size(1):
                # nothing to supervise; skip cleanly
                print("SFT skip: label_start >= seq_len (prefix consumed full context)")
                print(f"  Debug: max_len={max_len}, resp_budget={resp_budget}, label_start={label_start}, seq_len={labels.size(1)}")
                return None
            labels[:, :label_start] = -100

            optimizer.zero_grad(set_to_none=True)
            # Forward pass without built-in loss to control dtype for stability
            outputs = self.model(**enc_full)

            # Compute supervised loss in float32 for stability on MPS
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            import torch.nn.functional as F
            logits_f32 = logits.float()

            # Pre-sanitization diagnostics
            has_nan = torch.isnan(logits_f32).any().item()
            max_abs = torch.max(torch.abs(logits_f32)).item()

            # Causal LM convention: predict token t+1 at position t
            logits_shifted = logits_f32[:, :-1, :]
            labels_shifted = labels[:, 1:]

            # Sanitize logits to avoid NaN/Inf in softmax
            logits_shifted = torch.nan_to_num(logits_shifted, nan=0.0, posinf=80.0, neginf=-80.0)
            logits_shifted = torch.clamp(logits_shifted, -80.0, 80.0)

            labels_view = labels_shifted.reshape(-1)
            logits_view = logits_shifted.reshape(-1, logits_shifted.size(-1))

            # Ensure there are supervised tokens
            valid_mask = labels_view != -100
            if valid_mask.sum().item() == 0:
                print("SFT skip: no supervised tokens after masking")
                return None

            loss = F.cross_entropy(logits_view, labels_view, ignore_index=-100)

            # extra safety against NaNs/Infs
            if not torch.isfinite(loss):
                # Print some diagnostic stats to help debugging
                with torch.no_grad():
                    max_logit = torch.max(torch.abs(logits)).item()
                    print(f"SFT skip: non-finite loss (|logits| max={max_logit:.2f})")
                return None

            loss.backward()

            # Gradient diagnostics: check if any trainable grad is non-zero
            with torch.no_grad():
                grad_norm = 0.0
                nonzero_grads = 0
                total_params = 0
                for n, p in self.model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        g = p.grad.detach()
                        grad_norm += torch.norm(g).item()
                        if torch.any(g != 0):
                            nonzero_grads += 1
                        total_params += 1
                if total_params > 0 and nonzero_grads == 0:
                    print("SFT Debug: zero grads on all trainable params")
                elif total_params > 0 and not torch.isfinite(torch.tensor(grad_norm)):
                    print("SFT Debug: non-finite grad norm")
                # occasional log if nan logits were detected
                if has_nan:
                    print(f"SFT Debug: NaNs in logits pre-sanitize, max|logit|={max_abs:.2f}")
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            return float(loss.detach().cpu())
        except Exception as e:
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