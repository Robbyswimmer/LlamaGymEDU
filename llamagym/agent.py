from abc import ABC, abstractmethod
from typing import List, Dict
import json

import gymnasium as gym
import torch
from trl import (
    PPOTrainer,
    PPOConfig,
    create_reference_model,
)
from .replay import ReplayBuffer
from .sft_trainer import SFTTrainer
from .tokenization_utils import TokenizationUtils


class Agent(ABC):
    def __init__(
        self, model, tokenizer, device, generate_config_dict=None, ppo_config_dict=None,
        sft_warm_start=False, sft_steps=1, topk_p=0.2, use_target_kl=False, target_kl=0.02,
        reward_scale=0.1
    ):
        if generate_config_dict is None:
            generate_config_dict = {
                "max_new_tokens": 32,
                "do_sample": True,
                "top_p": 0.6,
                "top_k": 0,
                "temperature": 0.9,
            }
        if ppo_config_dict is None:
            ppo_config_dict = {
                "batch_size": 16, 
                "mini_batch_size": 16,
                "learning_rate": 1e-5,
                "clip_range": 0.2,
                "vf_coef": 0.1,
                "cliprange_value": 0.2,
                "ppo_epochs": 1
            }
        # Note: target_kl will be handled by our custom logic since newer TRL versions
        # may not support it directly in PPOConfig

        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.generate_config_dict = generate_config_dict
        
        # Initialize tokenization utilities
        self.tokenization_utils = TokenizationUtils(tokenizer)

        # put BOTH policy and ref on SAME device
        self.model.to(self.device)
        self.model_ref = create_reference_model(self.model)
        self.model_ref.to(self.device)

        # TRL 0.7.x: do NOT pass device/accelerator kwargs
        self.ppo_config = PPOConfig(**ppo_config_dict)
        # Guard against incompatible TRL versions (>=0.12 requires reward_model & datasets)
        import trl as _trl_pkg
        _trl_ver = getattr(_trl_pkg, "__version__", "0.0.0")
        try:
            _major_minor = tuple(int(p) for p in _trl_ver.split(".")[:2])
        except Exception:
            _major_minor = (0, 0)
        if _major_minor >= (0, 12):
            raise RuntimeError(
                f"Detected TRL version {_trl_ver}. This code targets TRL 0.7.x. "
                "Please install trl==0.7.9 (see pyproject.toml) or update the code to the new TRL API."
            )

        # Initialize PPOTrainer (compatible with TRL 0.7.x API)
        from trl import PPOTrainer
        self.ppo_trainer = PPOTrainer(  # <-- positional args avoid kwarg signature surprises
            self.ppo_config, self.model, self.model_ref, self.tokenizer
        )
        
        # hard-pin PPOTrainer's device to CPU (or your chosen device)
        import torch
        self.ppo_trainer.current_device = torch.device(self.device)  # e.g., torch.device("cpu")
        # best-effort: align any internal accelerator if present
        if hasattr(self.ppo_trainer, "accelerator"):
            try:
                self.ppo_trainer.accelerator.device = torch.device(self.device)
            except Exception:
                pass
        
        # Stability features
        self.sft_warm_start = sft_warm_start
        self.sft_steps = sft_steps
        self.topk_p = topk_p
        self.use_target_kl = use_target_kl
        self.target_kl = target_kl
        self.reward_scale = reward_scale
        
        self.replay_buffer = ReplayBuffer() if sft_warm_start else None
        
        # Initialize SFT trainer if warm start is enabled
        self.sft_trainer = None
        if sft_warm_start and self.replay_buffer:
            self.sft_trainer = SFTTrainer(
                model=self.model,
                tokenizer=self.tokenizer, 
                ppo_trainer=self.ppo_trainer,
                replay_buffer=self.replay_buffer,
                sft_steps=sft_steps,
                topk_p=topk_p
            )

        self.current_batch = {"queries": [], "responses": [], "rewards": []}

        self.current_episode_messages = [
            {
                "role": "system",
                "content": self.get_system_prompt(),
            }
        ]
        self.current_episode_rewards = []
        self.current_episode_data = []

    @abstractmethod
    def get_system_prompt(self) -> str:
        pass

    @abstractmethod
    def format_observation(self, observation: gym.core.ObsType) -> str:
        pass

    @abstractmethod
    def extract_action(self, response: str) -> gym.core.ActType:
        pass
    
    def _try_json_action_parse(self, response: str):
        """Try to parse action from JSON format first."""
        try:
            data = json.loads(response.strip())
            if "action" in data:
                a = data["action"]
                if isinstance(a, str) and a.isdigit():
                    return int(a)
                return a
        except (json.JSONDecodeError, TypeError):
            pass
        return None
    

    def llm(self, messages: List[Dict[str, str]]) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # use the model's real device
        model_device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        pad_id = self.tokenization_utils.get_pad_token_id()

        generate_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            pad_token_id=pad_id,
            **{k.split("/")[-1]: v for k, v in self.generate_config_dict.items()},
        )
        new_tokens = generate_ids[0, inputs["input_ids"].size(1):]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def act(self, observation):
        message = self.format_observation(observation)
        self.current_episode_messages += [{"role": "user", "content": message}]

        response = self.llm(self.current_episode_messages)
        
        # Try JSON parsing first, then fallback to extract_action
        json_action = self._try_json_action_parse(response)
        if json_action is not None:
            action = json_action
        else:
            try:
                action = self.extract_action(response)
            except Exception:
                # minimal, env-agnostic fallback; change if you have action_space
                action = 0

        self.current_episode_messages += [{"role": "assistant", "content": response}]
        
        # Store step data for replay buffer
        if self.replay_buffer is not None:
            step_data = {
                "obs_text": message,
                "response_text": response,
                "action": action,
                "reward": None  # Will be filled in assign_reward
            }
            self.current_episode_data.append(step_data)
        
        return action

    def assign_reward(self, reward):
        self.current_episode_rewards.append(reward)
        
        # Update the most recent step data with reward
        if self.replay_buffer is not None and self.current_episode_data:
            self.current_episode_data[-1]["reward"] = reward

    def format_episode_for_ppo(self, messages, rewards):
        queries, responses = [], []
        model_device = next(self.model.parameters()).device
        
        # Walk pairs of (user, assistant) using token-based splitting
        for i in range(2, len(messages), 2):
            msgs_prefix = messages[:i]      # ...system,(u,a)*, user_k  
            msgs_full = messages[:i+1]      # ... + assistant_k
            q_ids, r_ids = self.tokenization_utils.qr_token_ids(msgs_prefix, msgs_full, model_device)
            queries.append(q_ids)
            responses.append(r_ids)

        # Scale and squash rewards (stops value blow-ups)
        def _scale(x):  # gentle squash
            return torch.tanh(torch.tensor(float(x) * self.reward_scale, dtype=torch.float32))

        if all(reward == 0 for reward in rewards[:-1]):
            # if sparse rewards, give equal reward to all conversation turns
            per_turn_reward = rewards[-1] / max(1, len(queries))
            rewards = [_scale(per_turn_reward) for _ in range(len(queries))]
        else:
            rewards = [_scale(r) for r in rewards]

        return queries, responses, rewards

    def terminate_episode(self, train=True):
        if train:
            queries, responses, rewards = self.format_episode_for_ppo(
                self.current_episode_messages, self.current_episode_rewards
            )

        # Store episode in replay buffer before resetting
        if self.replay_buffer is not None and self.current_episode_data:
            total_return = sum(self.current_episode_rewards)
            for step in self.current_episode_data:
                step["total_return"] = total_return
            self.replay_buffer.add_episode(self.current_episode_data.copy())

        self.current_episode_messages = [
            {
                "role": "system",
                "content": self.get_system_prompt(),
            }
        ]
        self.current_episode_rewards = []
        self.current_episode_data = []

        if train:
            self.current_batch["queries"].extend(queries)
            self.current_batch["responses"].extend(responses)
            self.current_batch["rewards"].extend(rewards)

            if len(self.current_batch["queries"]) >= self.ppo_config.batch_size:
                # 1) warm-start SFT first (optional / gated)
                sft_stats = {}
                if self.sft_trainer and len(self.replay_buffer) > 0:
                    sft_stats = self.sft_trainer.run_sft_warmstart(self.get_system_prompt) or {}
                
                # 2) then PPO update (value head will re-fit to the new trunk)
                train_stats = self.train_batch(
                    self.current_batch["queries"],
                    self.current_batch["responses"],
                    self.current_batch["rewards"],
                )
                
                # Combine SFT and PPO stats
                combined_stats = {**train_stats, **sft_stats}
                return combined_stats

        return {}

    def train_batch(self, batch_queries, batch_responses, batch_rewards):
        if len(batch_queries) > self.ppo_config.batch_size:
            queries = batch_queries[: self.ppo_config.batch_size]
            responses = batch_responses[: self.ppo_config.batch_size]
            rewards = batch_rewards[: self.ppo_config.batch_size]

            # keep the remainder for the next batch
            self.current_batch["queries"] = batch_queries[self.ppo_config.batch_size :]
            self.current_batch["responses"] = batch_responses[
                self.ppo_config.batch_size :
            ]
            self.current_batch["rewards"] = batch_rewards[self.ppo_config.batch_size :]
        else:
            queries, responses, rewards = batch_queries, batch_responses, batch_rewards
            self.current_batch = {"queries": [], "responses": [], "rewards": []}

        train_stats = self.ppo_trainer.step(queries, responses, rewards)
        
        # Add KL monitoring if enabled (for newer TRL versions that don't support it in config)
        if self.use_target_kl:
            # For now, just log that KL monitoring is enabled
            # Full KL measurement could be re-added if needed
            train_stats["target_kl_enabled"] = True
            train_stats["target_kl"] = self.target_kl
        
        return train_stats
    
    def sft_from_jsonl(self, path: str, steps: int | None = None, **ingest_kwargs) -> int:
        """Warm-start SFT from an external JSONL of (obs, response) pairs.
        
        Args:
            path: Path to JSONL file containing training examples
            steps: Number of SFT steps to run (overrides current sft_steps if provided)
            **ingest_kwargs: Additional arguments passed to ingest_jsonl()
            
        Returns:
            Number of examples successfully ingested from the file
        """
        # Create replay buffer if it doesn't exist
        if self.replay_buffer is None:
            self.replay_buffer = ReplayBuffer()
        
        # Create SFT trainer if it doesn't exist but warm start is enabled
        if self.sft_trainer is None and self.sft_warm_start:
            self.sft_trainer = SFTTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                ppo_trainer=self.ppo_trainer,
                replay_buffer=self.replay_buffer,
                sft_steps=self.sft_steps,
                topk_p=self.topk_p
            )
        
        # Load data from JSONL file
        ingested = self.replay_buffer.ingest_jsonl(path, **ingest_kwargs)
        
        if ingested == 0:
            print(f"Warning: No valid examples loaded from {path}")
            return 0
        
        print(f"Loaded {ingested} examples from {path}")
        
        # Override SFT steps if provided
        if steps is not None:
            if self.sft_trainer:
                self.sft_trainer.sft_steps = steps
            else:
                self.sft_steps = steps
        
        # Run SFT if we have a trainer and data
        if self.sft_trainer and len(self.replay_buffer) > 0:
            sft_stats = self.sft_trainer.run_sft_warmstart(self.get_system_prompt)
            if sft_stats:
                print(f"SFT completed: {sft_stats.get('sft_steps', 0)} steps, "
                      f"loss: {sft_stats.get('sft_loss', 0.0):.4f}")
        else:
            print("Warning: SFT trainer not available. Data loaded but SFT not run.")
        
        return ingested
    
