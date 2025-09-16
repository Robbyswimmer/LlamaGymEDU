import os
import warnings
# Suppress verbose warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="bitsandbytes") 
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

# os.environ["CUDA_VISIBLE_DEVICES"] = ""        # hide CUDA (remove this line to use GPU)
os.environ.pop("ACCELERATE_USE_MPS_DEVICE", None)  # avoid MPS auto-pick if you want pure CPU

import torch
# Set fp32 as default for MPS stability with PPO+LoRA
torch.set_default_dtype(torch.float32)

from tqdm import trange
import wandb

import re
import gymnasium as gym
from llamagym import Agent, load_model_and_tokenizer, model_registry


class BlackjackAgent(Agent):
    def get_system_prompt(self) -> str:
        return """You are an expert blackjack player. Every turn, you'll see your current sum, the dealer's showing card value, and whether you have a usable ace. Win by exceeding the dealer's hand but not exceeding 21.
Respond in JSON format: {"action": 0} to stay with your current sum or {"action": 1} to accept another card. Accept a card unless very close to 21."""

    def format_observation(self, observation: gym.core.ObsType) -> str:
        return f"You: {observation[0]}. Dealer: {observation[1]}. You have {'an' if bool(observation[2]) else 'no'} ace."

    def extract_action(self, response: str) -> gym.core.ActType:
        # First try to parse JSON format: {"action": 0} or {"action": 1}
        import json
        try:
            # Handle malformed JSON with extra quotes
            cleaned_response = response.replace('""', '"')
            parsed = json.loads(cleaned_response)
            if 'action' in parsed:
                action = int(parsed['action'])
                if action in [0, 1]:
                    return action
        except:
            pass
        
        # Try to find "action": followed by a digit
        match = re.compile(r'"action"\s*:\s*(\d)').search(response)
        if match:
            action = int(match.group(1))
            if action in [0, 1]:
                return action
        
        # Original fallback logic
        match = re.compile(r"Action: (\d)").search(response)
        if match:
            return int(match.group(1))

        digits = [char for char in response if char.isdigit()]
        if len(digits) == 0 or digits[-1] not in ("0", "1"):
            if "stick" in response.lower():
                return 0
            elif "hit" in response.lower():
                return 1

        return 0


if __name__ == "__main__":
    # Configuration - try different models by changing model_name
    hyperparams = {
        "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",  # Llama 3.1 8B instruction-tuned model
        "env": "Blackjack-v1",
        "device": "cpu",  # Forcing CPU to avoid MPS stability issues
        "load_in_8bit": False,  # Enable for memory optimization
        "batch_size": 4,  # Smaller batch for faster training
        "seed": 42069,
        "episodes": 20,  # Quick test with 8B model  
        # Optimized generation config for Llama 3.1
        "generate/max_new_tokens": 32,  # Increased to ensure complete JSON responses
        "generate/do_sample": False,  # Use greedy decoding for stability
        "generate/temperature": None,  # Explicitly disable temperature for greedy decoding
        "generate/top_p": None,  # Explicitly disable top_p for greedy decoding
    }
    
    wandb_run = wandb.init(project=os.environ.get("WANDB_PROJECT"), config=hyperparams)
    
    # Smart model loading with automatic configuration
    print(f"📋 Available model families: {list(model_registry.list_families().keys())}")
    
    model, tokenizer, actual_device = load_model_and_tokenizer(
        model_name=hyperparams["model_name"],
        device=hyperparams["device"],
        load_in_8bit=hyperparams["load_in_8bit"],
        # Custom generation config can override model defaults
        # peft_config=custom_lora_config,  # Uncomment to use custom PEFT
    )

    agent = BlackjackAgent(
        model,
        tokenizer,
        actual_device,
        # Extract generation config from hyperparams
        generate_config_dict={
            key.split("/")[-1]: value
            for key, value in hyperparams.items()
            if key.startswith("generate/")
        },
        ppo_config_dict={
            "batch_size": hyperparams["batch_size"],
            "mini_batch_size": hyperparams["batch_size"],
        },
        # Enable stability features
        sft_warm_start=True,   # Offline→online bridge  
        use_target_kl=True     # Prevent policy drift
    )
    
    # Load expert demonstrations for SFT warm-start
    agent.sft_from_jsonl("demo_blackjack.jsonl", steps=3, max_items=200)
    
    env = gym.make(hyperparams["env"], natural=False, sab=False)

    for episode in trange(hyperparams["episodes"]):
        observation, info = env.reset()
        done = False

        step = 0
        while not done:
            # Format the observation for the model
            formatted_obs = agent.format_observation(observation)
            print(f"\n--- Episode {episode}, Step {step} ---")
            print(f"Observation: {observation}")
            print(f"Formatted for model: {formatted_obs}")
            
            # Get model response and action
            action = agent.act(observation)
            
            # Get the last message to see what the model actually responded
            if hasattr(agent, 'current_episode_messages') and agent.current_episode_messages:
                last_message = agent.current_episode_messages[-1]
                if last_message.get('role') == 'assistant':
                    model_response = last_message.get('content', 'No content')
                    print(f"Model response: {model_response}")
            
            print(f"Extracted action: {action}")
            
            wandb.log({"action": action})
            observation, reward, terminated, truncated, info = env.step(action)
            print(f"Reward: {reward}")
            agent.assign_reward(reward)
            done = terminated or truncated
            step += 1

        episode_stats = {
            "episode": episode,
            "total_return": sum(agent.current_episode_rewards),
            "message_ct": len(agent.current_episode_messages),
            "episode_messages": agent.current_episode_messages,
        }
        train_stats = agent.terminate_episode()
        episode_stats.update(train_stats)
        
        # Filter out NaN/inf values and problematic data types to prevent wandb logging errors
        import math
        import numpy as np
        import torch
        
        def convert_for_wandb(value):
            """Convert value to a wandb-safe format."""
            if value is None:
                return None
            if isinstance(value, (int, bool)):
                return value
            if isinstance(value, float):
                if math.isnan(value) or math.isinf(value):
                    return None
                return value
            if isinstance(value, torch.Tensor):
                try:
                    # Convert tensor to scalar mean if it contains valid data
                    if value.numel() == 1:
                        scalar_val = value.item()
                        if math.isnan(scalar_val) or math.isinf(scalar_val):
                            return None
                        return scalar_val
                    else:
                        # For multi-element tensors, use mean if finite
                        mean_val = value.float().mean().item()
                        if math.isnan(mean_val) or math.isinf(mean_val):
                            return None
                        return mean_val
                except:
                    return None
            if isinstance(value, np.ndarray):
                try:
                    if value.size == 1:
                        scalar_val = float(value.item())
                        if math.isnan(scalar_val) or math.isinf(scalar_val):
                            return None
                        return scalar_val
                    else:
                        mean_val = float(np.mean(value))
                        if math.isnan(mean_val) or math.isinf(mean_val):
                            return None
                        return mean_val
                except:
                    return None
            if isinstance(value, (list, tuple)):
                # For small lists of numbers, keep them; for large ones or problematic ones, skip
                if len(value) > 1000:
                    return None
                try:
                    # Check if it's a list of numbers
                    if all(isinstance(v, (int, float)) for v in value):
                        clean_values = [v for v in value if not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))]
                        if len(clean_values) == 0:
                            return None
                        return clean_values
                    # For non-numeric lists (like episode_messages), skip to avoid wandb issues
                    return None
                except:
                    return None
            if isinstance(value, str):
                return value if len(value) < 10000 else None
            return None
        
        filtered_stats = {}
        for key, value in episode_stats.items():
            converted_value = convert_for_wandb(value)
            if converted_value is not None:
                filtered_stats[key] = converted_value
            # Suppress the skipping messages to focus on model behavior
        
        try:
            wandb.log(filtered_stats)
        except Exception as e:
            print(f"Wandb logging failed: {e}")
            # Log minimal safe stats
            safe_stats = {
                "episode": episode,
                "total_return": sum(agent.current_episode_rewards) if agent.current_episode_rewards else 0,
                "message_ct": len(agent.current_episode_messages)
            }
            wandb.log(safe_stats)
