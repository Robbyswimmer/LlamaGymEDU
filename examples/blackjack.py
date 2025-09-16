import argparse
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="bitsandbytes")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

import torch

torch.set_default_dtype(torch.float32)

from tqdm import trange
import wandb

import json
import math
import re
from pathlib import Path

import gymnasium as gym

from llamagym import Agent, load_model_and_tokenizer, model_registry

DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


class BlackjackAgent(Agent):
    def get_system_prompt(self) -> str:
        return """You are an expert blackjack player. Every turn, you'll see your current sum, the dealer's showing card value, and whether you have a usable ace. Win by exceeding the dealer's hand but not exceeding 21. Respond in JSON format: {\"action\": 0} to stay with your current sum or {\"action\": 1} to accept another card. Accept a card unless very close to 21."""

    def format_observation(self, observation: gym.core.ObsType) -> str:
        return f"You: {observation[0]}. Dealer: {observation[1]}. You have {'an' if bool(observation[2]) else 'no'} ace."

    def extract_action(self, response: str) -> gym.core.ActType:
        try:
            cleaned = response.replace('""', '"')
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict) and "action" in parsed:
                action = int(parsed["action"])
                if action in (0, 1):
                    return action
        except Exception:
            pass

        match = re.compile(r'"action"\s*:\s*(\d)').search(response)
        if match:
            action = int(match.group(1))
            if action in (0, 1):
                return action

        match = re.compile(r"Action: (\d)").search(response)
        if match:
            return int(match.group(1))

        digits = [char for char in response if char.isdigit()]
        if digits and digits[-1] in ("0", "1"):
            return int(digits[-1])

        if "stick" in response.lower():
            return 0
        if "hit" in response.lower():
            return 1
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a blackjack agent with a laptop-friendly model.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model name to use")
    parser.add_argument("--device", default="auto", help="Device to run on (auto, cpu, cuda, mps)")
    parser.add_argument("--episodes", type=int, default=30, help="Number of training episodes")
    parser.add_argument("--batch-size", type=int, default=4, help="PPO batch size")
    parser.add_argument("--load-in-8bit", action="store_true", help="Enable 8-bit loading when supported")
    parser.add_argument("--max-new-tokens", type=int, default=48, help="Max tokens to generate per step")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for reproducibility")
    parser.add_argument("--wandb-project", default=None, help="Override the WANDB_PROJECT environment variable")
    parser.add_argument("--wandb-mode", default=None, help="Override WANDB_MODE (default: disabled if no project)")
    parser.add_argument(
        "--online-sft",
        action="store_true",
        help="Allow SFT warm-starting to continue during PPO updates",
    )
    return parser.parse_args()


def init_wandb(config: dict, project_override: str | None, mode_override: str | None):
    project = project_override or os.environ.get("WANDB_PROJECT")
    mode = mode_override or os.environ.get("WANDB_MODE")
    if mode is None:
        mode = "disabled" if project is None else "online"
    return wandb.init(project=project, config=config, mode=mode)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    hyperparams = {
        "model_name": args.model,
        "env": "Blackjack-v1",
        "device": args.device,
        "load_in_8bit": args.load_in_8bit,
        "batch_size": args.batch_size,
        "episodes": args.episodes,
        "seed": args.seed,
        "generate/max_new_tokens": args.max_new_tokens,
        "generate/do_sample": False,
        "generate/temperature": None,
        "generate/top_p": None,
        "online_sft": args.online_sft,
    }

    run = init_wandb(hyperparams, args.wandb_project, args.wandb_mode)
    print(f"📋 Available model families: {list(model_registry.list_families().keys())}")

    model, tokenizer, actual_device = load_model_and_tokenizer(
        model_name=args.model,
        device=args.device,
        load_in_8bit=args.load_in_8bit,
    )

    agent = BlackjackAgent(
        model,
        tokenizer,
        actual_device,
        generate_config_dict={
            key.split("/")[-1]: value
            for key, value in hyperparams.items()
            if key.startswith("generate/")
        },
        ppo_config_dict={
            "batch_size": args.batch_size,
            "mini_batch_size": args.batch_size,
        },
        sft_warm_start=True,
        use_target_kl=True,
    )

    demo_path = Path("demo_blackjack.jsonl")
    if demo_path.exists():
        agent.sft_from_jsonl(str(demo_path), steps=2, max_items=150)
    else:
        print("⚠️  demo_blackjack.jsonl not found; skipping SFT warm-start")

    if not args.online_sft:
        agent.disable_online_sft()

    env = gym.make("Blackjack-v1", natural=False, sab=False)

    for episode in trange(args.episodes):
        observation, info = env.reset()
        done = False
        step = 0

        while not done:
            formatted_obs = agent.format_observation(observation)
            print(f"\n--- Episode {episode}, Step {step} ---")
            print(f"Observation: {observation}")
            print(f"Formatted: {formatted_obs}")

            action = agent.act(observation)

            if getattr(agent, "current_episode_messages", None):
                last_message = agent.current_episode_messages[-1]
                if last_message.get("role") == "assistant":
                    print(f"Model response: {last_message.get('content', 'No content')}")

            print(f"Chosen action: {action}")

            wandb.log({"action": action})
            observation, reward, terminated, truncated, info = env.step(action)
            print(f"Reward: {reward}")
            agent.assign_reward(reward)
            done = terminated or truncated
            step += 1

        episode_messages = list(agent.current_episode_messages)
        episode_return = sum(agent.current_episode_rewards)
        episode_length = step

        train_stats = agent.terminate_episode()
        episode_stats = {
            "episode": episode,
            "total_return": episode_return,
            "rl/episode_return": episode_return,
            "rl/episode_length": episode_length,
            "message_ct": len(episode_messages),
            "episode_messages": episode_messages,
        }

        if getattr(agent, "replay_buffer", None) is not None:
            episode_stats.update(agent.replay_buffer.summary())

        episode_stats.update(train_stats)

        def convert_for_wandb(value):
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
                    if value.numel() == 1:
                        scalar_val = value.item()
                        if math.isnan(scalar_val) or math.isinf(scalar_val):
                            return None
                        return scalar_val
                    mean_val = value.float().mean().item()
                    if math.isnan(mean_val) or math.isinf(mean_val):
                        return None
                    return mean_val
                except Exception:
                    return None
            if hasattr(value, "tolist"):
                try:
                    return convert_for_wandb(float(value))
                except Exception:
                    return None
            return None

        clean_stats = {}
        for key, value in episode_stats.items():
            converted = convert_for_wandb(value)
            if converted is not None:
                clean_stats[key] = converted
        if clean_stats:
            wandb.log(clean_stats)

    env.close()
    if run:
        wandb.finish()


if __name__ == "__main__":
    main()
