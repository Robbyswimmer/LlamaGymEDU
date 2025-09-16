import argparse
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

from tqdm import trange
import wandb
import torch

try:
    import textworld.gym
    from textworld import EnvInfos
except ImportError as exc:  # pragma: no cover - TextWorld is optional
    raise SystemExit(
        "TextWorld is not installed. Run `python scripts/bootstrap.py --extras textworld` "
        "or `pip install llamagym[textworld]`."
    ) from exc

from llamagym import Agent, load_model_and_tokenizer

DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_GAME = Path("examples/tw_games/custom_game.z8")


class TextworldAgent(Agent):
    def get_system_prompt(self) -> str:
        return (
            "You will be playing a text-based game. Start by analysing the current room and inventory. "
            "Reason about the next best action and respond as: command: <your command>."
        )

    def format_observation(self, observation) -> str:
        observation = observation.split("$$$$$$$ \n\n")[-1].strip()
        return observation

    def extract_action(self, response: str):
        import re

        command_match = re.search(r"(C|c)ommand: (.+?)(?=\n|$)", response)
        if command_match:
            return command_match.group(2)
        return "look"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a TextWorld agent on laptop hardware.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model to use")
    parser.add_argument("--device", default="auto", help="Device to run on (auto, cpu, cuda, mps)")
    parser.add_argument("--episodes", type=int, default=50, help="Number of training episodes")
    parser.add_argument("--batch-size", type=int, default=4, help="PPO batch size")
    parser.add_argument("--load-in-8bit", action="store_true", help="Enable 8-bit loading when available")
    parser.add_argument("--max-steps", type=int, default=50, help="Maximum steps per episode")
    parser.add_argument("--game", type=Path, default=DEFAULT_GAME, help="Path to a TextWorld game file (.z8)")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--wandb-project", default=None, help="Override WANDB_PROJECT")
    parser.add_argument("--wandb-mode", default=None, help="Override WANDB_MODE")
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

    if not args.game.exists():
        raise FileNotFoundError(
            f"TextWorld game not found at {args.game}. Run from the repository root or provide --game."
        )

    hyperparams = {
        "model_name": args.model,
        "env": "TextWorld",
        "device": args.device,
        "load_in_8bit": args.load_in_8bit,
        "batch_size": args.batch_size,
        "episodes": args.episodes,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "generate/max_new_tokens": 96,
        "generate/do_sample": True,
        "generate/top_p": 0.6,
        "generate/temperature": 0.9,
    }

    run = init_wandb(hyperparams, args.wandb_project, args.wandb_mode)

    model, tokenizer, actual_device = load_model_and_tokenizer(
        model_name=args.model,
        device=args.device,
        load_in_8bit=args.load_in_8bit,
    )

    agent = TextworldAgent(
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
        sft_warm_start=False,
        use_target_kl=True,
    )

    env_id = textworld.gym.register_game(
        str(args.game),
        max_episode_steps=args.max_steps,
        request_infos=EnvInfos(admissible_commands=True),
    )
    env = textworld.gym.make(env_id)

    for episode in trange(args.episodes):
        observation, info = env.reset()
        env.render()
        done = False

        while not done:
            action = agent.act(observation)
            wandb.log({"action": action})
            observation, reward, done, info = env.step(action)
            env.render()
            agent.assign_reward(reward)

        episode_stats = {
            "episode": episode,
            "total_return": sum(agent.current_episode_rewards),
            "message_ct": len(agent.current_episode_messages),
        }
        train_stats = agent.terminate_episode()
        for key, value in train_stats.items():
            if isinstance(value, (int, float)):
                episode_stats[key] = value
        wandb.log(episode_stats)

    env.close()
    if run:
        wandb.finish()


if __name__ == "__main__":
    main()
