<p align="center">
  <img src="https://raw.githubusercontent.com/khoomeik/LlamaGym/main/llamagym.png" height="250" alt="Llama Gym" />
</p>
<p align="center">
  <em>Fine-tune LLM agents with online reinforcement learning</em>
</p>
<p align="center">
    <a href="https://pypi.org/project/llamagym/" target="_blank">
        <img alt="Python" src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" />
        <img alt="Version" src="https://img.shields.io/pypi/v/llamagym?style=for-the-badge&color=3670A0">
    </a>
</p>
<p align="center">
<a href="https://reworkd.ai/">🔗 Agents for Web Data Extraction</a>
<span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
<a href="https://x.com/khoomeik/status/1766805213644800011">🐦 Twitter</a>


# LlamaGym
"Agents" originated in reinforcement learning, where they learn by interacting with an environment and receiving a reward signal. However, LLM-based agents today do not learn online (i.e. continuously in real time) via reinforcement.

OpenAI created [Gym](https://github.com/Farama-Foundation/Gymnasium) to standardize and simplify RL environments, but if you try dropping an LLM-based agent into a Gym environment for training, you'd find it's still quite a bit of code to handle LLM conversation context, episode batches, reward assignment, PPO setup, and more.

LlamaGym seeks to simplify fine-tuning LLM agents with RL. Right now, it's a single `Agent` abstract class that handles all the issues mentioned above, letting you quickly iterate and experiment with agent prompting & hyperparameters across any Gym environment.

## 🚀 Quickstart (macOS / Windows / Linux)

1. **Clone the project and bootstrap the environment**

   ```bash
   git clone https://github.com/KhoomeiK/LlamaGymEDU.git
   cd LlamaGymEDU
   python scripts/bootstrap.py
   ```

   The bootstrapper creates a `.venv` virtual environment, installs the pinned dependencies from `requirements/base.txt`, and automatically adds Linux-only extras like `bitsandbytes`. Pass `--extras textworld` to add the optional TextWorld stack or `--no-platform-extras` if you want a minimal install. Run `python scripts/bootstrap.py --help` to see every option.

2. **Activate the virtual environment**

   ```bash
   source .venv/bin/activate          # macOS / Linux
   .\.venv\Scripts\activate           # Windows PowerShell
   ```

3. **Disable external logging unless you need it**

   W&B logging defaults to `mode="disabled"` unless `WANDB_PROJECT` is set. You can override the behaviour by exporting `WANDB_MODE` (e.g. `offline` or `online`). Hugging Face models that require authentication can use the standard `HF_TOKEN` environment variable.

### Run a quick sanity check

Laptop-friendly defaults ship with the repository so that even CPU-only machines can experiment quickly.

```bash
python examples/blackjack.py --episodes 5
```

The script loads `TinyLlama/TinyLlama-1.1B-Chat-v1.0` by default (≈1.1B parameters) and runs fully on CPU or Apple M-series hardware. To try TextWorld, install the optional dependencies and run:

```bash
python scripts/bootstrap.py --extras textworld
python examples/text-world.py --episodes 5
```

Both examples accept `--model`, `--device`, and other quality-of-life flags so students can switch between CPU, CUDA, or MPS and try other compact models without editing code.

### Laptop-sized model suggestions

| Model | Parameters | Notes |
| --- | --- | --- |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 1.1B | Fits in <4 GB of RAM; great for CPU-only runs |
| `microsoft/phi-2` or `microsoft/phi-3-mini-4k-instruct` | 2.7B–3.8B | Requires ~8 GB RAM; strong instruction following |
| `HuggingFaceTB/SmolLM-1.7B-Instruct` | 1.7B | Community model tuned for fast experiments |
| `Qwen/Qwen2-1.5B-Instruct` | 1.5B | Multilingual baseline with solid reasoning |

All of these models have dedicated entries in `llamagym.model_registry`, so `load_model_and_tokenizer` automatically applies appropriate generation and LoRA defaults.

## Usage
Fine-tuning an LLM-based agent to play in a Gym-style environment with RL has never been easier! Once you install LlamaGym...
```
pip install llamagym
```

First, implement 3 abstract methods on the Agent class:
```python
from llamagym import Agent

class BlackjackAgent(Agent):
    def get_system_prompt(self) -> str:
        return "You are an expert blackjack player."

    def format_observation(self, observation) -> str:
        return f"Your current total is {observation[0]}"

    def extract_action(self, response: str):
        return 0 if "stay" in response else 1
```

Then, define your base LLM (as you would for any fine-tuning job) and instantiate your agent:
```python
model = AutoModelForCausalLMWithValueHead.from_pretrained("Llama-2-7b").to(device)
tokenizer = AutoTokenizer.from_pretrained("Llama-2-7b")
agent = BlackjackAgent(model, tokenizer, device)
```

Finally, write your RL loop as usual and simply call your agent to act, reward, and terminate:
```python
env = gym.make("Blackjack-v1")

for episode in trange(5000):
    observation, info = env.reset()
    done = False

    while not done:
        action = agent.act(observation) # act based on observation
        observation, reward, terminated, truncated, info = env.step(action)
        agent.assign_reward(reward) # provide reward to agent
        done = terminated or truncated

    train_stats = agent.terminate_episode() # trains if batch is full
```

## Stability Features

LlamaGym includes optional stability features that can dramatically improve RL convergence:

```python
agent = BlackjackAgent(
    model, tokenizer, device,
    # Stability toggles (all optional, default False)
    sft_warm_start=True,  # Offline→online bridge via replay buffer
    use_target_kl=True    # Target-KL early stopping
)
```

**SFT Warm-Start**: Maintains a small replay buffer of successful episodes and runs supervised fine-tuning steps on top-performing trajectories after each PPO update. Provides the "offline→online bridge" that stabilizes pure online RL.

**Target-KL Controller**: Enables TRL's built-in KL divergence monitoring and early stopping to prevent catastrophic policy drift.

**Robust Action Extraction**: Automatically tries JSON parsing first (e.g., `{"action": 0}`), then falls back to regex, reducing action parsing failures.

Some reminders:
- above code snippets are mildly simplified above but a fully working example is available in [`examples/blackjack.py`](https://github.com/KhoomeiK/LlamaGym/blob/main/examples/blackjack.py)
- getting online RL to converge is notoriously difficult so you'll have to mess with hyperparameters to see improvement
- our implementation values simplicity so is not as compute efficient as e.g. [Lamorel](https://github.com/flowersteam/lamorel), but easier to start playing around with
- LlamaGym is a weekend project and still a WIP, but we love contributions!

## Relevant Work
- [Grounding Large Language Models with Online Reinforcement Learning](https://github.com/flowersteam/Grounding_LLMs_with_online_RL)
  - [Lamorel: Language Models for Reinforcement Learning](https://github.com/flowersteam/lamorel)
- [True Knowledge Comes from Practice: Aligning LLMs with Embodied Environments via Reinforcement Learning](https://github.com/WeihaoTan/TWOSOME)

## Citation
```
bibtex
@misc{pandey2024llamagym,
  title        = {LlamaGym: Fine-tune LLM agents with Online Reinforcement Learning},
  author       = {Rohan Pandey},
  year         = {2024},
  howpublished = {GitHub},
  url          = {https://github.com/KhoomeiK/LlamaGym}
}
```
