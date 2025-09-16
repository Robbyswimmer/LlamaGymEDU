import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from llamagym.replay import ReplayBuffer
from llamagym.sft_trainer import SFTTrainer


class DummyTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.eos_token = "<eos>"
        self.truncation_side = "right"
        self.padding_side = "left"
        self.model_max_length = 256
        self.name_or_path = "dummy"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = [f"{m['role']}::{m['content']}" for m in messages]
        text = "\n".join(parts)
        if add_generation_prompt:
            text += "\nassistant::"
        return text

    def __call__(self, text, return_tensors="pt", truncation=False, max_length=None):
        if isinstance(text, (list, tuple)):
            text = "".join(text)
        encoded = [(ord(ch) % 50) + 2 for ch in text]
        if not encoded:
            encoded = [0]
        input_ids = torch.tensor(encoded, dtype=torch.long).unsqueeze(0)
        if max_length is not None and input_ids.size(1) > max_length:
            input_ids = input_ids[:, -max_length:]
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, ids, skip_special_tokens=True):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join(chr((i - 2) % 50) for i in ids)


class DummyModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 64, hidden_size: int = 8):
        super().__init__()
        self.embed = torch.nn.Embedding(256, hidden_size)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size)
        self.v_head = torch.nn.Linear(hidden_size, 1)
        self.config = SimpleNamespace(use_cache=True)

    def forward(self, input_ids, attention_mask=None):
        hidden = self.embed(input_ids)
        logits = self.lm_head(hidden)
        return SimpleNamespace(logits=logits)


class DummyPbar:
    def __init__(self):
        self.total_updates = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def update(self, n):
        self.total_updates += n

    def set_postfix(self, **kwargs):
        pass


class ReplayBufferSummaryTest(unittest.TestCase):
    def test_summary_contains_expected_metrics(self):
        buffer = ReplayBuffer(seed=0)
        buffer.add_episode([
            {"obs_text": "obs1", "response_text": "resp1", "reward": 0.6, "total_return": 1.0},
            {"obs_text": "obs2", "response_text": "resp2", "reward": 0.4, "total_return": 1.0},
        ])
        buffer.add_episode([
            {"obs_text": "obs3", "response_text": "resp3", "reward": -1.0, "total_return": -1.0},
        ])

        summary = buffer.summary()
        self.assertEqual(summary["replay/episodes"], 2.0)
        self.assertEqual(summary["replay/steps"], 3.0)
        self.assertAlmostEqual(summary["replay/return/mean"], 0.0)
        self.assertAlmostEqual(summary["replay/return/std"], 1.0)
        self.assertEqual(summary["replay/pair_count"], 3.0)
        self.assertEqual(summary["replay/pair_unique"], 3.0)
        self.assertAlmostEqual(summary["replay/success_rate"], 0.5)


class SFTTrainerMetricsTest(unittest.TestCase):
    def test_warmstart_metrics_are_prefixed(self):
        buffer = ReplayBuffer(seed=0)
        buffer.add_episode([
            {"obs_text": "first", "response_text": "resp_a", "reward": 1.0, "total_return": 2.0},
            {"obs_text": "second", "response_text": "resp_b", "reward": 1.0, "total_return": 2.0},
        ])
        buffer.add_episode([
            {"obs_text": "third", "response_text": "resp_c", "reward": 0.2, "total_return": 1.0},
        ])

        model = DummyModel()
        tokenizer = DummyTokenizer()
        trainer = SFTTrainer(model, tokenizer, ppo_trainer=mock.Mock(), replay_buffer=buffer, sft_steps=2, topk_p=0.5)

        expected_pairs = len(buffer.get_sft_data(buffer.sample_topk(0.5), max_pairs=128, dedup=True))

        with mock.patch.object(SFTTrainer, "_sft_step", return_value=0.25) as step_mock, \
                mock.patch("llamagym.sft_trainer.tqdm", return_value=DummyPbar()):
            stats = trainer.run_sft_warmstart(lambda: "system prompt")

        self.assertIn("sft/loss", stats)
        self.assertAlmostEqual(stats["sft/loss"], 0.25)
        self.assertEqual(int(stats["sft/steps"]), expected_pairs * trainer.sft_steps)
        self.assertEqual(stats["sft/skipped_steps"], 0.0)
        self.assertGreaterEqual(stats["sft/episodes_used"], 1.0)
        self.assertEqual(stats["sft/pairs_used"], float(expected_pairs))
        self.assertEqual(stats["sft/optimizer/lr"], trainer.sft_lr)
        self.assertEqual(step_mock.call_count, expected_pairs * trainer.sft_steps)


class WarmstartDataTest(unittest.TestCase):
    def test_demo_blackjack_warmstart_data(self):
        data_path = Path(__file__).resolve().parents[1] / "demo_blackjack.jsonl"
        buffer = ReplayBuffer()
        count = buffer.ingest_jsonl(str(data_path))

        self.assertGreater(count, 0)
        summary = buffer.summary()
        self.assertEqual(summary["replay/episodes"], float(count))
        self.assertEqual(summary["replay/pair_count"], float(count))
        self.assertGreaterEqual(summary["replay/pair_unique"], 1.0)

        for episode in list(buffer.episodes)[:5]:
            self.assertTrue(episode[0]["response_text"].startswith("{"))


if __name__ == "__main__":
    unittest.main()
