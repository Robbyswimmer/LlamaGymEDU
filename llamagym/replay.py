from collections import deque
from typing import List, Tuple, Dict, Iterable
import random
import json


class ReplayBuffer:
    def __init__(self, max_size: int = 1000, seed: int | None = None):
        self.max_size = max_size
        self.episodes = deque(maxlen=max_size)
        self._rng = random.Random(seed) if seed is not None else random
    
    def add_episode(self, episode_data: List[Dict]) -> None:
        """Add an episode to the replay buffer.
        
        Args:
            episode_data: List of step dicts with keys:
                - obs_text: observation as string
                - response_text: LLM response
                - action: parsed action
                - reward: step reward
                - total_return: episode total return
        """
        if episode_data:
            self.episodes.append(episode_data)
    
    def sample_topk(self, p: float = 0.2) -> List[List[Dict]]:
        """Sample top percentile episodes by total return.
        
        Args:
            p: Percentile threshold (0.2 = top 20%)
            
        Returns:
            List of top episodes with diversity shuffle
        """
        if not self.episodes:
            return []
        
        # Sort episodes by total return (descending) - safe key access
        sorted_episodes = sorted(
            self.episodes,
            key=lambda ep: (ep and ep[0].get("total_return", 0)) or 0,
            reverse=True
        )
        
        # Get top percentile
        k = max(1, int(len(sorted_episodes) * p))
        top = sorted_episodes[:k]
        self._rng.shuffle(top)  # inject tiny diversity
        return top
    
    def get_sft_data(self, episodes: List[List[Dict]], max_pairs: int = 128, dedup: bool = True) -> List[Tuple[str, str]]:
        """Extract (observation, response) pairs for SFT training.
        
        Args:
            episodes: List of episodes from sample_topk()
            max_pairs: Maximum number of pairs to return
            dedup: Remove duplicate (obs, resp) pairs
            
        Returns:
            List of (obs_text, response_text) tuples
        """
        sft_pairs, seen = [], set()
        for episode in episodes:
            for step in episode:
                obs, resp = step.get("obs_text"), step.get("response_text")
                if not obs or not resp:
                    continue
                if dedup:
                    key = (obs, resp)
                    if key in seen:
                        continue
                    seen.add(key)
                sft_pairs.append((obs, resp))
                if len(sft_pairs) >= max_pairs:
                    return sft_pairs
        return sft_pairs

    def sample_steps(self, n: int = 64) -> List[Dict]:
        """Return up to n steps, weighted by episode total_return (>=0).
        
        Args:
            n: Number of steps to sample
            
        Returns:
            List of step dictionaries sampled by return weights
        """
        if not self.episodes:
            return []
        buckets, weights = [], []
        for ep in self.episodes:
            if not ep:
                continue
            ret = max(0.0, float(ep[0].get("total_return", 0)))
            if ret == 0:
                continue
            buckets.append(ep)
            weights.append(ret)
        if not buckets:
            # fallback: uniform over all steps if all returns are 0/negative
            flat = [s for ep in self.episodes for s in ep]
            self._rng.shuffle(flat)
            return flat[:n]
        picks = self._rng.choices(buckets, weights=weights, k=n)
        return [self._rng.choice(ep) for ep in picks]
    
    def __len__(self) -> int:
        return len(self.episodes)

    def clear(self) -> None:
        """Clear all episodes from buffer."""
        self.episodes.clear()

    def summary(self) -> Dict[str, float]:
        """Return aggregate statistics for monitoring replay health."""
        episodes = [ep for ep in self.episodes if ep]
        metrics: Dict[str, float] = {
            "replay/episodes": float(len(episodes)),
            "replay/steps": float(sum(len(ep) for ep in episodes)),
        }

        if not episodes:
            metrics.update(
                {
                    "replay/return/mean": 0.0,
                    "replay/return/max": 0.0,
                    "replay/return/min": 0.0,
                    "replay/return/std": 0.0,
                    "replay/success_rate": 0.0,
                    "replay/pair_count": 0.0,
                    "replay/pair_unique": 0.0,
                }
            )
            return metrics

        returns = [float(ep[0].get("total_return", 0.0)) for ep in episodes]
        mean_return = sum(returns) / len(returns)
        metrics["replay/return/mean"] = mean_return
        metrics["replay/return/max"] = max(returns)
        metrics["replay/return/min"] = min(returns)
        if len(returns) > 1:
            variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
            metrics["replay/return/std"] = variance ** 0.5
        else:
            metrics["replay/return/std"] = 0.0
        metrics["replay/success_rate"] = sum(r > 0 for r in returns) / len(returns)

        pair_count = 0
        unique_pairs = set()
        for ep in episodes:
            for step in ep:
                obs = step.get("obs_text")
                resp = step.get("response_text")
                if obs and resp:
                    pair_count += 1
                    unique_pairs.add((obs, resp))

        metrics["replay/pair_count"] = float(pair_count)
        metrics["replay/pair_unique"] = float(len(unique_pairs))
        return metrics
    
    def ingest_pairs(self, pairs: Iterable[Tuple[str, str]], score: float = 1.0) -> None:
        """Insert (obs_text, response_text) pairs as 1-step episodes.
        
        Args:
            pairs: Iterable of (obs_text, response_text) tuples
            score: Score to assign to each episode (used for ranking)
        """
        for obs_text, response_text in pairs:
            if not obs_text or not response_text:
                continue
            episode = [{
                "obs_text": obs_text,
                "response_text": response_text,
                "action": None,
                "reward": None,
                "total_return": score
            }]
            self.episodes.append(episode)
    
    def ingest_jsonl(self, path: str,
                     obs_key: str = "obs_text",
                     resp_key: str = "response_text", 
                     score_key: str | None = None,
                     default_score: float = 1.0,
                     max_items: int | None = None) -> int:
        """Load pairs from a JSONL file and add as episodes.
        
        Args:
            path: Path to JSONL file
            obs_key: Key for observation text (also tries 'prompt', 'input')
            resp_key: Key for response text (also tries 'completion', 'output')
            score_key: Key for quality score (optional)
            default_score: Default score if no score_key or value found
            max_items: Maximum number of items to load (None for unlimited)
            
        Returns:
            Number of items successfully ingested
        """
        ingested = 0
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    if max_items is not None and ingested >= max_items:
                        break
                        
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        
                        # Try flexible field names for observation
                        obs_text = (data.get(obs_key) or 
                                   data.get("prompt") or 
                                   data.get("input"))
                        
                        # Try flexible field names for response  
                        resp_text = (data.get(resp_key) or
                                    data.get("completion") or
                                    data.get("output"))
                        
                        if not obs_text or not resp_text:
                            continue
                        
                        # Get score
                        score = default_score
                        if score_key and score_key in data:
                            try:
                                score = float(data[score_key])
                            except (ValueError, TypeError):
                                score = default_score
                        
                        # Ingest the pair
                        self.ingest_pairs([(obs_text, resp_text)], score=score)
                        ingested += 1
                        
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping malformed JSON on line {line_num}: {e}")
                        continue
                        
        except FileNotFoundError:
            print(f"Warning: JSONL file not found: {path}")
            return 0
        except Exception as e:
            print(f"Warning: Error reading JSONL file {path}: {e}")
            return 0
            
        return ingested