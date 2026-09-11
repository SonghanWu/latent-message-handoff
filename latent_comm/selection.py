"""The five selection rules that compete at equal budget.

All five answer the same question -- "which B positions of the worker's cache do we
ship?" -- and differ only in the score they rank positions by.  The point of the
experiment is the contrast between rules 3, 4 and 5:

    3. sender_surprisal          task-conditioned, blind to the receiver
    4. dedup_sender_surprisal    the same, minus anything the receiver literally holds
    5. receiver_surprisal        conditioned on the receiver's side information

    5 vs 3  isolates "does conditioning on the receiver help at all?"
    5 vs 4  isolates "does it help beyond dropping literal duplicates?" -- this is the
            claim the Wyner-Ziv reading makes and the one that can fail.

Two invariants worth knowing when reading results:

*   With no side information (s = 0) rules 3, 4 and 5 select identical sets by
    construction.  Any divergence there is a bug, and the experiment asserts it.
*   Every rule is forced to keep the first `n_sink` positions, and those count against
    the budget.  Attention sinks matter enough that leaving them to chance would make
    the sink, not the selection rule, the variable under test.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set

import torch

RULES = [
    "random",
    "sender_attention",
    "sender_surprisal",
    "dedup_sender_surprisal",
    "receiver_surprisal",
]


@dataclass
class Selection:
    rule: str
    positions: List[int]        # ascending, length == budget (or doc_len if smaller)
    budget: int
    wasted: int                 # how many selected positions the receiver already had


def _smooth(score: torch.Tensor, window: int) -> torch.Tensor:
    """SnapKV-style pooling: makes the selection pick short spans instead of confetti."""
    if window <= 1:
        return score
    pad = window // 2
    x = score.view(1, 1, -1).float()
    x = torch.nn.functional.avg_pool1d(
        torch.nn.functional.pad(x, (pad, pad), mode="replicate"),
        kernel_size=window,
        stride=1,
    )
    return x.view(-1)[: score.numel()]


def _top_k_positions(
    score: torch.Tensor,
    k: int,
    forced: Sequence[int],
    excluded: Optional[Set[int]] = None,
) -> List[int]:
    doc_len = int(score.numel())
    chosen = list(dict.fromkeys(int(p) for p in forced if 0 <= int(p) < doc_len))
    if len(chosen) >= k:
        return sorted(chosen[:k])

    masked = score.clone().float()
    masked[torch.tensor(chosen, dtype=torch.long, device=score.device)] = float("-inf")
    if excluded:
        idx = torch.tensor(
            sorted(p for p in excluded if 0 <= p < doc_len),
            dtype=torch.long,
            device=score.device,
        )
        if idx.numel():
            masked[idx] = float("-inf")

    remaining = k - len(chosen)
    finite = int(torch.isfinite(masked).sum())
    remaining = min(remaining, finite)
    if remaining > 0:
        extra = torch.topk(masked, remaining).indices.tolist()
        chosen.extend(int(i) for i in extra)
    return sorted(chosen)


def select(
    rule: str,
    budget: int,
    doc_len: int,
    attn_score: torch.Tensor,
    sender_surprisal: torch.Tensor,
    receiver_surprisal: torch.Tensor,
    side_positions: Sequence[int],
    n_sink: int = 4,
    pool_window: int = 1,
    seed: int = 0,
) -> Selection:
    budget = min(budget, doc_len)
    side_set = set(int(p) for p in side_positions)
    forced = list(range(min(n_sink, doc_len)))

    if rule == "random":
        rng = random.Random(f"{seed}:{budget}:{doc_len}")
        pool = [p for p in range(doc_len) if p not in set(forced)]
        rng.shuffle(pool)
        positions = sorted(forced + pool[: max(0, budget - len(forced))])
    elif rule == "sender_attention":
        positions = _top_k_positions(_smooth(attn_score, pool_window), budget, forced)
    elif rule == "sender_surprisal":
        positions = _top_k_positions(
            _smooth(sender_surprisal, pool_window), budget, forced
        )
    elif rule == "dedup_sender_surprisal":
        positions = _top_k_positions(
            _smooth(sender_surprisal, pool_window), budget, forced, excluded=side_set
        )
    elif rule == "receiver_surprisal":
        positions = _top_k_positions(
            _smooth(receiver_surprisal, pool_window), budget, forced
        )
    else:
        raise ValueError(f"unknown rule: {rule}")

    wasted = sum(1 for p in positions if p in side_set)
    return Selection(rule=rule, positions=positions, budget=budget, wasted=wasted)
