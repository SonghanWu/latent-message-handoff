"""KV-cache plumbing: extract a worker cache, slice it by position, reuse it as a
receiver's past.

The whole experiment rests on one operation -- take a subset of the positions in a
cache and hand that subset to a second forward pass -- so this module is deliberately
small and explicit about the two things that are easy to get wrong:

1.  RoPE.  Keys are written into the cache *after* the rotary embedding is applied,
    so every key carries the absolute position it was computed at.  We keep those
    original position ids (we never re-rotate), which means the receiver sees a
    position sequence with holes in it.  That is the standard choice in the KV
    compression literature and it is the one we document in the report.

2.  Cache format.  transformers has moved from tuple-of-tuples to `DynamicCache`.
    We normalise to legacy tuples internally and convert back at the boundary.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch

LegacyCache = Tuple[Tuple[torch.Tensor, torch.Tensor], ...]


# --------------------------------------------------------------------------------------
# format normalisation
# --------------------------------------------------------------------------------------
def to_legacy(cache) -> LegacyCache:
    """Return a tuple of (key, value) pairs, one per layer, each [B, H_kv, S, D]."""
    if cache is None:
        raise ValueError("cache is None -- did you forget use_cache=True?")
    if isinstance(cache, tuple):
        return cache
    if hasattr(cache, "to_legacy_cache"):
        return cache.to_legacy_cache()
    # transformers >= 4.56 exposes .layers[i].keys / .values
    if hasattr(cache, "layers"):
        return tuple((layer.keys, layer.values) for layer in cache.layers)
    raise TypeError(f"unrecognised cache type: {type(cache)}")


def from_legacy(legacy: LegacyCache):
    """Rebuild whatever cache object the installed transformers expects."""
    try:
        from transformers import DynamicCache
    except ImportError:  # very old transformers
        return legacy
    if hasattr(DynamicCache, "from_legacy_cache"):
        return DynamicCache.from_legacy_cache(legacy)
    cache = DynamicCache()
    for layer_idx, (k, v) in enumerate(legacy):
        cache.update(k, v, layer_idx)
    return cache


def cache_length(cache) -> int:
    return to_legacy(cache)[0][0].shape[2]


def num_layers(cache) -> int:
    return len(to_legacy(cache))


# --------------------------------------------------------------------------------------
# the one operation that matters
# --------------------------------------------------------------------------------------
def select_positions(cache, positions: Sequence[int], device=None):
    """Keep only `positions` (indices into the sequence axis) in every layer.

    Positions are kept in ascending order.  We do *not* renumber them: the keys keep
    the rotary phase they were computed with, so the receiver's attention sees the
    original relative distances (with gaps).  See report section "Position handling".
    """
    legacy = to_legacy(cache)
    idx = torch.as_tensor(sorted(int(p) for p in positions), dtype=torch.long)
    if idx.numel() == 0:
        # an empty past: keep the layer structure, zero length on the sequence axis
        out = tuple(
            (k[:, :, :0, :].clone(), v[:, :, :0, :].clone()) for k, v in legacy
        )
        return from_legacy(out)
    out = []
    for k, v in legacy:
        i = idx.to(k.device)
        out.append((k.index_select(2, i).clone(), v.index_select(2, i).clone()))
    return from_legacy(tuple(out))


def slice_layers(cache, layer_ids: Sequence[int]) -> LegacyCache:
    """Pull a few layers out for saving to disk (the submission asks for tensors)."""
    legacy = to_legacy(cache)
    return tuple((legacy[i][0].cpu(), legacy[i][1].cpu()) for i in layer_ids)


def cache_bytes(cache) -> int:
    """Exact byte count of a cache -- the communication cost we are budgeting."""
    total = 0
    for k, v in to_legacy(cache):
        total += k.numel() * k.element_size() + v.numel() * v.element_size()
    return total


def scalars_per_token(cache) -> int:
    """Number of scalars one token of this cache occupies (2 * L * H_kv * d_head)."""
    legacy = to_legacy(cache)
    _, h, _, d = legacy[0][0].shape
    return 2 * len(legacy) * h * d
