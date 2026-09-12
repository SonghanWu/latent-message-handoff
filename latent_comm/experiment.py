"""The experiment loop: one worker prefill per question, one scoring pass per
side-information level, then one short answer-scoring pass per (rule x budget) cell.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

import torch

from . import data as data_mod
from . import selection as sel_mod
from .kvcache import cache_bytes, scalars_per_token, select_positions, slice_layers
from .model_io import LM, answer_nll, greedy_answer, prefill_with_attention, token_surprisal


@dataclass
class Config:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    n_examples: int = 100
    side_info_levels: Sequence[int] = (0, 3, 6)      # distractor paragraphs the receiver holds
    budget_fractions: Sequence[float] = (0.05, 0.10, 0.20)
    rules: Sequence[str] = tuple(sel_mod.RULES)
    n_sink: int = 4
    pool_window: int = 1
    seed: int = 0
    save_tensors_for: int = 3
    compute_em: bool = False                          # greedy decode is the slow part
    out_dir: str = "results"


def render_side_info(ex: data_mod.Example, para_ids: Sequence[int]) -> str:
    if not para_ids:
        return ""
    return "".join(f"[{j + 1}] {ex.paragraphs[j]}\n\n" for j in sorted(para_ids))


_render_side_info = render_side_info  # backwards-compatible alias


def _normalise(text: str) -> str:
    import re
    import string

    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _f1(pred: str, gold: str) -> float:
    p, g = _normalise(pred).split(), _normalise(gold).split()
    if not p or not g:
        return float(p == g)
    common: Dict[str, int] = {}
    for t in p:
        if t in g:
            common[t] = min(p.count(t), g.count(t))
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision, recall = n_same / len(p), n_same / len(g)
    return 2 * precision * recall / (precision + recall)


def run(lm: LM, examples: Sequence[data_mod.Example], cfg: Config) -> List[dict]:
    os.makedirs(cfg.out_dir, exist_ok=True)
    rows: List[dict] = []
    saved_tensors: Dict[str, dict] = {}

    for ex_i, ex in enumerate(examples):
        tex = data_mod.render_and_tokenize(lm, ex)
        doc_len = tex.doc_len

        cache_full, attn_score = prefill_with_attention(lm, tex.doc_ids)
        per_token_scalars = scalars_per_token(cache_full)

        # sender-side score: task-conditioned, blind to the receiver
        sender_prefix = lm.encode(f"Question: {ex.question}\n\n")
        sender_sup = token_surprisal(lm, sender_prefix, tex.doc_ids)

        # reference points, identical for every rule
        nll_full = answer_nll(
            lm,
            select_positions(cache_full, range(doc_len)),
            tex.question_ids,
            tex.answer_ids,
            question_start_pos=doc_len,
        )

        for s_level in cfg.side_info_levels:
            side_paras = data_mod.choose_side_info(tex, s_level, cfg.seed)
            side_positions = tex.positions_of(side_paras)

            receiver_prefix = lm.encode(
                render_side_info(ex, side_paras) + f"Question: {ex.question}\n\n"
            )
            receiver_sup = token_surprisal(lm, receiver_prefix, tex.doc_ids)

            if s_level == 0:
                # invariant: with no side information the two scores are the same pass
                # fp16 on GPU is not bit-reproducible across two identical passes,
                # so allow a little slack -- a genuine misalignment of the two scoring
                # contexts moves this by orders of magnitude more than 2e-2.
                gap = float((sender_sup - receiver_sup).abs().max())
                assert gap < 2e-2, (
                    f"sender and receiver surprisal must coincide at s=0 (max gap {gap:.4f}); "
                    "they differ, which means the two scoring contexts are not aligned"
                )

            if ex_i < cfg.save_tensors_for:
                saved_tensors[f"{ex.qid}|s{s_level}|receiver_surprisal"] = (
                    receiver_sup.cpu()
                )
                saved_tensors[f"{ex.qid}|s{s_level}|side_positions"] = list(
                    side_positions
                )

            nll_side_only = answer_nll(
                lm,
                select_positions(cache_full, side_positions),
                tex.question_ids,
                tex.answer_ids,
                question_start_pos=doc_len,
            )

            for frac in cfg.budget_fractions:
                budget = max(cfg.n_sink + 1, int(round(frac * doc_len)))
                for rule in cfg.rules:
                    sel = sel_mod.select(
                        rule=rule,
                        budget=budget,
                        doc_len=doc_len,
                        attn_score=attn_score,
                        sender_surprisal=sender_sup,
                        receiver_surprisal=receiver_sup,
                        side_positions=side_positions,
                        n_sink=cfg.n_sink,
                        pool_window=cfg.pool_window,
                        seed=cfg.seed + ex_i,
                    )
                    final_positions = sorted(set(sel.positions) | set(side_positions))
                    cache = select_positions(cache_full, final_positions)
                    nll = answer_nll(
                        lm,
                        cache,
                        tex.question_ids,
                        tex.answer_ids,
                        question_start_pos=doc_len,
                    )

                    row = dict(
                        qid=ex.qid,
                        example_index=ex_i,
                        doc_len=doc_len,
                        side_level=s_level,
                        n_side_positions=len(side_positions),
                        budget_frac=frac,
                        budget=budget,
                        rule=rule,
                        nll=nll,
                        nll_full_cache=nll_full,
                        nll_side_only=nll_side_only,
                        wasted_budget=sel.wasted,
                        transmitted_scalars=budget * per_token_scalars,
                        cache_bytes_after_merge=cache_bytes(cache),
                    )
                    if cfg.compute_em:
                        pred = greedy_answer(
                            lm, cache, tex.question_ids, question_start_pos=doc_len
                        )
                        row["prediction"] = pred
                        row["em"] = float(_normalise(pred) == _normalise(ex.answer))
                        row["f1"] = _f1(pred, ex.answer)
                    rows.append(row)

                    if ex_i < cfg.save_tensors_for:
                        key = f"{ex.qid}|s{s_level}|b{frac}|{rule}"
                        saved_tensors[key] = dict(positions=sel.positions)

        if ex_i < cfg.save_tensors_for:
            saved_tensors[f"{ex.qid}|meta"] = dict(
                doc_len=doc_len,
                attn_score=attn_score.cpu(),
                sender_surprisal=sender_sup.cpu(),
                para_spans=tex.para_spans,
                gold_idx=ex.gold_idx,
                question=ex.question,
                answer=ex.answer,
            )
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    with open(os.path.join(cfg.out_dir, "rows.json"), "w") as fh:
        json.dump(rows, fh, indent=2)
    torch.save(saved_tensors, os.path.join(cfg.out_dir, "score_tensors.pt"))
    return rows


def save_reference_tensors(lm: LM, ex: data_mod.Example, cfg: Config, path: str) -> None:
    """Dump a worker KV cache (first / middle / last layer) for the submission.

    The task asks for "a collection of saved attention or hidden-state tensors"; this
    writes the actual object the whole argument is about, at a size that fits in a repo.
    """
    tex = data_mod.render_and_tokenize(lm, ex)
    cache, attn = prefill_with_attention(lm, tex.doc_ids)
    from .kvcache import num_layers

    n = num_layers(cache)
    layer_ids = [0, n // 2, n - 1]
    payload = dict(
        model=cfg.model_name,
        qid=ex.qid,
        question=ex.question,
        answer=ex.answer,
        doc_ids=tex.doc_ids.cpu(),
        para_spans=tex.para_spans,
        gold_idx=ex.gold_idx,
        layer_ids=layer_ids,
        kv=slice_layers(cache, layer_ids),
        accumulated_attention=attn.cpu(),
    )
    torch.save(payload, path)
