"""Worker prefill with attention accounting, token surprisal, and answer scoring.

All plain `model(...)` calls.  Two non-obvious parts: `prefill_with_attention` chunks
the document because full-length `output_attentions=True` costs ~3 GB in fp16 across
24 layers, and `answer_nll` passes explicit `position_ids` because the handed-over
cache is shorter than the positions it came from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .kvcache import cache_length, from_legacy, to_legacy


# --------------------------------------------------------------------------------------
@dataclass
class LM:
    model: torch.nn.Module
    tokenizer: object
    device: torch.device
    dtype: torch.dtype

    @property
    def config(self):
        return self.model.config

    def encode(self, text: str, add_special_tokens: bool = False) -> torch.Tensor:
        ids = self.tokenizer(
            text, return_tensors="pt", add_special_tokens=add_special_tokens
        ).input_ids
        return ids.to(self.device)


def load_lm(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    attn_implementation: str = "eager",
) -> LM:
    """Load the model.

    `attn_implementation="eager"` is required: SDPA and FlashAttention never
    materialise the attention matrix, so `output_attentions=True` returns None and the
    salience baseline cannot be computed.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if dtype is None:
        dtype = torch.float16 if device.type == "cuda" else torch.float32

    tok = AutoTokenizer.from_pretrained(model_name)
    # transformers >= 5 renamed torch_dtype -> dtype
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype, attn_implementation=attn_implementation
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, attn_implementation=attn_implementation
        )
    model.to(device)
    model.eval()
    return LM(model=model, tokenizer=tok, device=device, dtype=dtype)


# --------------------------------------------------------------------------------------
@torch.no_grad()
def prefill_with_attention(
    lm: LM, input_ids: torch.Tensor, chunk_size: int = 128
) -> Tuple[object, torch.Tensor]:
    """Run the worker's pass over the document.

    Returns the full cache plus [seq_len] accumulated attention per position, summed
    over every layer, head and query.  Note this statistic is dominated by the
    position-0 sink at prefill time -- see report, "A baseline that did not work".
    """
    seq_len = input_ids.shape[1]
    score = torch.zeros(seq_len, dtype=torch.float32, device=lm.device)
    cache = None
    pos = 0

    while pos < seq_len:
        end = min(pos + chunk_size, seq_len)
        chunk = input_ids[:, pos:end]
        position_ids = torch.arange(pos, end, device=lm.device).unsqueeze(0)
        attention_mask = torch.ones(1, end, dtype=torch.long, device=lm.device)

        out = lm.model(
            chunk,
            past_key_values=cache,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_attentions=True,
        )
        cache = out.past_key_values

        # out.attentions: tuple over layers of [B, H_q, chunk_len, end]
        for layer_attn in out.attentions:
            # sum over heads and over the queries in this chunk
            contrib = layer_attn[0].float().sum(dim=0).sum(dim=0)  # [end]
            score[:end] += contrib
        del out

        pos = end

    return cache, score


@torch.no_grad()
def token_surprisal(
    lm: LM,
    prefix_ids: torch.Tensor,
    target_ids: torch.Tensor,
    chunk_size: int = 512,
) -> torch.Tensor:
    """-log p(target_t | prefix, target_<t) for every token of `target`.

    Returned tensor is [len(target)] float32.  `prefix_ids` must be non-empty so that
    the first target token has something to condition on.
    """
    assert prefix_ids.shape[1] > 0, "need a non-empty prefix"
    full = torch.cat([prefix_ids, target_ids], dim=1)
    n_prefix = prefix_ids.shape[1]
    n_target = target_ids.shape[1]

    out = torch.empty(n_target, dtype=torch.float32, device=lm.device)
    cache = None
    pos = 0
    total = full.shape[1]

    while pos < total:
        end = min(pos + chunk_size, total)
        chunk = full[:, pos:end]
        position_ids = torch.arange(pos, end, device=lm.device).unsqueeze(0)
        attention_mask = torch.ones(1, end, dtype=torch.long, device=lm.device)
        res = lm.model(
            chunk,
            past_key_values=cache,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
        cache = res.past_key_values
        logits = res.logits[0].float()  # [chunk_len, vocab]

        # logits[i] predicts full[pos + i + 1]
        for i in range(chunk.shape[1]):
            predicted_index = pos + i + 1
            if predicted_index < n_prefix or predicted_index >= total:
                continue
            tgt = full[0, predicted_index]
            out[predicted_index - n_prefix] = -F.log_softmax(logits[i], dim=-1)[tgt]
        del res, logits
        pos = end

    return out


@torch.no_grad()
def answer_nll(
    lm: LM,
    cache,
    question_ids: torch.Tensor,
    answer_ids: torch.Tensor,
    question_start_pos: int,
) -> float:
    """Mean -log p(answer | handed-over cache, question).

    `question_start_pos` is in the *document's* coordinate system (normally `doc_len`),
    not `len(cache)`: surviving keys keep their original rotary phase, so the question
    belongs after the document, not after the shortened cache.
    """
    qa = torch.cat([question_ids, answer_ids], dim=1)
    n_q = question_ids.shape[1]
    n_a = answer_ids.shape[1]
    past_len = cache_length(cache)
    new_len = qa.shape[1]

    position_ids = torch.arange(
        question_start_pos, question_start_pos + new_len, device=lm.device
    ).unsqueeze(0)
    attention_mask = torch.ones(1, past_len + new_len, dtype=torch.long, device=lm.device)

    res = lm.model(
        qa,
        past_key_values=cache if past_len > 0 else None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    logits = res.logits[0].float()  # [new_len, vocab]

    # logits[i] predicts qa[i + 1]; the answer occupies qa[n_q : n_q + n_a]
    total = 0.0
    for j in range(n_a):
        predictor = n_q + j - 1  # index into qa of the token that predicts answer[j]
        tgt = qa[0, n_q + j]
        total += float(-F.log_softmax(logits[predictor], dim=-1)[tgt])
    return total / max(n_a, 1)


@torch.no_grad()
def greedy_answer(
    lm: LM,
    cache,
    question_ids: torch.Tensor,
    question_start_pos: int,
    max_new_tokens: int = 24,
) -> str:
    """Greedy decode from a handed-over cache, for the secondary EM/F1 metric."""
    from .kvcache import to_legacy, from_legacy

    work = from_legacy(tuple((k.clone(), v.clone()) for k, v in to_legacy(cache)))
    ids = question_ids
    pos = question_start_pos
    generated = []
    for _ in range(max_new_tokens):
        past_len = cache_length(work)
        new_len = ids.shape[1]
        position_ids = torch.arange(pos, pos + new_len, device=lm.device).unsqueeze(0)
        attention_mask = torch.ones(
            1, past_len + new_len, dtype=torch.long, device=lm.device
        )
        res = lm.model(
            ids,
            past_key_values=work if past_len > 0 else None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
        work = res.past_key_values
        nxt = int(res.logits[0, -1].argmax())
        if nxt == lm.tokenizer.eos_token_id:
            break
        generated.append(nxt)
        pos += new_len
        ids = torch.tensor([[nxt]], device=lm.device)
    return lm.tokenizer.decode(generated, skip_special_tokens=True)
