"""End-to-end smoke test on a randomly initialised tiny Qwen2 -- no download, no GPU.

This does not test whether the *claim* is true; it tests that the plumbing is right,
which is the part that silently produces plausible-looking garbage:

  * a cache sliced by position keeps the right rows in every layer
  * an answer scored against a full-cache slice equals the same answer scored in one
    ordinary forward pass  (the handoff path is not secretly changing the model)
  * position_ids override actually reaches RoPE
  * with no side information, sender and receiver surprisal coincide exactly
  * budgets are honoured and wasted-budget accounting is right

Run:  python scripts/smoke_test.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from latent_comm.kvcache import cache_length, select_positions, to_legacy
from latent_comm.model_io import LM, answer_nll, prefill_with_attention, token_surprisal
from latent_comm.selection import select


class ToyTokenizer:
    """Whitespace tokenizer with a stable hash vocabulary -- enough for plumbing tests."""

    def __init__(self, vocab_size: int = 512):
        self.vocab_size = vocab_size
        self.eos_token_id = 0

    def _ids(self, text: str):
        toks = text.split() or ["<empty>"]
        return [1 + (abs(hash(t)) % (self.vocab_size - 1)) for t in toks]

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        ids = torch.tensor([self._ids(text)], dtype=torch.long)

        class Out:
            input_ids = ids

        return Out()

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(i)) for i in ids)


def build_toy_lm() -> LM:
    from transformers import AutoModelForCausalLM, Qwen2Config

    cfg = Qwen2Config(
        vocab_size=512,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=8,
        num_key_value_heads=2,      # GQA, like the real model
        max_position_embeddings=2048,
        rope_theta=1000000.0,
        tie_word_embeddings=True,
    )
    try:
        model = AutoModelForCausalLM.from_config(cfg, attn_implementation="eager")
    except TypeError:
        model = AutoModelForCausalLM.from_config(cfg)
        model.config._attn_implementation = "eager"
    model.eval()
    torch.manual_seed(0)
    return LM(model=model, tokenizer=ToyTokenizer(), device=torch.device("cpu"),
              dtype=torch.float32)


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    return ok


def main() -> int:
    torch.manual_seed(0)
    lm = build_toy_lm()
    ok = True

    doc = " ".join(f"tok{i}" for i in range(120))
    doc_ids = lm.encode(doc)
    doc_len = doc_ids.shape[1]
    question_ids = lm.encode("what is the answer ?")
    answer_ids = lm.encode("forty two")

    print("\n1. prefill and attention accounting")
    cache, attn = prefill_with_attention(lm, doc_ids, chunk_size=32)
    ok &= check("cache covers every position", cache_length(cache) == doc_len,
                f"{cache_length(cache)} vs {doc_len}")
    ok &= check("attention score is finite and non-negative",
                bool(torch.isfinite(attn).all() and (attn >= 0).all()))
    ok &= check("attention score has one entry per position", attn.numel() == doc_len)

    # chunked accumulation must equal a single full-length pass
    with torch.no_grad():
        ref = lm.model(doc_ids, use_cache=True, output_attentions=True,
                       position_ids=torch.arange(doc_len).unsqueeze(0))
    ref_score = torch.zeros(doc_len)
    for layer_attn in ref.attentions:
        ref_score += layer_attn[0].float().sum(dim=0).sum(dim=0)
    ok &= check("chunked attention == full-pass attention",
                torch.allclose(attn, ref_score, atol=1e-3),
                f"max diff {float((attn - ref_score).abs().max()):.2e}")

    print("\n2. cache slicing")
    keep = [0, 1, 2, 50, 51, 119]
    sliced = select_positions(cache, keep)
    ok &= check("sliced length == len(keep)", cache_length(sliced) == len(keep))
    full_legacy, slice_legacy = to_legacy(cache), to_legacy(sliced)
    same = all(
        torch.equal(slice_legacy[l][0][:, :, i, :], full_legacy[l][0][:, :, p, :])
        for l in range(len(full_legacy))
        for i, p in enumerate(keep)
    )
    ok &= check("sliced rows are the original rows, all layers", same)

    print("\n3. handoff path equals an ordinary forward pass")
    # if we keep everything, scoring through the cache must equal scoring in one pass
    nll_via_cache = answer_nll(
        lm, select_positions(cache, range(doc_len)), question_ids, answer_ids,
        question_start_pos=doc_len,
    )
    with torch.no_grad():
        full_seq = torch.cat([doc_ids, question_ids, answer_ids], dim=1)
        out = lm.model(full_seq, position_ids=torch.arange(full_seq.shape[1]).unsqueeze(0))
        logits = out.logits[0].float()
    n_a = answer_ids.shape[1]
    start = doc_len + question_ids.shape[1]
    direct = sum(
        float(-F.log_softmax(logits[start + j - 1], dim=-1)[full_seq[0, start + j]])
        for j in range(n_a)
    ) / n_a
    ok &= check("answer NLL through cache == direct", abs(nll_via_cache - direct) < 1e-3,
                f"{nll_via_cache:.6f} vs {direct:.6f}")

    print("\n4. position_ids really drive RoPE")
    a = answer_nll(lm, select_positions(cache, range(doc_len)), question_ids,
                   answer_ids, question_start_pos=doc_len)
    b = answer_nll(lm, select_positions(cache, range(doc_len)), question_ids,
                   answer_ids, question_start_pos=doc_len + 500)
    ok &= check("moving the question changes the score", abs(a - b) > 1e-6,
                f"{a:.6f} vs {b:.6f}")

    print("\n5. surprisal")
    prefix = lm.encode("Question: what ?")
    sup = token_surprisal(lm, prefix, doc_ids)
    ok &= check("one surprisal per document token", sup.numel() == doc_len)
    ok &= check("surprisal is finite and positive",
                bool(torch.isfinite(sup).all() and (sup > 0).all()))
    # chunk size must not change the values
    sup2 = token_surprisal(lm, prefix, doc_ids, chunk_size=37)
    ok &= check("surprisal independent of chunk size",
                torch.allclose(sup, sup2, atol=1e-4),
                f"max diff {float((sup - sup2).abs().max()):.2e}")

    print("\n6. selection rules")
    side = list(range(20, 40))
    budget = 30
    sels = {
        rule: select(rule, budget, doc_len, attn, sup, sup, side, n_sink=4, seed=0)
        for rule in ["random", "sender_attention", "sender_surprisal",
                     "dedup_sender_surprisal", "receiver_surprisal"]
    }
    ok &= check("every rule spends exactly the budget",
                all(len(s.positions) == budget for s in sels.values()),
                str({k: len(v.positions) for k, v in sels.items()}))
    ok &= check("every rule keeps the sinks",
                all(set(range(4)).issubset(s.positions) for s in sels.values()))
    ok &= check("dedup rule wastes nothing", sels["dedup_sender_surprisal"].wasted == 0)
    ok &= check("identical scores -> identical sets (the s=0 invariant)",
                sels["sender_surprisal"].positions == sels["receiver_surprisal"].positions)
    ok &= check("wasted-budget accounting matches",
                all(s.wasted == sum(1 for p in s.positions if p in set(side))
                    for s in sels.values()))

    print("\n7. budget monotonicity sanity")
    nlls = {}
    for frac in [0.05, 0.2, 0.5, 1.0]:
        b = max(5, int(frac * doc_len))
        s = select("sender_attention", b, doc_len, attn, sup, sup, [], n_sink=4)
        c = select_positions(cache, sorted(set(s.positions)))
        nlls[frac] = answer_nll(lm, c, question_ids, answer_ids, question_start_pos=doc_len)
    ok &= check("larger budget -> cache actually grows",
                cache_length(select_positions(cache, range(int(0.5 * doc_len))))
                > cache_length(select_positions(cache, range(int(0.05 * doc_len)))))
    print(f"       (untrained model, NLL by budget: "
          f"{ {k: round(v, 3) for k, v in nlls.items()} })")

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
