"""HotpotQA (distractor) as a controllable side-information testbed.

Why this dataset.  Each question ships with exactly ten paragraphs: two that are
needed to answer it ("gold") and eight distractors.  That gives us a knob no
synthetic setup gives for free -- we can hand the receiver a chosen number of
*distractor* paragraphs as its own context while guaranteeing that the information
it actually needs only exists in the worker's cache.  Growing that number grows the
overlap between the worker's state and what the receiver already has, which is
exactly the axis the Wyner-Ziv argument makes a prediction about.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

_DATASET_CANDIDATES = [
    ("hotpotqa/hotpot_qa", "distractor"),
    ("hotpot_qa", "distractor"),
]


@dataclass
class Example:
    qid: str
    question: str
    answer: str
    paragraphs: List[str]           # rendered "Title. sentences..." blocks
    gold_idx: List[int]             # indices into `paragraphs` that support the answer

    @property
    def distractor_idx(self) -> List[int]:
        gold = set(self.gold_idx)
        return [i for i in range(len(self.paragraphs)) if i not in gold]


@dataclass
class TokenizedExample:
    ex: Example
    doc_ids: torch.Tensor                     # [1, doc_len]
    para_spans: List[Tuple[int, int]]         # (start, end) token index per paragraph
    question_ids: torch.Tensor                # [1, q_len]
    answer_ids: torch.Tensor                  # [1, a_len]

    @property
    def doc_len(self) -> int:
        return int(self.doc_ids.shape[1])

    def positions_of(self, para_ids: Sequence[int]) -> List[int]:
        out: List[int] = []
        for p in para_ids:
            s, e = self.para_spans[p]
            out.extend(range(s, e))
        return out


def load_examples(
    n: int = 100,
    split: str = "validation",
    min_doc_chars: int = 2000,
    seed: int = 0,
) -> List[Example]:
    """Pull `n` distractor-setting examples.  Requires network on first call."""
    from datasets import load_dataset

    ds = None
    errors = []
    for name, config in _DATASET_CANDIDATES:
        try:
            ds = load_dataset(name, config, split=split)
            break
        except Exception as exc:  # noqa: BLE001 - we want the fallback chain
            errors.append(f"{name}/{config}: {exc}")
    if ds is None:
        raise RuntimeError("could not load HotpotQA:\n  " + "\n  ".join(errors))

    rng = random.Random(seed)
    order = list(range(len(ds)))
    rng.shuffle(order)

    out: List[Example] = []
    for i in order:
        row = ds[i]
        titles = row["context"]["title"]
        sentence_lists = row["context"]["sentences"]
        if len(titles) < 10:
            continue
        paragraphs = [
            f"{t}. " + " ".join(s.strip() for s in sents)
            for t, sents in zip(titles, sentence_lists)
        ]
        if sum(len(p) for p in paragraphs) < min_doc_chars:
            continue
        gold_titles = set(row["supporting_facts"]["title"])
        gold_idx = [j for j, t in enumerate(titles) if t in gold_titles]
        if not gold_idx or len(gold_idx) >= len(titles):
            continue
        out.append(
            Example(
                qid=row["id"],
                question=row["question"].strip(),
                answer=row["answer"].strip(),
                paragraphs=paragraphs,
                gold_idx=gold_idx,
            )
        )
        if len(out) >= n:
            break
    return out


def render_and_tokenize(lm, ex: Example) -> TokenizedExample:
    """Lay the paragraphs out as one document and record each paragraph's token span.

    Plain text, no chat template: the template's control tokens would sit inside the
    document and complicate the position bookkeeping for no benefit here.
    """
    pieces: List[str] = []
    for j, para in enumerate(ex.paragraphs):
        pieces.append(f"[{j + 1}] {para}\n\n")

    ids_per_piece = [lm.encode(p) for p in pieces]
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for ids in ids_per_piece:
        length = int(ids.shape[1])
        spans.append((cursor, cursor + length))
        cursor += length
    doc_ids = torch.cat(ids_per_piece, dim=1)

    question_ids = lm.encode(f"Question: {ex.question}\nAnswer:")
    answer_ids = lm.encode(f" {ex.answer}")

    return TokenizedExample(
        ex=ex,
        doc_ids=doc_ids,
        para_spans=spans,
        question_ids=question_ids,
        answer_ids=answer_ids,
    )


def choose_side_info(
    tex: TokenizedExample, n_paragraphs: int, seed: int
) -> List[int]:
    """Pick which paragraphs the receiver already holds.

    Drawn only from distractors, so the gold evidence is always missing from the
    receiver and must cross the handoff.  With `n_paragraphs=0` the receiver has
    nothing but the question -- the control condition in which the receiver-conditioned
    and sender-side scores are mathematically identical.
    """
    distractors = tex.ex.distractor_idx
    if n_paragraphs <= 0:
        return []
    rng = random.Random(f"{tex.ex.qid}:{seed}:{n_paragraphs}")
    n = min(n_paragraphs, len(distractors))
    return sorted(rng.sample(distractors, n))
