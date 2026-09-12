"""Run the full experiment loop on a toy model and fabricated paragraphs.

Checks the loop itself: every cell of the (side level x budget x rule) grid gets filled,
the s=0 invariant holds inside `run`, the empty-side-info path works, and the paired
statistics line up on the same questions.

Run:  python scripts/integration_test.py
"""

from __future__ import annotations

import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

from latent_comm.data import Example
from latent_comm.experiment import Config, run
from latent_comm.stats import comparison_table, interaction_test, to_frame
from smoke_test import build_toy_lm


def fake_examples(n: int = 4):
    out = []
    for q in range(n):
        paragraphs = [
            f"Title{q}_{p} . " + " ".join(f"w{q}_{p}_{i}" for i in range(28))
            for p in range(10)
        ]
        out.append(
            Example(
                qid=f"q{q}",
                question=f"who did thing {q} ?",
                answer=f"person {q}",
                paragraphs=paragraphs,
                gold_idx=[0, 5],
            )
        )
    return out


def main() -> int:
    lm = build_toy_lm()
    examples = fake_examples(4)
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(
            n_examples=len(examples),
            side_info_levels=(0, 2, 5),
            budget_fractions=(0.05, 0.2),
            save_tensors_for=1,
            out_dir=tmp,
        )
        rows = run(lm, examples, cfg)
        df = to_frame(rows)

        ok = True
        expected = len(examples) * 3 * 2 * 5
        ok &= _check("grid is complete", len(df) == expected, f"{len(df)} vs {expected}")
        ok &= _check("no NaN NLLs", bool(df.nll.notna().all()))
        ok &= _check(
            "budget honoured everywhere",
            bool((df.budget <= df.doc_len).all() and (df.budget > 0).all()),
        )

        # at s=0 the three surprisal rules must coincide -> identical NLL
        z = df[df.side_level == 0]
        for frac in cfg.budget_fractions:
            sub = z[z.budget_frac == frac]
            a = sub[sub.rule == "sender_surprisal"].set_index("qid").nll
            b = sub[sub.rule == "receiver_surprisal"].set_index("qid").nll
            c = sub[sub.rule == "dedup_sender_surprisal"].set_index("qid").nll
            ok &= _check(
                f"s=0 invariant holds at budget {frac}",
                bool((a - b).abs().max() < 1e-9 and (a - c).abs().max() < 1e-9),
            )

        ok &= _check(
            "side info grows the receiver's own context",
            bool(
                df[df.side_level == 5].n_side_positions.mean()
                > df[df.side_level == 2].n_side_positions.mean()
                > df[df.side_level == 0].n_side_positions.mean()
            ),
        )
        ok &= _check(
            "dedup rule never wastes budget",
            bool((df[df.rule == "dedup_sender_surprisal"].wasted_budget == 0).all()),
        )
        ok &= _check(
            "sender rules do waste budget once side info exists",
            bool(
                df[(df.rule == "sender_attention") & (df.side_level == 5)]
                .wasted_budget.sum()
                > 0
            ),
        )

        tbl = comparison_table(df, n_boot=200)
        ok &= _check("comparison table is non-empty", len(tbl) > 0)
        ok &= _check(
            "paired comparisons use every question",
            bool((tbl.n == len(examples)).all()),
            str(sorted(set(tbl.n))),
        )
        inter = interaction_test(df, n_boot=200)
        ok &= _check("interaction test runs", len(inter) == len(cfg.budget_fractions))

        ok &= _check("rows.json written", os.path.exists(os.path.join(tmp, "rows.json")))
        ok &= _check(
            "score tensors written",
            os.path.exists(os.path.join(tmp, "score_tensors.pt")),
        )

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


def _check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    return ok


if __name__ == "__main__":
    raise SystemExit(main())
