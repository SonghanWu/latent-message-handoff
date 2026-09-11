"""Command-line entry point.

    python scripts/run_experiment.py --n-examples 100 --out results

Writes results/rows.json, results/comparisons.csv, results/interaction.csv,
results/main.png and results/reference_tensors.pt.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from latent_comm.data import load_examples
from latent_comm.experiment import Config, run, save_reference_tensors
from latent_comm.model_io import load_lm
from latent_comm.stats import comparison_table, interaction_test, main_figure, to_frame


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--n-examples", type=int, default=100)
    p.add_argument("--side-levels", type=int, nargs="+", default=[0, 3, 6])
    p.add_argument("--budgets", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    p.add_argument("--pool-window", type=int, default=1)
    p.add_argument("--n-sink", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--compute-em", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    print(f"loading {args.model} ...")
    lm = load_lm(args.model, device=args.device)
    print(f"  device={lm.device} dtype={lm.dtype} layers={lm.config.num_hidden_layers} "
          f"kv_heads={lm.config.num_key_value_heads}")

    print(f"loading {args.n_examples} HotpotQA distractor examples ...")
    examples = load_examples(n=args.n_examples, seed=args.seed)
    print(f"  got {len(examples)}")

    cfg = Config(
        model_name=args.model,
        n_examples=len(examples),
        side_info_levels=tuple(args.side_levels),
        budget_fractions=tuple(args.budgets),
        n_sink=args.n_sink,
        pool_window=args.pool_window,
        seed=args.seed,
        compute_em=args.compute_em,
        out_dir=args.out,
    )

    rows = run(lm, examples, cfg)
    df = to_frame(rows)

    comparisons = comparison_table(df)
    interaction = interaction_test(df)
    comparisons.to_csv(os.path.join(args.out, "comparisons.csv"), index=False)
    interaction.to_csv(os.path.join(args.out, "interaction.csv"), index=False)
    main_figure(df, os.path.join(args.out, "main.png"))

    save_reference_tensors(
        lm, examples[0], cfg, os.path.join(args.out, "reference_tensors.pt")
    )

    print("\n=== mean answer NLL by rule and side-information level ===")
    print(df.pivot_table(index="rule", columns="side_level", values="nll").round(4))
    print("\n=== paired comparisons (negative = first rule better) ===")
    print(comparisons.round(4).to_string(index=False))
    print("\n=== does the advantage grow with side information? ===")
    print(interaction.round(4).to_string(index=False))
    print(f"\nwrote everything to {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
