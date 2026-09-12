"""Paired statistics and the report's figures.

Every rule sees the same questions at the same budget, so comparisons are paired and
bootstrapped over questions -- between-question variance dwarfs the between-rule
effect, and an unpaired test would hide a real difference in noise.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    n_boot: int = 10000,
    seed: int = 0,
) -> Dict[str, float]:
    """Mean of (a - b) with a bootstrap CI over the pairing index.

    Negative mean means rule `a` gives lower NLL, i.e. `a` is better.
    """
    a_arr, b_arr = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a_arr.shape != b_arr.shape:
        raise ValueError(f"unpaired inputs: {a_arr.shape} vs {b_arr.shape}")
    diff = a_arr - b_arr
    n = diff.size
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = diff[idx].mean(axis=1)
    return dict(
        mean=float(diff.mean()),
        lo=float(np.percentile(boot, 2.5)),
        hi=float(np.percentile(boot, 97.5)),
        n=int(n),
        p_better=float((boot < 0).mean()),
    )


def to_frame(rows: List[dict]):
    import pandas as pd

    return pd.DataFrame(rows)


def pivot_paired(df, rule_a: str, rule_b: str, side_level: int, budget_frac: float):
    """Line up two rules on the same questions."""
    sub = df[(df.side_level == side_level) & (df.budget_frac == budget_frac)]
    a = sub[sub.rule == rule_a].set_index("qid").nll
    b = sub[sub.rule == rule_b].set_index("qid").nll
    common = a.index.intersection(b.index)
    return a.loc[common].to_numpy(), b.loc[common].to_numpy()


def comparison_table(
    df,
    pairs: Sequence[Tuple[str, str]] = (
        ("receiver_surprisal", "sender_attention"),
        ("receiver_surprisal", "sender_surprisal"),
        ("receiver_surprisal", "dedup_sender_surprisal"),
        ("dedup_sender_surprisal", "sender_surprisal"),
    ),
    n_boot: int = 10000,
):
    """The table the claim lives or dies by."""
    import pandas as pd

    out = []
    for side_level in sorted(df.side_level.unique()):
        for frac in sorted(df.budget_frac.unique()):
            for rule_a, rule_b in pairs:
                a, b = pivot_paired(df, rule_a, rule_b, side_level, frac)
                if a.size == 0:
                    continue
                res = paired_bootstrap(a, b, n_boot=n_boot)
                out.append(
                    dict(
                        side_level=side_level,
                        budget_frac=frac,
                        comparison=f"{rule_a} - {rule_b}",
                        delta_nll=res["mean"],
                        ci_lo=res["lo"],
                        ci_hi=res["hi"],
                        n=res["n"],
                        significant=bool(res["hi"] < 0 or res["lo"] > 0),
                    )
                )
    return pd.DataFrame(out)


def interaction_test(
    df,
    rule_a: str = "receiver_surprisal",
    rule_b: str = "dedup_sender_surprisal",
    budget_frac: Optional[float] = None,
    n_boot: int = 10000,
    seed: int = 0,
):
    """Does the advantage of `rule_a` grow with the receiver's side information?

    Bootstrapped difference-of-differences between the largest and smallest
    side-information level.  Pass rule_b="sender_surprisal" for the matched comparison.
    """
    import pandas as pd

    fracs = sorted(df.budget_frac.unique()) if budget_frac is None else [budget_frac]
    levels = sorted(df.side_level.unique())
    lo_level, hi_level = levels[0], levels[-1]
    rng = np.random.default_rng(seed)
    out = []

    for frac in fracs:
        a_lo, b_lo = pivot_paired(df, rule_a, rule_b, lo_level, frac)
        a_hi, b_hi = pivot_paired(df, rule_a, rule_b, hi_level, frac)
        n = min(a_lo.size, a_hi.size)
        if n == 0:
            continue
        d_lo, d_hi = (a_lo - b_lo)[:n], (a_hi - b_hi)[:n]
        dd = d_hi - d_lo
        idx = rng.integers(0, n, size=(n_boot, n))
        boot = dd[idx].mean(axis=1)
        out.append(
            dict(
                budget_frac=frac,
                side_low=lo_level,
                side_high=hi_level,
                delta_at_low=float(d_lo.mean()),
                delta_at_high=float(d_hi.mean()),
                difference_of_differences=float(dd.mean()),
                ci_lo=float(np.percentile(boot, 2.5)),
                ci_hi=float(np.percentile(boot, 97.5)),
                n=int(n),
            )
        )
    return pd.DataFrame(out)


def main_figure(df, out_path: str = "results/main.png"):
    """One figure: answer NLL vs budget, one panel per side-information level."""
    import matplotlib.pyplot as plt

    levels = sorted(df.side_level.unique())
    rules = [
        ("random", "#9aa5a8", "-"),
        ("sender_attention", "#b06a34", "-"),
        ("sender_surprisal", "#6b5ea8", "-"),
        ("dedup_sender_surprisal", "#3f7fa8", "--"),
        ("receiver_surprisal", "#0d7a72", "-"),
    ]

    fig, axes = plt.subplots(
        1, len(levels), figsize=(4.2 * len(levels), 3.6), sharey=True
    )
    if len(levels) == 1:
        axes = [axes]

    for ax, level in zip(axes, levels):
        sub = df[df.side_level == level]
        for rule, color, style in rules:
            r = sub[sub.rule == rule].groupby("budget_frac").nll
            if len(r) == 0:
                continue
            mean, sem = r.mean(), r.sem()
            ax.errorbar(
                mean.index * 100,
                mean.to_numpy(),
                yerr=sem.to_numpy(),
                label=rule,
                color=color,
                linestyle=style,
                marker="o",
                markersize=4,
                capsize=2,
                linewidth=1.6,
            )
        full = sub.nll_full_cache.mean()
        none = sub.nll_side_only.mean()
        ax.axhline(full, color="#444", linewidth=0.9, linestyle=":")
        ax.axhline(none, color="#444", linewidth=0.9, linestyle=":")
        ax.text(
            ax.get_xlim()[1], full, " full cache", va="center", fontsize=7, color="#444"
        )
        ax.text(
            ax.get_xlim()[1], none, " no handoff", va="center", fontsize=7, color="#444"
        )
        ax.set_title(f"receiver holds {level} distractor paragraphs", fontsize=10)
        ax.set_xlabel("communication budget (% of document tokens)")
        ax.grid(alpha=0.25, linewidth=0.6)

    axes[0].set_ylabel("answer NLL (lower is better)")
    axes[-1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    return fig


def plot_scores(
    tex,
    attn_score,
    sender_surprisal,
    receiver_surprisal,
    side_paragraphs,
    out_path: str = "results/scores.png",
    smooth: int = 15,
):
    """Show the three ranking signals over one document.

    Over the paragraphs the receiver holds, receiver-conditioned surprisal collapses
    while the sender-side signals carry on unchanged.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    def prep(x):
        v = np.asarray(x.detach().cpu(), dtype=float)
        if smooth > 1:
            kernel = np.ones(smooth) / smooth
            v = np.convolve(v, kernel, mode="same")
        lo, hi = v.min(), v.max()
        return (v - lo) / (hi - lo + 1e-9)

    fig, ax = plt.subplots(figsize=(11, 3.4))
    side = set(side_paragraphs)
    gold = set(tex.ex.gold_idx)

    for p, (s, e) in enumerate(tex.para_spans):
        if p in side:
            ax.axvspan(s, e, color="#c9d2d1", alpha=0.55, linewidth=0)
        if p in gold:
            ax.axvspan(s, e, color="#0d7a72", alpha=0.16, linewidth=0)

    ax.plot(prep(attn_score), color="#b06a34", linewidth=1.2, label="sender attention (H2O)")
    ax.plot(prep(sender_surprisal), color="#6b5ea8", linewidth=1.2, label="sender surprisal")
    ax.plot(prep(receiver_surprisal), color="#0d7a72", linewidth=1.6,
            label="receiver-conditioned surprisal")

    ax.set_xlabel("document token position")
    ax.set_ylabel("normalised score")
    ax.set_xlim(0, tex.doc_len)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper center")
    ax.set_title(
        "grey = paragraphs the receiver already holds   |   teal = gold evidence",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    return fig
