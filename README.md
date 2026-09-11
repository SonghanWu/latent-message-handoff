# What Should a Latent Message Preserve?

Submission for the DAPLab **Latent-comm** task (Compressed Latent Communication for
Language-Model Agents).

Every KV-cache compression method in production ranks tokens by a statistic computed
inside the sender — accumulated attention ([H2O](https://arxiv.org/abs/2306.14048)), the
attention a trailing window pays to the prefix
([SnapKV](https://arxiv.org/abs/2404.14469)), or position
([StreamingLLM](https://arxiv.org/abs/2309.17453)). That is the only question available
when a model talks to itself. An agent handoff has a receiver with its own context, and
classical distributed source coding says the sender's own statistics are then the wrong
ranking.

**Claim.** At a fixed communication budget, ranking cache entries by *receiver-conditioned
surprisal* beats ranking by sender salience or sender surprisal, and **the margin grows
with the side information the receiver holds**. The interaction, not the main effect, is
the load-bearing prediction — a main effect has many explanations, a margin that scales
with the receiver's context is what conditionality specifically predicts.

**Falsifiers.** Sender-side rules matching or beating it; the receiver-conditioned rule
collapsing to literal deduplication; the margin not growing with side information. All
three are measured, not asserted.

---

## Reproduce

```bash
git clone https://github.com/SonghanWu/latent-message-handoff.git
cd latent-message-handoff
pip install -r requirements.txt

# plumbing checks -- no download, no GPU, ~1 minute
python scripts/smoke_test.py
python scripts/integration_test.py

# the experiment -- ~20 min on a T4, downloads Qwen2.5-0.5B-Instruct and HotpotQA
python scripts/run_experiment.py --n-examples 100 --out results
```

Writes `results/rows.json`, `comparisons.csv`, `interaction.csv`, `main.png`,
`scores.png`, `reference_tensors.pt`, `score_tensors.pt`.

The same thing narrated end to end, with the figures inline, is
[`notebooks/latent_message.ipynb`](notebooks/latent_message.ipynb) — open it in Colab and
run all.

---

## Design

**Setup.** HotpotQA distractor gives each question two gold paragraphs and eight
distractors. A worker reads all ten and builds a KV cache. The receiver holds a chosen
number of **distractor** paragraphs — never a gold one — so the evidence it needs must
cross the handoff, while the overlap between the worker's cache and the receiver's
context is a knob we turn (0, 3, 6 paragraphs).

**Rules, all spending the same budget.**

| rule | ranks by | explanation it stands for |
|---|---|---|
| `random` | — | floor |
| `sender_attention` | accumulated attention over all layers and heads | salience (H2O) |
| `sender_surprisal` | `-log p(token \| question, doc_<t)` | information content, blind to the receiver |
| `dedup_sender_surprisal` | same, excluding what the receiver literally holds | surface redundancy |
| `receiver_surprisal` | `-log p(token \| side info, question, doc_<t)` | conditional novelty (the claim) |

Rules 3–5 use the *same scoring pass* with the *same task conditioning*; the only
difference is whether the receiver's side information is in the prefix. That makes 5 vs 3
an isolation of conditionality, and 5 vs 4 an isolation of *inferable* redundancy from
*literal* redundancy.

**Metric.** Mean token NLL of the gold answer read out of the handed-over cache.
Continuous and low-variance; exact-match on free-form answers from a 0.5B model mostly
measures formatting. `--compute-em` adds greedy EM/F1 as a secondary check.

**Statistics.** Paired bootstrap over questions. Between-question variance dwarfs the
effect, so unpaired comparisons would hide a real difference in noise.

### Three decisions worth arguing about

**Position handling.** Selected keys keep their original rotary phase; the question is
placed at `doc_len`, not at `len(cache)`. The receiver therefore sees a position sequence
with holes. The alternative — unrotate and renumber onto contiguous positions — keeps the
position distribution in-domain but destroys the original spacing between transmitted
spans. Both are defensible; this one is what the KV-compression literature does, and
`answer_nll` takes `question_start_pos` explicitly so the other is a one-line change.

**Budget accounting.** A position the receiver already holds still costs budget when a
rule selects it. That waste is the phenomenon under study, so charging for it is the
point; `wasted_budget` reports it per rule.

**Attention sinks.** Every rule is forced to keep the first four positions, charged
against its budget. Left to chance, sink survival — not the selection rule — would be the
variable under test.

### The honest limitation

Our receiver-conditioned rule reads the receiver's state directly, so it is an
**oracle-side upper bound, not a codec**. The substance of Wyner–Ziv is that the encoder
reaches the conditional rate *without seeing* the side information, via binning; we do
not implement that. This experiment measures whether the conditional *objective* is worth
pursuing, not whether it is reachable under the constraint that makes the theorem
interesting. The follow-up that matters: charge a receiver→sender digest to the same
budget and see whether the advantage survives paying for it.

---

## Layout

```
latent_comm/
  kvcache.py     cache format normalisation, position-wise slicing, byte accounting
  model_io.py    chunked prefill with attention accounting, surprisal, answer scoring
  data.py        HotpotQA distractor loading, paragraph token spans, side-info sampling
  selection.py   the five rules
  experiment.py  the loop
  stats.py       paired bootstrap, interaction test, figures
scripts/
  smoke_test.py         plumbing checks on a randomly initialised tiny Qwen2
  integration_test.py   the full loop on a toy model
  run_experiment.py     CLI
notebooks/
  latent_message.ipynb  the narrated version
report/
  report.md             the technical document
```

`smoke_test.py` is worth a look before trusting any number: it asserts that scoring an
answer through a full-cache handoff equals scoring it in one ordinary forward pass, that
chunked attention accumulation matches a full-length pass, and that `position_ids`
actually reaches RoPE.

## References

- A. D. Wyner and J. Ziv. The rate-distortion function for source coding with side
  information at the decoder. *IEEE Trans. Inform. Theory*, 22(1):1–10, 1976.
  [doi:10.1109/TIT.1976.1055508](https://doi.org/10.1109/TIT.1976.1055508)
- D. Slepian and J. K. Wolf. Noiseless coding of correlated information sources.
  *IEEE Trans. Inform. Theory*, 19(4):471–480, 1973.
  [doi:10.1109/TIT.1973.1055037](https://doi.org/10.1109/TIT.1973.1055037)
- Z. Zhang et al. H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large
  Language Models. [arXiv:2306.14048](https://arxiv.org/abs/2306.14048)
- Y. Li et al. SnapKV: LLM Knows What You are Looking for Before Generation.
  [arXiv:2404.14469](https://arxiv.org/abs/2404.14469)
- G. Xiao et al. Efficient Streaming Language Models with Attention Sinks.
  [arXiv:2309.17453](https://arxiv.org/abs/2309.17453)
