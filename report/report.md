# What Should a Latent Message Preserve?

**Songhan (Mason) Wu** · DAPLab Latent-comm task
Code: <https://github.com/SonghanWu/latent-message-handoff>

---

## 1. The cross-field idea

**Source.** A. D. Wyner and J. Ziv, "The rate-distortion function for source coding with
side information at the decoder," *IEEE Transactions on Information Theory* 22(1):1–10,
1976 ([doi:10.1109/TIT.1976.1055508](https://doi.org/10.1109/TIT.1976.1055508)), which
extends the lossless result of D. Slepian and J. K. Wolf, *IEEE Trans. Inform. Theory*
19(4):471–480, 1973 ([doi:10.1109/TIT.1973.1055037](https://doi.org/10.1109/TIT.1973.1055037)).

Slepian–Wolf establishes that when a decoder holds correlated side information *Y*, a
source *X* can be compressed to *H(X | Y)* bits **even though the encoder never observes
Y**. Wyner–Ziv carries this into the lossy regime, characterising the rate–distortion
function when *Y* is available only at the decoder.

Two consequences reframe latent communication, and they are the only two this submission
uses:

1. **Value is conditional, not marginal.** What a piece of *X* is worth transmitting
   depends on the uncertainty that remains about it *after* conditioning on *Y*. Content
   the decoder can already reconstruct is worth nothing at any rate.
2. **The distortion measure is a free choice.** Nothing in the theory requires
   *d(x, x̂)* to be reconstruction error. Reconstruction is one admissible distortion;
   a task-defined distortion is equally admissible.

Consequence 2 is why this is not a restatement of "avoid duplicates". Existing KV-cache
compression optimises faithfulness to the sender's own attention output — H2O ranks by
accumulated attention, SnapKV by the attention a trailing observation window pays to the
prefix, StreamingLLM by position. Every one of those signals is computed *inside the
sender*. That is not an oversight: in single-model inference there is no receiver to
condition on. An agent handoff introduces one, and the theory says the ranking should
change.

## 2. The mapping, and where it stops

| Wyner–Ziv | latent handoff | status |
|---|---|---|
| source *X* | worker's KV cache over the document | exact |
| side information *Y* at the decoder | receiver's own context + model priors | context exact; priors informal |
| message, rate *R* | selected cache entries; scalars shipped | exact, and counted |
| distortion *d* | receiver's NLL of the gold answer | a valid distortion, not the MSE the closed forms assume |
| encoder blind to *Y* | H2O / SnapKV ranking blind to the receiver | exact |

**Where it is analogy only.** A transformer cache is not an i.i.d. source: no block
length, no asymptotics, hence no achievable-rate claim to inherit — only the *direction*
of the prediction survives. More seriously, we perform **selection, not binning**.
Wyner–Ziv's remarkable content is that the encoder attains the conditional rate *without
seeing Y*, through a coset construction we do not implement; our rule reads the
receiver's state directly. It is therefore an **oracle-side upper bound**, and §7 says
what would close the gap. Finally, "model priors as side information" is a gesture: we
approximate it by conditioning a forward pass on the receiver's context, which folds
priors and context together and cannot separate them.

## 3. The falsifiable claim

> At a fixed communication budget, ranking cache entries by **receiver-conditioned
> surprisal** yields lower answer NLL than ranking by sender salience or by sender-side
> surprisal, and **the margin grows with the amount of side information the receiver
> holds**.

The interaction is the load-bearing half. A main effect admits many explanations; a
margin that *scales with the receiver's context* is what conditionality specifically
predicts, and it is what distinguishes this account from the alternatives:

| competing explanation | its prediction | ours differs by |
|---|---|---|
| salience — the sender's attention identifies what matters | sender attention wins regardless of receiver | receiver dependence |
| information content — rare/surprising content matters | unconditional surprisal wins regardless of receiver | conditioning, not surprisal per se |
| surface redundancy — just drop what the receiver literally has | literal dedup captures the whole effect | inferable ≠ verbatim redundancy |

**What would falsify it.** (a) Sender-side rules match or beat the receiver-conditioned
rule. (b) The receiver-conditioned rule is indistinguishable from literal deduplication —
then the cross-field connection contributed nothing beyond a triviality. (c) The margin
does not grow with side information — then conditionality is not what is doing the work,
whatever the main effects show.

## 4. The smallest experiment that tests it

**Testbed.** HotpotQA (distractor): every question ships with two gold paragraphs and
eight distractors. A worker reads all ten as one document and builds a KV cache. The
receiver holds *s* ∈ {0, 3, 6} **distractor** paragraphs — never a gold one — so the
evidence it needs always has to cross the handoff while the *overlap* between the
worker's cache and the receiver's context is a free parameter. That is the axis the claim
makes a prediction about, and HotpotQA hands it over without any synthetic construction.

**Five rules, one budget** (5 %, 10 %, 20 % of document tokens): `random`;
`sender_attention` (accumulated attention, all layers and heads — the H2O statistic);
`sender_surprisal` (−log p(token | question, doc\_<t)); `dedup_sender_surprisal` (the
same, minus positions the receiver literally holds); `receiver_surprisal` (−log p(token |
side info, question, doc\_<t)).

Rules 3–5 share one scoring pass and one task conditioning; the *only* difference is
whether the receiver's side information sits in the prefix. So **5 vs 3** isolates
conditionality and **5 vs 4** isolates inferable from verbatim redundancy.

**Metric.** Mean token NLL of the gold answer, read out of the handed-over cache.
Continuous and low-variance; exact-match on free-form answers from a 0.5B model largely
measures formatting. Reference lines: the full cache, and side-information-only.

**Statistics.** Paired bootstrap over questions (between-question variance dwarfs the
effect); the interaction is a bootstrapped difference-of-differences between *s* = 6 and
*s* = 0.

**Three choices that could otherwise confound the comparison.**

* *Position handling.* Selected keys keep their original rotary phase and the question is
  placed at `doc_len`, so the receiver sees a position sequence with holes. The
  alternative (unrotate, renumber contiguously) keeps positions in-domain but destroys
  the original spacing between transmitted spans. Both defensible; this is what the
  KV-compression literature does.
* *Budget accounting.* A position the receiver already holds still costs budget when a
  rule picks it — that waste is the phenomenon, so it is charged, and reported per rule.
* *Attention sinks.* All rules are forced to keep the first four positions, charged to
  budget, so sink survival is not the hidden variable.

**Built-in invariant.** At *s* = 0 rules 3, 4 and 5 score with an identical context and
must select identical sets. The code asserts it; a failure means the scoring passes are
misaligned and no number in the run can be trusted.

## 5. What was implemented

Only what the comparison needs: cache slicing by position, three scoring signals, five
selection rules, an answer-NLL read-out, and a paired bootstrap. No compression system,
no learned projector, no benchmark chase.

Correctness is checked before any result is believed (`scripts/smoke_test.py`, on a
randomly initialised tiny Qwen2, no download): scoring an answer through a full-cache
handoff equals scoring it in one ordinary forward pass to 1e-3; chunked attention
accumulation matches a full-length pass; `position_ids` demonstrably reaches RoPE;
sliced caches contain exactly the original rows in every layer; budgets and
wasted-budget accounting are exact. `scripts/integration_test.py` runs the whole loop on
a toy model and asserts the *s* = 0 invariant end to end.

## 6. Result

100 questions, Qwen2.5-0.5B-Instruct, documents averaging 1391 tokens. Figure:
`results/main.png`. Full tables: `results/comparisons.csv`, `results/rows.json`.

**Mean answer NLL** (lower is better), 10 % budget:

| rule | s = 0 | s = 3 | s = 6 |
|---|---|---|---|
| no handoff (reference) | 3.739 | 8.304 | 6.215 |
| random | 3.417 | 3.940 | 4.110 |
| sender_attention | 3.902 | 4.157 | 4.154 |
| sender_surprisal | 3.024 | 3.348 | 3.610 |
| dedup_sender_surprisal | 3.024 | 3.107 | 2.803 |
| receiver_surprisal | 3.024 | 3.024 | 2.918 |
| full cache (reference) | 2.386 | 2.386 | 2.386 |

**Paired differences** (negative = first rule better; bootstrap CI over the 100 questions):

| comparison | s = 0 | s = 3 | s = 6 |
|---|---|---|---|
| receiver − sender_surprisal | 0.000 | **−0.324** [−0.492, −0.174] | **−0.692** [−0.990, −0.415] |
| receiver − dedup_sender | 0.000 | −0.083 [−0.201, +0.015] | +0.115 [−0.025, +0.286] |
| dedup_sender − sender_surprisal | 0.000 | **−0.242** [−0.399, −0.098] | **−0.807** [−1.102, −0.528] |

Three things this says.

**The invariant holds exactly.** At s = 0 the three surprisal rules score with an
identical context, and their answer NLLs agree to every printed digit (3.024 / 3.024 /
3.024; paired difference 0.000, not merely small). The scoring channels are aligned;
nothing downstream is comparing misaligned signals.

**Conditioning works, and its advantage grows with side information.** Against the
matched sender-side baseline the margin is 0.000 → −0.324 → −0.692 as the receiver goes
from 0 to 3 to 6 held paragraphs. Because the s = 0 difference is identically zero per
question, the difference-of-differences equals the s = 6 column: −0.598 [−0.839, −0.374],
−0.692 [−0.990, −0.415], −0.615 [−0.904, −0.328] at the 5 / 10 / 20 % budgets — the
predicted interaction, significant at every budget.

The mechanism is visible in the budget accounting. At s = 6 and a 20 % budget,
`sender_surprisal` spends 182 of its 278 tokens (**65 %**) on positions the receiver
already holds; `receiver_surprisal` spends 8 (**2.8 %**). And `results/scores.png` shows
why: over the paragraphs the receiver holds, receiver-conditioned surprisal collapses to
the floor while the sender-side signal carries on unchanged.

In absolute terms, at s = 6 and a 20 % budget `receiver_surprisal` closes **96.5 %** of
the distance between no handoff and the full cache (2.520 vs 6.215 and 2.386), against
80.5 % for `sender_surprisal`.

**The sharper claim fails.** `receiver_surprisal` and `dedup_sender_surprisal` are
statistically indistinguishable in all six non-trivial cells; every CI straddles zero.
Everything conditioning buys here is already bought by dropping what the receiver holds
verbatim. This is falsifier (b) from §3, declared before the run, and it triggered.

## 7. Interpretation

**What is supported.** The directional prediction: at a fixed budget, ranking by
receiver-conditioned novelty beats ranking by sender-side novelty, and the margin scales
with the receiver's side information. The effect is not subtle — 0.69 nats at s = 6 — and
the interaction is significant at every budget.

**What is not.** The interesting half of the Wyner–Ziv reading — that the relevant
redundancy is what the decoder can *infer*, not only what it literally holds — is not
supported by this experiment. The reason is a limitation of the testbed rather than
evidence against the idea: HotpotQA hands the receiver its paragraphs **verbatim**, so
essentially all redundancy here *is* literal, and the design has no power to separate the
two hypotheses. On this testbed, Wyner–Ziv earns exactly one thing: know what the receiver
has and skip it. That is worth stating plainly, because it is less than the argument in §1
promised.

**A baseline that did not work, and why it matters.** `sender_attention` performs at or
below `random` (59.8 % vs 59.2 % of the gap closed at s = 6, 20 %). `results/scores.png`
shows the cause: accumulated attention is dominated by the attention sink at position 0,
and after normalisation every other position is flattened to near zero, so the rule
degenerates into "keep the beginning". This is a property of our summed-over-everything
implementation in a prefill setting, **not a fair reproduction of H2O**, which also keeps a
recent window and accumulates during decoding. The `receiver − sender_attention` column
should therefore not be read as beating H2O; the load-bearing comparison is against
`sender_surprisal`, which is matched to the proposed rule in everything but conditioning.
Excluding sink positions before ranking, or scoring with a SnapKV-style observation
window, is the fix.

**One more alternative explanation.** The receiver-conditioned score comes from a forward
pass with a longer prefix, so it inherits a prefix-*length* effect alongside the
prefix-*content* effect. The control that separates them: pad the prefix to the same
length with paragraphs drawn from *other* questions. If the advantage survives, content is
doing the work.

**An oddity worth noting.** The no-handoff reference is *worse* with side information than
without (8.30 at s = 3 and 6.22 at s = 6, against 3.74 at s = 0): handing the model
distractor paragraphs and no evidence is worse than handing it nothing. It does not affect
any within-cell comparison, but it is a reminder that "more receiver context" is not
monotonically good, and that the handoff is doing more than topping up a partial answer.

**What is still an oracle.** Our rule reads the receiver's state directly. The substance
of Wyner–Ziv is reaching the conditional rate *without* observing the side information, via
binning, which we do not implement. The measured margin is an upper bound on what a
deployable scheme could reach.

**Follow-up, in priority order.**

1. *Paraphrased side information.* Give the receiver rewritten paragraphs — same facts,
   different wording. Literal deduplication stops working by construction, so only semantic
   conditioning can find the redundancy. This is the experiment that would actually test
   the claim that failed above, and the present result is what makes it the obvious next
   step.
2. *A fair salience baseline.* Re-run `sender_attention` with sink positions excluded, so
   the comparison against the deployed family means something.
3. *Charge the receiver's digest to the budget.* Let the receiver send k scalars
   summarising its state, rank at the sender against that digest, and count k against the
   communication budget. That is the step from oracle to codec.
