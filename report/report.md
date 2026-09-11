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

> **[RESULTS PENDING — fill from `results/` after the run.]**
> Insert `results/main.png` (answer NLL vs budget, one panel per side-information level,
> with full-cache and no-handoff reference lines), the paired comparison table
> (`comparisons.csv`) at the 10 % budget, and the interaction table
> (`interaction.csv`). State the sign and CI of the difference-of-differences before
> interpreting anything else.

`results/scores.png` carries the qualitative version: over the paragraphs the receiver
already holds, receiver-conditioned surprisal collapses while both sender-side signals
continue unchanged — they have no way to notice.

## 7. Interpretation

*To be written against the numbers, not the hypothesis.*

**One plausible alternative explanation.** The receiver-conditioned score comes from a
forward pass with a longer prefix, so it inherits a prefix-length effect as well as a
prefix-*content* effect. The control that separates them: match prefix length using
paragraphs sampled from *other* questions. If the advantage survives that, content is
doing the work; if it does not, the result is about prefix length. This is the first
thing I would add.

**Where the connection breaks down.** Our rule is not a Wyner–Ziv code. The theorem's
substance — the conditional rate without observing *Y* — depends on binning, which we do
not implement, so the measured advantage is an upper bound on what a deployable scheme
could reach. What the experiment can establish is whether the conditional *objective* is
worth the engineering; it cannot establish that the objective is attainable under the
constraint that makes the theorem interesting.

**Follow-up.** Charge a receiver→sender digest to the same budget: let the receiver send
*k* scalars summarising its state, rank at the sender against that digest, and count *k*
against the communication budget. If a small digest recovers most of the oracle gap, the
idea is practical and the next question is what the digest should be — which is exactly
where the binning literature becomes relevant rather than decorative. If it does not, the
oracle result stays a curiosity and the honest conclusion is that sender-side ranking is
good enough.
