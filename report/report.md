# What Should a Latent Message Preserve?

Songhan (Mason) Wu · DAPLab Latent-comm task
Code and results: <https://github.com/SonghanWu/latent-message-handoff>

## 1. The idea

A. D. Wyner and J. Ziv, "The rate-distortion function for source coding with side
information at the decoder," *IEEE Trans. Inform. Theory* 22(1):1–10, 1976
([doi](https://doi.org/10.1109/TIT.1976.1055508)), extending D. Slepian and J. K. Wolf,
*IEEE Trans. Inform. Theory* 19(4):471–480, 1973
([doi](https://doi.org/10.1109/TIT.1973.1055037)).

Slepian–Wolf: when the decoder holds correlated side information *Y*, a source *X*
compresses to *H(X|Y)* bits even though the encoder never observes *Y*. Wyner–Ziv carries
this into the lossy regime.

Two consequences matter here. First, value is conditional rather than marginal — what a
piece of *X* is worth depends on the uncertainty left after conditioning on *Y*. Second,
the distortion measure is a free choice; nothing requires *d(x, x̂)* to be reconstruction
error.

The second is why this is not just "avoid duplicates". H2O ranks cache entries by
accumulated attention, SnapKV by the attention a trailing window pays to the prefix,
StreamingLLM by position. All three are computed inside the sender, because in
single-model inference there is no receiver to condition on. An agent handoff has one.

## 2. The mapping

| Wyner–Ziv | here | exact or analogy |
|---|---|---|
| encoder, source *X* | the worker and its KV cache over the document | exact |
| decoder | the receiver, a separate instance answering the question | exact |
| side information *Y* | the receiver's own context, plus model priors | context exact; priors informal |
| message | the selected cache entries | exact |
| rate *R* | the budget: *B* positions × 6,144 scalars (1.71M at 20 %) | exact, counted per run |
| distortion *d* | the receiver's NLL of the gold answer | valid distortion, not the MSE the closed forms assume |
| encoder blind to *Y* | H2O / SnapKV ranking blind to the receiver | exact |

Where it stops. A transformer cache is not an i.i.d. source — no block length, no
asymptotics, so no achievable-rate claim transfers, only the direction of the prediction.
And we do selection, not binning: Wyner–Ziv's substance is that the encoder reaches the
conditional rate *without* seeing *Y*, via a coset construction we do not implement. Our
rule reads the receiver's state directly, which makes it an oracle bound rather than a
codec.

## 3. The claim

> At a fixed budget, ranking by receiver-conditioned surprisal beats ranking by sender
> salience or sender-side surprisal, and the margin grows with the receiver's side
> information.

The interaction is the load-bearing half: a main effect admits many explanations, a
margin that scales with the receiver's context is what conditionality specifically
predicts.

| competing explanation | its prediction |
|---|---|
| salience — the sender's attention marks what matters | sender attention wins regardless of receiver |
| information content — rare content matters | unconditional surprisal wins regardless of receiver |
| surface redundancy — drop what the receiver literally has | literal dedup captures the whole effect |

Falsifiers, declared before the run: (a) sender-side rules match or beat it; (b) it is
indistinguishable from literal deduplication; (c) the margin does not grow with side
information.

## 4. The experiment

HotpotQA distractor gives each question two gold paragraphs and eight distractors. A
worker reads all ten as one document. The receiver holds *s* ∈ {0, 3, 6} **distractor**
paragraphs, never a gold one, so the evidence must cross the handoff while the overlap
with the worker's cache is a free parameter.

Five rules, each spending the same budget (5 / 10 / 20 % of document tokens):

1. `random`
2. `sender_attention` — accumulated attention over all layers and heads
3. `sender_surprisal` — −log p(token | question, doc<t)
4. `dedup_sender_surprisal` — the same, minus positions the receiver literally holds
5. `receiver_surprisal` — −log p(token | side info, question, doc<t)

Rules 3–5 share one scoring function and differ only in whether the side information is
in the prefix, so 5 vs 3 isolates conditionality and 5 vs 4 isolates inferable from
verbatim redundancy.

Metric: mean token NLL of the gold answer read out of the handed-over cache. Statistics:
paired bootstrap over questions.

Three choices that would otherwise confound the comparison. Selected keys keep their
original rotary phase and the question sits at `doc_len`, so the receiver sees a position
sequence with holes — the alternative, unrotating and renumbering, destroys the original
spacing between transmitted spans. A position the receiver already holds still costs
budget when a rule picks it, because that waste is the phenomenon. Every rule is forced
to keep the first four positions, so attention-sink survival is not the hidden variable.

At *s* = 0 rules 3–5 score with an identical context and must select identical sets; the
code asserts it.

## 5. Implementation

Cache slicing by position, three scoring signals, five rules, an answer-NLL read-out, a
paired bootstrap. No compression system, no learned projector.

`scripts/smoke_test.py` runs on a randomly initialised tiny Qwen2, no download, and
asserts the things that would otherwise fail silently: scoring an answer through a
full-cache handoff equals scoring it in one ordinary forward pass to 1e-3; chunked
attention accumulation matches a full-length pass; `position_ids` reaches RoPE; sliced
caches hold exactly the original rows in every layer.

## 6. Result

100 questions, Qwen2.5-0.5B-Instruct, ~1391-token documents. Figure: `results/main.png`.

Mean answer NLL at the 10 % budget:

| rule | s = 0 | s = 3 | s = 6 |
|---|---|---|---|
| no handoff | 3.739 | 8.304 | 6.215 |
| random | 3.417 | 3.940 | 4.110 |
| sender_attention | 3.902 | 4.157 | 4.154 |
| sender_surprisal | 3.024 | 3.348 | 3.610 |
| dedup_sender_surprisal | 3.024 | 3.107 | 2.803 |
| receiver_surprisal | 3.024 | 3.024 | 2.918 |
| full cache | 2.386 | 2.386 | 2.386 |

Paired differences, negative meaning the first rule is better:

| comparison | s = 0 | s = 3 | s = 6 |
|---|---|---|---|
| receiver − sender_surprisal | 0.000 | −0.324 [−0.492, −0.174] | −0.692 [−0.990, −0.415] |
| receiver − dedup_sender | 0.000 | −0.083 [−0.201, +0.015] | +0.115 [−0.025, +0.286] |
| dedup_sender − sender_surprisal | 0.000 | −0.242 [−0.399, −0.098] | −0.807 [−1.102, −0.528] |

The *s* = 0 invariant holds to every printed digit, so the scoring channels are aligned.

Conditioning works and its margin grows: 0.000 → −0.324 → −0.692. Since the *s* = 0
difference is identically zero per question, the difference-of-differences equals the
*s* = 6 column — −0.598 [−0.839, −0.374], −0.692 [−0.990, −0.415], −0.615 [−0.904,
−0.328] at the three budgets, significant at each. The mechanism shows up in the budget
accounting: at *s* = 6 and 20 %, `sender_surprisal` spends 182 of 278 tokens (65 %) on
content the receiver already holds, `receiver_surprisal` spends 8 (2.8 %). In absolute
terms it closes 96.5 % of the gap between no handoff and the full cache, against 80.5 %
for `sender_surprisal`.

The sharper claim fails. `receiver_surprisal` and `dedup_sender_surprisal` are
indistinguishable in all six non-trivial cells. Falsifier (b), triggered.

## 7. Interpretation

The directional prediction holds against the matched baseline, and the interaction holds
at every budget.

What does not hold is the interesting half — that the relevant redundancy is what the
decoder can *infer*, not only what it holds verbatim. HotpotQA hands the receiver its
paragraphs verbatim, so all redundancy here is literal and the design cannot separate the
two hypotheses. On this testbed Wyner–Ziv earns one thing: know what the receiver has and
skip it.

One baseline did not work. `sender_attention` performs at or below `random` (59.8 % vs
59.2 % of the gap closed at *s* = 6, 20 %). `results/scores.png` shows why: accumulated
attention is dominated by the position-0 sink, flattening every other position, so the
rule degenerates into "keep the beginning". That is a property of summing over everything
at prefill time, not a fair reproduction of H2O, which also keeps a recent window and
accumulates during decoding. The comparison against it should not be read as beating the
deployed family; the load-bearing comparison is against `sender_surprisal`.

An alternative explanation for the main effect: the receiver-conditioned score comes from
a forward pass with a longer prefix, so it inherits a prefix-length effect alongside a
prefix-content effect. Padding the prefix to the same length with paragraphs from other
questions would separate them.

One oddity: the no-handoff reference is worse with side information than without (8.30 at
*s* = 3, 6.22 at *s* = 6, against 3.74 at *s* = 0). Distractors and no evidence is worse
than nothing at all. It does not affect any within-cell comparison.

Follow-ups, in order. Paraphrased side information — same facts, different wording — so
literal deduplication stops working by construction and only semantic conditioning can
find the redundancy; this is the experiment that would actually test what failed above.
Then a fair salience baseline with sink positions excluded. Then charging a
receiver→sender digest to the same budget, which is the step from oracle to codec.
