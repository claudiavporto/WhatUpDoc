# RQ2: Prompt-Profile Ablation and Grounding Evaluation

Owner: Christopher Swartz (feature/llm-infrastructure)

## Research Question

**RQ2: How does prompt design (and, within that, top-k/verbosity choices)
affect LLaMA 3 8B response accuracy and grounding?**

This document reports results from `experiments/03_prompt_ablation.py`, which
compares two prompt profiles (`strict_cited`, `concise`) against a labeled
question set over the sample corpus (`data/raw/`), scoring each on refusal
accuracy, citation rate, and mean grounding score. See `src/llm.py` for the
prompt definitions and `src/grounding.py` for the scoring implementation.

## Methodology

- **Evaluation set:** 9 questions over 3 sample documents (lease PDF, medical
  record DOCX, utility policy TXT) — 6 answerable, 3 deliberately unanswerable
  (patient blood type, landlord phone number, authority budget; none present
  in the documents).
- **Metrics:**
  - **Refusal accuracy** — of the unanswerable questions, the fraction where
    the model correctly emitted the fixed refusal string instead of guessing.
  - **Citation rate** — of the answerable questions, the fraction where the
    model's answer included at least one citation.
  - **Mean grounding score** — of the citations produced, the fraction that
    point to a chunk that was actually retrieved for that question (vs.
    fabricated). Computed by `src/grounding.py`'s `verify_grounding()`.

**Self-test validation (offline, no model required):**
`python experiments/03_prompt_ablation.py --selftest` validates the grounding
verifier itself against fixed, hand-written answer fixtures before trusting it
on live model output:

```
PASS  fully grounded answer scores 1.0
PASS  fabricated citation detected and scored 0.0
PASS  refusal recognized, not scored as a claim
PASS  mixed answer scores 0.5 with one fabrication flagged
Grounding verifier self-test passed (4/4). No Ollama required.
```

This confirms the *scoring logic* is correct before it's applied to real
model answers below — a fabricated citation is reliably caught, a genuinely
grounded answer scores 1.0, and refusals aren't miscounted as ungrounded
claims.

## Results (Live Run, LLaMA 3 8B / Mistral 7B fallback)

`python experiments/03_prompt_ablation.py` — full run against a live local
model:

| Profile | Refusal Accuracy | Citation Rate | Mean Grounding Score |
|---|---|---|---|
| `strict_cited` | 1.0 | 0.67 | 0.0 |
| `concise` | 1.0 | 0.5 | 0.0 |

**Note on model used:** the primary model (`llama3:8b`) was not pulled in the
environment this run was executed in; `src/llm.py`'s automatic fallback
engaged and the run completed against `mistral:7b` instead (logged: `Model
'llama3:8b' not found; falling back to 'mistral:7b'`). These results
therefore reflect Mistral 7B's behavior under these prompts, not LLaMA 3 8B's.
Re-running after `ollama pull llama3:8b` would be needed to report on the
primary model specifically.

## Interpretation

**Refusal accuracy: strong evidence the safety behavior works.** Both
profiles correctly refused all 3 unanswerable questions (1.0/1.0) — the
model did not hallucinate an answer to any question the documents genuinely
don't cover, which is the most safety-critical behavior this system depends
on.

**Grounding score: 0.0 for both profiles is the central finding of this
run, and it warrants a specific explanation rather than being read at face
value as "the model fabricates everything it cites."** Every citation
produced during this run was flagged as unsupported by `verify_grounding()`.
Inspecting the actual flagged citations from the run log shows a consistent
pattern rather than random hallucination:

- Several citations were logged as `('source: sample_utility_policy.txt', 1)`
  — including the literal word `"source:"` inside the citation, which mirrors
  the excerpt header format shown to the model as context
  (`"(source: X, page Y)"` in `format_context()`) rather than the citation
  format the prompt actually instructs (`[source, page N]`, i.e. no
  `"source:"` prefix).
- Two citations were logged as `('excerpt 1', 1)` — citing by excerpt number
  rather than source filename at all.

This is consistent with a **citation-format mismatch between what the prompt
asks for and what the model actually produces**, which `verify_grounding()`
then fails to parse as a match — rather than the model citing content it was
never given. This is more than a speculative hypothesis: `run_selftest()`'s
own hand-written "good" example uses exactly this format —

```python
good = "Rent is $1,850.00, due on the 1st [sample_lease_agreement.pdf, page 1]."
```

— a bare filename with no `"source:"` prefix, no other wording. That is
precisely what the verifier was built and tested to recognize as valid.
Neither of the live run's flagged formats, `source: sample_utility_policy.txt`
or `excerpt 1`, matches it. The most likely explanation: the model is echoing
the `"(source: X, page Y)"` phrasing from the excerpt header shown to it as
context (`format_context()` in `src/llm.py`) rather than stripping the
`"source:"` label the way the prompt's rule 2 instructs and the self-test
fixture demonstrates. If that holds up under direct inspection of a raw
answer, the fix is straightforward: either add a negative example to the
prompt ("cite as `[filename, page N]`, NOT `[source: filename, page N]`"),
or loosen `verify_grounding()`'s parsing to also accept the `"source: "`
prefix and excerpt-number forms as valid.

**This still has not been directly confirmed against a raw generated
answer** — the run log shows the parsed citation tuples, not the full
original text — but the self-test's own definition of a valid citation
makes the format-mismatch explanation considerably more likely than a
genuine hallucination in every single case across two full prompt profiles.

**Citation rate differs as expected between profiles.** `strict_cited`
(0.67) produces citations more often than `concise` (0.5), consistent with
`strict_cited`'s explicit "cite every claim" rule versus `concise`'s lighter
instruction — the prompt design is measurably changing model behavior in the
intended direction, even though the citations themselves aren't currently
being verified as grounded for the reason above.

## Milestone 4 Update

The four next steps below were identified at the end of Milestone 3. Items
1, 2, and 4 are now implemented; item 3 (and the live reruns) are the only
remaining actions, and each is a single command.

### What changed

**1. Raw-answer inspection is now built in (settles next step 1).** Every
full run of `experiments/03_prompt_ablation.py` now writes
`experiments/results/prompt_ablation_raw_answers.md` containing the complete
generated text of every answer, alongside the context headers shown to the
model and the per-answer grounding summary. The format-mismatch hypothesis
no longer has to be argued from parsed citation tuples — the raw evidence is
captured on every run.

**2. The citation-format mismatch is fixed on both sides (next step 2).**

- *Prompt side* (`src/llm.py`): rule 2 of `strict_cited` now includes an
  explicit correct/incorrect example pair — `CORRECT: [lease_agreement.pdf,
  page 2]`, `INCORRECT: [source: lease_agreement.pdf, page 2]`, `INCORRECT:
  [Excerpt 1]` — since the M3 finding was that the rule stated the format
  but never showed the model what not to do. `concise` received a compact
  version of the same example.
- *Verifier side* (`src/grounding.py`): `resolve_citations()` now resolves
  the two nonstandard-but-unambiguous formats the M3 run actually produced
  (`[source: file, page N]` and excerpt-number citations, with or without a
  page) to the real retrieved chunk they refer to, instead of miscounting
  them as fabrications. Out-of-range excerpt numbers are still fabrications.

Critically, the two failure modes are now scored as **separate metrics**:

- **Grounding score** — does the citation point to content that was
  actually retrieved? (Resolved nonstandard citations count as grounded.)
- **Format compliance** — was the citation written in the exact instructed
  format? (Resolved nonstandard citations count as failures here.)

The M3 run's `0.0` grounding score conflated these; separating them is what
lets the report say whether the model was *fabricating* (a safety problem)
or *misformatting* (a prompt-adherence problem). The M3 evidence pointed to
the latter, and the rerun below tests whether the prompt fix closes it.

**3. Model provenance is recorded per answer.** `src/llm.py` now tracks
`last_model_used` on every generation, and the results CSVs include a
`model_used` column. The M3 run silently reported Mistral 7B fallback
numbers as if they were the primary model's; that can no longer happen
unnoticed.

**4. The top-k sweep is implemented (next step 4).** RQ2's phrasing ("How
does top-k context size affect response accuracy?") is now directly
testable: `--topk-sweep 1,2,4,6` holds the prompt profile fixed and varies
retrieval depth, reporting the same metrics plus mean per-answer latency at
each setting (`experiments/results/topk_sweep.csv`).

The verifier changes are covered by an expanded offline self-test (8 cases,
up from 4) using the exact citation formats logged in the M3 run:

```
python experiments/03_prompt_ablation.py --selftest
```

### Reruns to execute (needs Ollama, ~10 min total on the sample corpus)

```powershell
ollama pull llama3:8b                                      # next step 3
python experiments/03_prompt_ablation.py                   # rerun ablation
python experiments/03_prompt_ablation.py --topk-sweep 1,2,4,6
```

### Results (Milestone 4 rerun, primary model)

*To be filled from `prompt_ablation.csv` after the rerun:*

| Profile | Model Used | Refusal Acc. | Citation Rate | Grounding | Format Compliance |
|---|---|---|---|---|---|
| `strict_cited` | | | | | |
| `concise` | | | | | |

Expected movement vs. M3: grounding score should rise substantially once
format-mismatched citations are resolved rather than miscounted; format
compliance measures whether the prompt's new negative examples changed the
model's citation behavior itself.

### Results (top-k sweep, `strict_cited`)

*To be filled from `topk_sweep.csv`:*

| top_k | Refusal Acc. | Citation Rate | Grounding | Format Compliance | Mean Latency (s) |
|---|---|---|---|---|---|
| 1 | | | | | |
| 2 | | | | | |
| 4 | | | | | |
| 6 | | | | | |

Interpretation guide: rising top_k gives the model more context to ground
in (citation rate and grounding may improve) but more header formats to
echo and more tokens to process (format compliance and latency may
degrade). Refusal accuracy at high top_k is the safety-critical cell —
more context increases the temptation to answer unanswerable questions
from near-miss chunks.

## Next Steps (as identified at Milestone 3)

1. Inspect 2–3 raw generated answers directly to confirm the model is echoing
   the excerpt-header `"source: "` phrasing rather than following the
   prompt's `[filename, page N]` instruction — this would settle the
   hypothesis above definitively.
2. If confirmed: add an explicit correct/incorrect citation example pair to
   the system prompt (e.g., "cite as `[filename, page N]` — NOT `[source:
   filename, page N]`"), since the current instruction states the rule but
   doesn't show the model what NOT to do, which is likely why it defaulted
   to echoing the header format it saw in its own context window.
3. Re-run with `llama3:8b` actually pulled, to report on the primary model
   rather than the Mistral 7B fallback.
4. Consider adding the top-k/verbosity sweep referenced in the README's RQ2
   phrasing ("How does top-k context size affect... response accuracy?") —
   the current ablation varies prompt profile only, not top-k, as a separate
   factor.
