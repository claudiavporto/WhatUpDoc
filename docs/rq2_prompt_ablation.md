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

## Next Steps

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
