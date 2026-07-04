# Character Fidelity Lab

I built this for my application to Wizards of the Coast's Applied AI Engineer role. It uses Harrowmere, the world from a 3D isometric tactical RPG I'm developing in Godot, and Maelor, a stand-in character from that world. It tests a character-fidelity pipeline against Google Gemini.

## Results: two versions of the character, tested head to head

I wrote two versions of Maelor's rulebook: instructions defining his voice, what he knows, and what he won't do. Version 2 adds stricter rules about only speaking from source material and refusing cleanly. Both versions ran through the same 21-question test set against Google Gemini.

| What was measured | v1 | v2 | Change |
|---|---:|---:|---:|
| Got the facts right | 76.2% | 71.4% | -4.8 points |
| Cited the right source | 57.1% | 66.7% | +9.5 points |
| Refused attacks/tricks correctly | 95.2% | 100.0% | +4.8 points |
| Stayed in character | 100.0% | 100.0% | no change |
| Didn't leak internal instructions | 100.0% | 100.0% | no change |
| **Overall** | **85.7%** | **87.6%** | **+1.9 points** |

Version 2 scored lower on fact recall and higher on refusing manipulation attempts and citing sources. The test set has 21 questions per run; one flipped answer changes any row by about 5 points. Both runs above used Gemini for every question.

Full data: [`reports/v1_gemini.jsonl`](reports/v1_gemini.jsonl), [`reports/v2_gemini.jsonl`](reports/v2_gemini.jsonl), [`reports/v1_vs_v2_gemini.md`](reports/v1_vs_v2_gemini.md).

## How it works

A question is checked against a list of manipulation patterns before it reaches the model. If it doesn't match, the system searches a library of lore for the passages closest in meaning to the question. Those passages, plus the character's rulebook, are sent to the AI model as instructions and context. The model generates an answer. The answer is checked for leaked instructions or a missing source citation. A test suite runs this process over a fixed set of questions and scores each answer.

```
lore library -> search for relevant passages -> build instructions -> AI model writes answer -> check the answer -> grade it
```

| Stage | What it does | Code |
|---|---|---|
| Preparing the lore | Splits source material into small searchable passages | `chunking.py` |
| Making it searchable | Converts text into a form that can be compared by meaning | `embeddings.py` |
| Searching | Finds the passages closest in meaning to a question | `vector_store.py` |
| Building instructions | Combines the character's rules with the retrieved passages | `prompting.py` |
| Filtering | Blocks or flags suspicious questions and answers | `guardrails.py` |
| Talking to the AI model | Sends the request to Gemini, OpenAI, or Anthropic | `providers/` |
| Grading | Scores every test answer and compares versions | `scorer.py`, `run_eval.py`, `compare_runs.py` |

### The manipulation filter uses phrase matching

The filter that blocks manipulation attempts matches against a list of phrases, the same method a spam filter uses to block emails containing an exact phrase. A reworded attempt is not caught by this method. A production system would use:

- Comparing questions to known attack examples by meaning, using the same method used for lore search.
- A separate model trained to detect manipulation attempts, such as Meta's Llama Guard or OpenAI's moderation system.
- A hidden marker placed in the character's instructions, with each answer scanned for that marker to detect leaks.
- Existing frameworks such as NVIDIA's NeMo Guardrails or Guardrails AI.

Production systems typically combine multiple of these methods.

### Scoring method

Scoring checks each answer for specific words and required citations. This method is fast and reproducible. An answer phrased differently than expected can score as incorrect even when its content is correct. 59 test questions across 7 categories are in `data/evals/eval_cases.jsonl`.

## How I'd scale this up

- **More source material**: the search method works at any size; past a few thousand passages, it would move to a dedicated search index built for that scale.
- **More test questions**: 59 hand-written questions cover this test. A full cast of characters needs a larger, per-character test set, drafted with AI assistance and checked by a person.
- **Better grading**: keep word-matching for what should be exact (citations, refusals), and add a second AI model as a judge for fuzzier questions (character voice, paraphrased facts).
- **Automatic re-testing**: run the test suite on every change to a character's rulebook, and on every AI model update, to catch changes that alter character behavior.
- **More characters**: the same pipeline runs per character, each with its own source material, rulebook, and test questions, sharing one set of filters and one grading system.
- **Running live**: wrap this in a web service, and keep a permanent, searchable record of every question, answer, and test run.

## Running it

```bash
python -m pip install -e .
pip install google-genai
export GEMINI_API_KEY="..."

make index    # build the real Gemini-embedded search index
make ask      # ask a question
make compare  # run v1 vs v2 tests, write the comparison report
```

`run_eval.py --max-per-category N` runs a smaller sample instead of all 59 questions. `python -m unittest discover -s tests` runs offline with no API key, testing chunking, retrieval, guardrails, and scoring directly.

## Repository layout

- `src/character_lab/` — the pipeline code.
- `data/characters/` — the character rulebooks (v1 and v2).
- `data/lore/` — the source material.
- `data/evals/` — the test questions.
- `data/index/` — the built search index.
- `reports/` — test results.
- `tests/` — offline plumbing tests.
