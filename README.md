# mlr-kg

Builds a knowledge graph from MLR (Medical-Legal-Regulatory) review comments
left on PDF materials. It links each reviewer comment to the sentence it marks,
the concern it raises, the edit it proposes, and whether the next version of
the document took that edit. The build itself makes no LLM calls.

It also has the LLM labelling pipeline (`mlr-kg label …`). This labels the
comments nobody tagged by hand and measures agreement with the hand labels.

**This repository holds code only. It contains no review data**, and
`.gitignore` blocks PDFs, label files and every output, because all of them
quote confidential client material. Keep your data outside the repo, or in a
git-ignored folder, and point the commands at it.

## Install

Needs Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Inputs: you must say where they are

There are no default locations. Give each one as an option or an environment
variable. If one is missing, the path doesn't exist, or a file doesn't look
like what it should be, the command stops before doing any work and says
which input is wrong (exit status 2).

| Option | Environment variable | What it is |
|---|---|---|
| `--pdfs DIR` | `MLR_PDF_DIR` | Folder of reviewed PDFs, searched recursively (see *PDF layout*). |
| `--notes FILE` | `MLR_NOTES` | Human labels: the `notes.jsonl` event log (see *Label files*). |
| `--llm-labels FILE` | `MLR_LLM_LABELS` | LLM labels for the comments not labelled by hand: `llm_labels.jsonl`. |
| `--out DIR` | `MLR_OUT` | Where to write results. It is created if missing, and must not hold the label files. |

```bash
uv run mlr-kg build \
  --pdfs "/path/to/MLR FOLDER - April 2026" \
  --notes /path/to/notes.jsonl \
  --llm-labels /path/to/llm_labels.jsonl \
  --out /path/to/mlr-kg-output
```

Or set the variables once:

```bash
export MLR_PDF_DIR="/path/to/MLR FOLDER - April 2026"
export MLR_NOTES=/path/to/notes.jsonl
export MLR_LLM_LABELS=/path/to/llm_labels.jsonl
export MLR_OUT=/path/to/mlr-kg-output
uv run mlr-kg build
```

The input files are only read, never written.

## Commands

| Command | Needs | Does |
|---|---|---|
| `build` | all four | Reads the PDFs and labels, writes the graph, then runs `check` and `export`. Takes a few minutes. |
| `check` | `--out` | Prints node and edge counts and how comments were anchored. |
| `export` | `--out` | Writes `kg_overview.html` (an interactive viewer) and `kg_full.graphml` (the whole graph, for Gephi or Cytoscape). |
| `handoff` | all four | Writes `handoff/comments_labelled.{parquet,csv}`: one row per comment with its labels, who made them, and its file, page, sentence and passage. Run `build` first. |

`uv run mlr-kg COMMAND --help` lists the options for each command.

## LLM labelling (`mlr-kg label …`)

The graph reads LLM labels for the comments nobody labelled by hand. These
commands make that file. They prompt Claude or GPT-4o using the human labels as
examples, score the answers against the human labels, and export the result as
`llm_labels.jsonl`.

```bash
uv sync --extra label                       # API clients, only needed for ping/run
export ANTHROPIC_API_KEY=…                  # and/or OPENAI_API_KEY; never read from a file
L=(--pdfs "$MLR_PDF_DIR" --notes "$MLR_NOTES" --out "$MLR_OUT" --glossary glossary.txt)

uv run mlr-kg label splits   "${L[@]}"                        # free: dev / test / unlabelled counts
uv run mlr-kg label show P4 <uid> "${L[@]}"                   # free: print one prompt
uv run mlr-kg label estimate "${L[@]}"                        # free: cost table
uv run mlr-kg label ping --model claude-sonnet-5-5 "${L[@]}"  # 1 paid call: checks the key
uv run mlr-kg label run --model claude-sonnet-5-5 --variant P4 --split test "${L[@]}"
uv run mlr-kg label score --split test "${L[@]}"              # free: agreement with the human labels
uv run mlr-kg label run --model claude-sonnet-5-5 --variant P4 --split unlabelled "${L[@]}"
uv run mlr-kg label export --model claude-sonnet-5-5 --variant P4 \
    --llm-labels /path/to/llm_labels.jsonl "${L[@]}"          # free: write the labels file
```

- **Inputs.** The label commands need `--pdfs`, `--notes` and `--out`.
  `export` also needs `--llm-labels`, which is the file it *writes*. It refuses
  to replace an existing file unless you pass `--overwrite`, and it writes the
  whole file or nothing.
- **Splits.** Human-labelled comments are split into **dev** and **test**.
  Prompts are tuned on dev, and few-shot examples come only from dev. Test is
  scored to report agreement. **unlabelled** is everything else, the comments
  the LLM labels for real.
- **Prompt variants.** P1 gives tag names only and P2 adds definitions. P3 adds
  fixed examples; P4 instead adds the 8 most similar dev comments. P5 and P6 add
  a written rationale. P7 is P3 without the page text. P8 is P4 with examples
  from other documents only. `mlr-kg label -h` describes them in full. P4 with
  Claude is what the current labels use.
- **`--glossary`** is a plain-text file explaining names the model will meet,
  such as agencies and their abbreviations. It is added to every prompt. See
  `glossary.example.txt`. A real glossary names clients, so `.gitignore` keeps
  `glossary*.txt` out of the repo.
- **Cost safety.** Each model stops once its total spend would pass `--budget`
  (default $10). The spend is logged in `<out>/labelling/spend.jsonl`. Answers
  are cached in `<out>/labelling/answers.jsonl`, so reruns, `score` and `export`
  are free. `run` only pays for prompts it hasn't seen before. Both files
  quote the data, so keep them out of git.
- **Reproducing existing labels.** Copy the original `answers.jsonl` into
  `<out>/labelling/`, and use the same `--glossary` and `--taxonomy` text as the
  original run. `export` then rebuilds `llm_labels.jsonl` without any API
  calls. Any change to the prompt text makes `run` ask again, and pay again.
- **Versions.** The SDKs are pinned to the versions the current labels were
  made with (anthropic 0.111.0, openai 2.43.0).

## Outputs (in `--out`)

| File | Contents |
|---|---|
| `nodes.parquet` | `id, kind, label, text, file, page, x0, y0, x1, y1, attrs`. `attrs` is a JSON string. |
| `edges.parquet` | `src, rel, dst, attrs`. |
| `kg_overview.html` | A browser viewer. It loads sigma.js from a CDN, so it needs internet. |
| `kg_full.graphml` | Every node and edge. |
| `handoff/` | The flat comments table and a copy of `taxonomy.yaml`. |
| `.cache/` | Annotations extracted from the PDFs. They are re-read automatically when any PDF changes. |
| `labelling/` | The LLM answer cache and spend ledger (`mlr-kg label`). |

Coordinates are PDF points, with the origin at the top-left of the page, and
pages are numbered from 1. In Python:

```python
import json, pandas as pd
nodes = pd.read_parquet('out/nodes.parquet')
edges = pd.read_parquet('out/edges.parquet')
nodes['attrs'] = nodes['attrs'].map(json.loads)   # use ['attrs']: df.attrs is a pandas property
```

## The graph

```
Brand -[HAS_PIECE]-> Piece -[HAS_VERSION]-> Version -[NEXT]-> Version
Version -[HAS_PASSAGE]-> Passage -[HAS_CLAIM]-> Claim (a sentence)
Comment -[ON]-> Claim             the sentence under its mark, or the nearest within 60 pt
Comment -[ON_PAGE]-> Version      no text near the mark
Reply -[REPLIES_TO | AGREES | DISAGREES]-> Comment
Comment -[BY]-> Person
Comment -[RAISES]-> Concern       issue tags only
Comment -[PROPOSES]-> Edit -[APPLIED_IN | NOT_APPLIED_IN]-> Version
Claim -[SAME_AS | REVISED_TO]-> Claim    the matching sentence in the next version
Claim -[CITES]-> Reference        inline [n/Source] citations and reference-link notes
Comment -[MENTIONS]-> Reference   job codes
Comment -[ABOUT_REFERENCE]-> Reference
Comment -[RESOLVES]-> Comment     an author's "changed X as requested"
```

The docstring at the top of `src/mlr_kg/kg.py` gives the exact rules and
thresholds.

## PDF layout

`--pdfs` is expected to look like this:

```
<pdfs>/<Brand> veeva examples/<Piece>/<vX.Y name>.pdf
```

- The first folder names the brand, with ` veeva examples` stripped if present.
- Each folder below it is one piece.
- The `vX.Y` in each file name orders that piece's versions.

Comments are the PDF annotations. Replies are linked through the PDF's
in-reply-to field.

Comment ids (`uid`) are a hash of the PDF's path under `--pdfs` and the
annotation's own id. Keep the same folder structure, or the ids, and so the
labels, will no longer match.

## Label files

**`notes.jsonl`** holds the human labels. It is an append-only log, one JSON
object per line, and the current labels are the log replayed in order:

```json
{"event": "add", "note_id": "n…", "at": "2026-10-27T21:09:51+00:00", "by": "…",
 "target": "<comment uid>", "text": "free-text note", "tags": ["wording"], "snapshot": {…}}
{"event": "edit", "note_id": "n…", "at": "…", "by": "…", "text": "…", "tags": […]}
{"event": "delete", "note_id": "n…", "at": "…", "by": "…"}
```

Tags are mapped onto the 13 tags in `src/mlr_kg/taxonomy.yaml`, and every tag
used must appear there under some `from:` list. In any PDF with at least one
note, comments left without a note count as `no tag`. Human-labelled comments
are split into dev and test halves (seed 0).

**`llm_labels.jsonl`** holds one JSON object per comment:

```json
{"target": "<comment uid>", "model": "…", "variant": "P4", "tags": ["references"],
 "note": "one-line reason", "confidence": "high", "suggested_new_tag": null,
 "gpt-4o_tags": […], "my_tags": […], "split": "unlabelled"}
```

Human tags always win over LLM tags for the same comment.
