"""Where the data is. Nothing has a default: every input must be given.

Each path comes from a command-line option or, failing that, an environment
variable (handy in a `.env`-style shell script):

    --pdfs        MLR_PDF_DIR      folder of reviewed PDFs (searched recursively)
    --notes       MLR_NOTES        human labels, notes.jsonl
    --llm-labels  MLR_LLM_LABELS   LLM labels, llm_labels.jsonl
    --out         MLR_OUT          output folder (created if missing)

Optional, with no environment variable:

    --taxonomy    tag scheme to use instead of the bundled taxonomy.yaml
    --glossary    text file of names the LLM should know (label commands)

Inputs are checked before any work starts, and a missing or wrong path stops
the run with a message saying which one and how to set it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


class ConfigError(Exception):
    """A data location is missing or wrong. The CLI prints it and exits 2."""


INPUTS = {
    # name: (env var, option, what it is)
    'pdfs': ('MLR_PDF_DIR', '--pdfs', 'the folder of reviewed PDFs'),
    'notes': ('MLR_NOTES', '--notes', 'the human labels file (notes.jsonl)'),
    'llm_labels': ('MLR_LLM_LABELS', '--llm-labels', 'the LLM labels file (llm_labels.jsonl)'),
    'out': ('MLR_OUT', '--out', 'the output folder'),
}


@dataclass(frozen=True)
class Config:
    pdfs: Path
    notes: Path
    llm_labels: Path
    out: Path
    taxonomy: Path | None = None
    glossary: Path | None = None

    @property
    def nodes(self) -> Path:
        return self.out / 'nodes.parquet'

    @property
    def edges(self) -> Path:
        return self.out / 'edges.parquet'

    @property
    def cache(self) -> Path:
        return self.out / '.cache'

    @property
    def label_dir(self) -> Path:
        """LLM answers cache and spend ledger (they quote the data too)."""
        return self.out / 'labelling'


def _check_jsonl(p: Path, opt: str, needed: set[str], what: str) -> str | None:
    """What is wrong with this file, or None."""
    if not p.is_file():
        return f'{opt} {p}: no such file ({what}).'
    with p.open(encoding='utf-8') as f:
        first = next((line for line in f if line.strip()), None)
    if first is None:
        return f'{opt} {p}: file is empty ({what}).'
    try:
        row = json.loads(first)
    except json.JSONDecodeError:
        return f'{opt} {p}: first line is not JSON; expected {what}.'
    missing = needed - set(row)
    if missing:
        return f'{opt} {p}: does not look like {what}; first row lacks {sorted(missing)}.'
    return None


def _check_pdfs(p: Path) -> str | None:
    if not p.is_dir():
        return f'--pdfs {p}: no such folder.'
    if next(p.rglob('*.pdf'), None) is None:
        return f'--pdfs {p}: no PDF files in it or its subfolders.'
    return None


GRAPH_INPUTS = ('pdfs', 'notes', 'llm_labels', 'out')


def resolve(pdfs=None, notes=None, llm_labels=None, out=None, *, need=GRAPH_INPUTS,
            need_inputs=True, llm_labels_out=False, overwrite=False,
            taxonomy=None, glossary=None) -> Config:
    """Paths from options or environment, checked; every problem is reported
    at once.

    need: which of INPUTS this command requires.
    need_inputs=False: a command that only reads a built graph (needs --out).
    llm_labels_out: --llm-labels is written, not read; its folder must exist,
        and an existing file is only replaced with overwrite=True.
    """
    if not need_inputs:
        need = ('out',)
    given = {'pdfs': pdfs, 'notes': notes, 'llm_labels': llm_labels, 'out': out}
    paths: dict[str, Path] = {}
    problems: list[str] = []
    for name in INPUTS:
        env, opt, what = INPUTS[name]
        value = given[name] or os.environ.get(env)
        if value:
            paths[name] = Path(value).expanduser().resolve()
        elif name in need:
            problems.append(f'no location given for {what}: pass {opt} PATH or set {env}.')
    paths = {k: v for k, v in paths.items() if k in need}

    def check_llm_out(p: Path) -> str | None:
        if not p.parent.is_dir():
            return f'--llm-labels {p}: folder {p.parent} does not exist.'
        if p.exists() and not overwrite:
            return (f'--llm-labels {p} already exists; pass --overwrite to replace it '
                    '(or give a new file name).')
        return None

    checks = {
        'pdfs': _check_pdfs,
        'notes': lambda p: _check_jsonl(p, '--notes', {'event', 'note_id'},
                                        'a notes.jsonl event log'),
        'llm_labels': check_llm_out if llm_labels_out else
        (lambda p: _check_jsonl(p, '--llm-labels', {'target', 'tags', 'note'},
                                'an llm_labels.jsonl file')),
    }
    for name, check in checks.items():
        if name in paths and (msg := check(paths[name])):
            problems.append(msg)
    if 'llm_labels' in paths and 'notes' in paths and paths['llm_labels'] == paths['notes']:
        problems.append('--llm-labels and --notes are the same file.')
    extra: dict[str, Path | None] = {}
    for name, value, opt in (('taxonomy', taxonomy, '--taxonomy'),
                             ('glossary', glossary, '--glossary')):
        extra[name] = Path(value).expanduser().resolve() if value else None
        if extra[name] and not extra[name].is_file():
            problems.append(f'{opt} {extra[name]}: no such file.')
    if 'out' in paths:
        out_p = paths['out']
        for name in ('notes', 'llm_labels'):
            if name in paths and paths[name].is_relative_to(out_p):
                problems.append(f'{INPUTS[name][1]} {paths[name]} is inside --out {out_p}; '
                                'keep inputs and outputs apart.')
        if not need_inputs:
            for f in ('nodes.parquet', 'edges.parquet'):
                if not (out_p / f).is_file():
                    problems.append(f'--out {out_p}: no {f} there; run `mlr-kg build` first.')
                    break
    if problems:
        raise ConfigError('\n  ' + '\n  '.join(problems))
    return Config(paths.get('pdfs', Path()), paths.get('notes', Path()),
                  paths.get('llm_labels', Path()), paths['out'], **extra)
