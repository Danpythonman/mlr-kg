"""Label MLR comments with an LLM, using hand-labelled comments as the key.

The human labels (--notes) cover some documents. This module turns them
into the fixed tag set in taxonomy.yaml, asks an LLM to tag the same
comments with several prompt variants, measures how often it agrees with
the human labels, and then labels the remaining comments.

Gold labels (see labels.py):
- A comment's gold tags are the union of the (merged) tags of its notes.
  Comments in documents with at least one note, but none of their own, are
  gold `no tag`. Comments whose notes have text but no tag are left out.
- Gold comments are split once into dev and test (stratified by tag,
  seed 0). Prompts are developed on dev; few-shot examples only ever come
  from dev; test is scored only for the final comparison.

Prompt variants (each adds one thing to the previous one):
    P1  tag names only
    P2  + a one-line definition per tag
    P3  + fixed examples (up to EXAMPLES_PER_TAG per tag, from dev)
    P4  P2 + the NEIGHBOURS most similar dev comments as examples
    P5  P3 + a rationale written before the tags
    P6  P4 + a rationale written before the tags
    P7  P3 without the page text around the mark
    P8  P4, but examples only from other content folders, as for a comment
        in a document nobody has labelled (all versions of a piece share a
        folder, so P4 can find the same comment in another version)
Every prompt includes the comment's whole thread, and the --glossary text
if one is given.

Cost safety:
- Every call's cost is appended to <out>/labelling/spend.jsonl; a model's
  calls stop once its total would pass --budget (default BUDGET_USD).
- Answers are cached in <out>/labelling/answers.jsonl keyed by model,
  prompt and sample number, so reruns and rescoring are free.
- --notes is only read. `label export` writes --llm-labels, and will not
  replace an existing file without --overwrite.

Commands: see `mlr-kg label --help`.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import sys
import threading
from collections import Counter
from datetime import datetime, timezone
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pymupdf as fitz
from rapidfuzz import fuzz

from . import annotations as raw
from . import labels
from .config import Config, ConfigError

CFG: Config = None  # type: ignore[assignment]   set by configure()

MODELS = ('claude-sonnet-5-5', 'gpt-4o')
API_KEYS = {'claude': 'ANTHROPIC_API_KEY', 'gpt': 'OPENAI_API_KEY'}
VARIANTS = ('P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8')
BUDGET_USD = 10.0
SEED = labels.SEED
EXAMPLES_PER_TAG = 2
NEIGHBOURS = 8
CONTEXT_PT = 30          # page text this far above/below the mark, in points
MAX_FIELD_CHARS = 1200   # longer comment text is cut, with a visible marker
MAX_TOKENS = 600
CONCURRENCY = 8

# Prices per token, USD (Anthropic and OpenAI list prices, 2026-11).
PRICES = {
    'claude-sonnet-5-5': {'in': 2e-6, 'cache_write': 2.5e-6, 'cache_read': 1e-7, 'out': 1e-5},
    'gpt-4o': {'in': 2.5e-6, 'cache_write': 2.5e-6, 'cache_read': 1.25e-6, 'out': 1e-5},
}

TYPE_HELP = {
    'StrikeOut': 'text struck through for deletion; the comment, if any, is usually the replacement',
    'Caret': 'an insertion point; the comment is usually the text to insert',
    'Highlight': 'highlighted text with a comment',
    'Underline': 'underlined text with a comment',
    'Square': 'a box drawn around part of the page',
    'Text': 'a sticky note',
    'FreeText': 'a text box typed onto the page',
    'Line': 'a line drawn on the page',
}


def configure(cfg: Config) -> None:
    global CFG
    CFG = cfg
    raw.configure(cfg)
    labels.use_taxonomy(cfg.taxonomy)
    for f in (items, _by_uid, fixed_examples, glossary):
        f.cache_clear()


def answers_path() -> Path:
    return CFG.label_dir / 'answers.jsonl'


def spend_path() -> Path:
    return CFG.label_dir / 'spend.jsonl'


taxonomy = labels.taxonomy
tag_names = labels.tag_names


# --- Items -----------------------------------------------------------------

def _context(r, doc_cache: dict) -> str:
    """Page lines around the mark, de-duplicated the same way annotations does."""
    key = (r.file, int(r.page))
    if key not in doc_cache:
        with fitz.open(raw.ROOT / r.file) as doc:
            page = doc[int(r.page) - 1]
            doc_cache[key] = (page.get_text('words', sort=True), page.rect)
    words, prect = doc_cache[key]
    band = fitz.Rect(prect.x0, r.y0 - CONTEXT_PT, prect.x1, r.y1 + CONTEXT_PT)
    taken: list[tuple[fitz.Rect, str]] = []
    for w in raw._words_in(words, band):
        box = fitz.Rect(w[:4])
        if not any(t == w[4] and (box & b).get_area() > 0.5 * box.get_area() for b, t in taken):
            taken.append((box, w[4]))
    return ' '.join(t for _, t in taken)


@cache
def items() -> pd.DataFrame:
    """Every comment: thread links, human labels, split, and page context."""
    df = labels.items(raw.load_annotations(), CFG.notes)
    doc_cache: dict = {}
    df['context'] = [_context(r, doc_cache) for r in df.itertuples()]
    return df


@cache
def _by_uid() -> pd.DataFrame:
    return items().set_index('uid', drop=False)


def row(uid: str) -> pd.Series:
    if uid not in _by_uid().index:
        raise ConfigError(f'no comment with uid {uid!r} in --pdfs.')
    return _by_uid().loc[uid]


# --- Prompts ---------------------------------------------------------------

def _cut(s: str) -> str:
    s = (s or '').strip()
    return s if len(s) <= MAX_FIELD_CHARS else s[:MAX_FIELD_CHARS] + ' …[cut]'


def _when(ts) -> str:
    return pd.Timestamp(ts).strftime('%Y-%m-%d %H:%M') if pd.notna(ts) else '?'


def _thread(r: pd.Series) -> pd.DataFrame:
    it = items()
    return it[it.root_uid == r.root_uid].sort_values('created', kind='stable')


def render_item(r: pd.Series, with_context: bool = True) -> str:
    """The comment to label, with its whole thread."""
    doc = Path(r.file)
    lines = [f'Document: {doc.parent.as_posix()} / {doc.stem}', f'Page: {r.page}',
             f'Annotation type: {r.type} ({TYPE_HELP.get(r.type, "annotation")})']
    if r.marked_text:
        lines.append(f'Text under the mark: "{_cut(r.marked_text)}"')
    if with_context and r.context:
        lines.append(f'Page text around the mark: "{_cut(r.context)}"')
    lines.append('Thread (oldest first; >>> is the annotation to label):')
    for _, t in _thread(r).iterrows():
        role = 'reply' if pd.notna(t.parent_uid) else 'comment'
        mark = '>>>' if t.uid == r.uid else '   '
        lines.append(f'{mark} [{role}] {t.author or "(no author)"}, {_when(t.created)}: '
                     f'{_cut(t.content) or "(no text)"}')
    return '\n'.join(lines)


def _short(s: str, n: int = 160) -> str:
    s = ' '.join((s or '').split())
    return s if len(s) <= n else s[:n] + ' …'


def render_example(r: pd.Series) -> str:
    """A labelled example in a few lines, to keep the prompt small."""
    out = [f'{r.type}' + (f' over "{_short(r.marked_text, 80)}"' if r.marked_text else '')]
    for _, t in _thread(r).iterrows():
        mark = '>>>' if t.uid == r.uid else '   '
        out.append(f'{mark} {t.author or "(no author)"}: {_short(t.content) or "(no text)"}')
    out.append(f'Tags: {", ".join(r.gold)}' + (f'  |  Note: {r.my_note}' if r.my_note else ''))
    return '\n'.join(out)


def _tag_block(defs: bool) -> str:
    if not defs:
        return '\n'.join(f'- {t}' for t in tag_names())
    return '\n'.join(f'- {t}: {d["definition"]}' for t, d in taxonomy().items())


@cache
def fixed_examples() -> pd.DataFrame:
    """Up to EXAMPLES_PER_TAG dev comments per tag, picked at random (seeded)."""
    dev = items()[items().split == 'dev']
    rng = random.Random(SEED)
    root = dict(zip(dev.uid, dev.root_uid))
    picks: list[str] = []
    for t in tag_names():
        uids = sorted(dev[dev.gold.map(lambda g: t in g)].uid)  # noqa: B023 (used at once)
        rng.shuffle(uids)
        # One example per thread, so two examples never repeat a thread.
        for u in uids:
            if sum(t in row(p).gold for p in picks) >= EXAMPLES_PER_TAG:
                break
            if root[u] not in {root[p] for p in picks}:
                picks.append(u)
    return dev.set_index('uid').loc[picks].reset_index()


def _sim_key(r: pd.Series) -> str:
    return f'{r.type} {r.content} {r.marked_text}'.lower()


def neighbours(r: pd.Series, other_folders: bool = False) -> pd.DataFrame:
    """The NEIGHBOURS dev comments most like r (never r itself or its thread)."""
    it = items()
    dev = it[(it.split == 'dev') & (it.root_uid != r.root_uid)]
    if other_folders:
        dev = dev[dev.file.map(lambda f: Path(f).parent) != Path(r.file).parent]
    key = _sim_key(r)
    score = dev.apply(lambda d: fuzz.token_set_ratio(key, _sim_key(d)), axis=1)
    return dev.assign(_s=score).sort_values(['_s', 'uid'], ascending=[False, True]).head(NEIGHBOURS)


def _spec(variant: str) -> dict[str, Any]:
    base = {'P1': dict(defs=False, examples=None, rationale=False, context=True),
            'P2': dict(defs=True, examples=None, rationale=False, context=True),
            'P3': dict(defs=True, examples='fixed', rationale=False, context=True),
            'P4': dict(defs=True, examples='nearest', rationale=False, context=True),
            'P8': dict(defs=True, examples='nearest-other', rationale=False, context=True)}
    derived = {'P5': ('P3', dict(rationale=True)), 'P6': ('P4', dict(rationale=True)),
               'P7': ('P3', dict(context=False))}
    if variant in base:
        return base[variant]
    parent, change = derived[variant]
    return {**base[parent], **change}


SYSTEM_INTRO = """\
You label reviewer comments from Medical-Legal-Regulatory (MLR) review of \
pharmaceutical patient materials. Each comment is a PDF annotation made by \
a reviewer or the material's author, often with replies.

Give each comment the tags that describe what it does. Use the tag list \
below; give one main tag first, and add a second or third only when the \
comment clearly does more than one thing. If no tag fits, use the closest \
one and put a better name in suggested_new_tag.

"""

SYSTEM_TAIL = """\
Also write `note`: one short plain sentence (at most 20 words) saying what \
the comment does or what is notable about it.

Tags:
"""


@cache
def glossary() -> str:
    """The --glossary text as one paragraph ('' without one). Line breaks
    inside a paragraph are joined, so the file can be wrapped freely."""
    if not CFG.glossary:
        return ''
    paras = CFG.glossary.read_text(encoding='utf-8').strip().split('\n\n')
    return '\n\n'.join(' '.join(p.split()) for p in paras if p.strip())


def system_prompt(variant: str) -> str:
    s = _spec(variant)
    g = glossary()
    out = SYSTEM_INTRO + (g + '\n\n' if g else '') + SYSTEM_TAIL + _tag_block(s['defs'])
    if s['examples'] == 'fixed':
        ex = '\n\n'.join(render_example(r) for _, r in fixed_examples().iterrows())
        out += f'\n\nLabelled examples:\n\n{ex}'
    if s['rationale']:
        out += '\n\nFirst write `rationale`: two or three sentences of reasoning, then the tags.'
    return out


def user_prompt(variant: str, r: pd.Series) -> str:
    s = _spec(variant)
    out = ''
    if s['examples'] in ('nearest', 'nearest-other'):
        near = neighbours(r, other_folders=s['examples'] == 'nearest-other')
        ex = '\n\n'.join(render_example(d) for _, d in near.iterrows())
        out += f'Similar labelled comments:\n\n{ex}\n\n---\n\n'
    return out + 'Label this comment:\n\n' + render_item(r, with_context=s['context'])


def schema(variant: str) -> dict[str, Any]:
    props: dict[str, Any] = {}
    if _spec(variant)['rationale']:
        props['rationale'] = {'type': 'string'}
    props |= {
        'tags': {'type': 'array', 'items': {'type': 'string', 'enum': tag_names()}},
        'note': {'type': 'string'},
        'confidence': {'type': 'string', 'enum': ['high', 'medium', 'low']},
        'suggested_new_tag': {'type': ['string', 'null']},
    }
    return {'type': 'object', 'properties': props, 'required': list(props),
            'additionalProperties': False}


# --- Calls, cache, budget --------------------------------------------------

_lock = threading.Lock()


def _key(model: str, variant: str, system: str, user: str, sample: int) -> str:
    raw_key = json.dumps([model, system, user, schema(variant), sample])
    return hashlib.sha256(raw_key.encode()).hexdigest()


def load_answers() -> dict[str, dict]:
    out = {}
    if answers_path().exists():
        for line in answers_path().read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                out[rec['key']] = rec
    return out


def spent(model: str) -> float:
    if not spend_path().exists():
        return 0.0
    return sum(json.loads(line)['usd'] for line in spend_path().read_text().splitlines()
               if line.strip() and json.loads(line)['model'] == model)


def _append(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock, path.open('a') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def require_key(model: str) -> None:
    var = API_KEYS['claude' if model.startswith('claude') else 'gpt']
    if not os.environ.get(var):
        raise ConfigError(f'{var} is not set; {model} needs it. Export it in your shell '
                          '(it is never read from a file in this repo).')


class BudgetExceeded(RuntimeError):
    pass


class Caller:
    """One model; async calls with structured output, cost tracking and a cap."""

    def __init__(self, model: str, budget: float = BUDGET_USD):
        require_key(model)
        self.model = model
        self.budget = budget
        self.spent = spent(model)
        self.reserved = 0.0
        try:
            if model.startswith('claude'):
                import anthropic
                self.client = anthropic.AsyncAnthropic()
            else:
                import openai
                self.client = openai.AsyncOpenAI()
        except ImportError:
            raise ConfigError('the API clients are not installed; run `uv sync --extra label`.') from None

    def _cost(self, u: dict) -> float:
        p = PRICES[self.model]
        return (u['in'] * p['in'] + u['cache_write'] * p['cache_write']
                + u['cache_read'] * p['cache_read'] + u['out'] * p['out'])

    async def call(self, variant: str, system: str, user: str) -> tuple[dict, dict]:
        # Reserve a pessimistic estimate so concurrent calls can't overshoot.
        est = (len(system) + len(user)) / 3 * PRICES[self.model]['in'] + MAX_TOKENS * PRICES[self.model]['out']
        with _lock:
            if self.spent + self.reserved + est > self.budget:
                raise BudgetExceeded(f'{self.model}: ${self.spent:.2f} spent of ${self.budget:.2f}')
            self.reserved += est
        try:
            if self.model.startswith('claude'):
                ans, usage, served = await self._claude(variant, system, user)
            else:
                ans, usage, served = await self._openai(variant, system, user)
        finally:
            with _lock:
                self.reserved -= est
        usd = self._cost(usage)
        with _lock:
            self.spent += usd
        _append(spend_path(), {'at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                               'model': self.model, 'served_by': served, 'variant': variant,
                               'usd': usd, **usage})
        return ans, usage

    async def _claude(self, variant, system, user):
        r = await self.client.messages.create(
            model=self.model, max_tokens=MAX_TOKENS,
            system=[{'type': 'text', 'text': system, 'cache_control': {'type': 'ephemeral'}}],
            messages=[{'role': 'user', 'content': user}],
            # Thinking off: a short classification; P5/P6 reason in the JSON.
            extra_body={'thinking': {'type': 'between_tools'},
                        'output_config': {'effort': 'low',
                                          'format': {'type': 'json_schema', 'schema': schema(variant)}}},
        )
        u = r.usage
        usage = {'in': u.input_tokens, 'cache_write': u.cache_creation_input_tokens or 0,
                 'cache_read': u.cache_read_input_tokens or 0, 'out': u.output_tokens}
        if r.stop_reason == 'refusal':
            raise RuntimeError(f'refusal: {getattr(r, "stop_details", None)}')
        if r.stop_reason == 'max_tokens':
            raise RuntimeError('hit max_tokens')
        text = next(b.text for b in r.content if b.type == 'text')
        return json.loads(text), usage, r.model

    async def _openai(self, variant, system, user):
        r = await self.client.chat.completions.create(
            model=self.model, max_tokens=MAX_TOKENS,
            messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
            response_format={'type': 'json_schema', 'json_schema': {
                'name': 'labels', 'strict': True, 'schema': schema(variant)}},
        )
        u = r.usage
        cached = (u.prompt_tokens_details.cached_tokens or 0) if u.prompt_tokens_details else 0
        usage = {'in': u.prompt_tokens - cached, 'cache_write': 0, 'cache_read': cached,
                 'out': u.completion_tokens}
        if r.choices[0].finish_reason == 'length':
            raise RuntimeError('hit max_tokens')
        return json.loads(r.choices[0].message.content), usage, r.model


def _valid(ans: dict) -> bool:
    return bool(ans.get('tags')) and all(t in tag_names() for t in ans['tags'])


async def run(model: str, variant: str, uids: list[str], sample: int = 0,
              budget: float = BUDGET_USD) -> dict[str, dict]:
    cached = load_answers()
    caller = Caller(model, budget)
    sem = asyncio.Semaphore(CONCURRENCY)
    system = system_prompt(variant)
    results: dict[str, dict] = {}
    stats = Counter()

    async def one(uid: str) -> None:
        user = user_prompt(variant, row(uid))
        key = _key(model, variant, system, user, sample)
        if key in cached:
            results[uid] = cached[key]['answer']
            stats['cached'] += 1
            return
        async with sem:
            for _attempt in range(2):
                try:
                    ans, _ = await caller.call(variant, system, user)
                except BudgetExceeded:
                    stats['budget'] += 1
                    return
                except Exception as e:  # noqa: BLE001 - logged and counted
                    print(f'  {uid}: {type(e).__name__}: {str(e)[:150]}', file=sys.stderr)
                    stats['error'] += 1
                    continue
                if _valid(ans):
                    break
                stats['invalid'] += 1
            else:
                return
        # Tags in order, without repeats.
        ans['tags'] = list(dict.fromkeys(ans['tags']))
        _append(answers_path(), {'key': key, 'model': model, 'variant': variant, 'uid': uid,
                                 'sample': sample, 'answer': ans})
        results[uid] = ans
        stats['new'] += 1

    await asyncio.gather(*(one(u) for u in uids))
    print(f'{model} {variant} sample={sample}: {dict(stats)}; '
          f'spent so far ${caller.spent:.3f} of ${caller.budget:.2f}')
    return results


def answers_for(model: str, variant: str, uids: list[str], sample: int = 0) -> dict[str, dict]:
    """Cached answers only (no calls)."""
    cached = load_answers()
    system = system_prompt(variant)
    out = {}
    for uid in uids:
        k = _key(model, variant, system, user_prompt(variant, row(uid)), sample)
        if k in cached:
            out[uid] = cached[k]['answer']
    return out


# --- Scoring ---------------------------------------------------------------

def cohen_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    if not n:
        return float('nan')
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in ca) / n / n
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def fleiss_kappa(rows: list[list[str]]) -> float:
    cats = sorted({x for r in rows for x in r})
    m = len(rows[0])
    counts = np.array([[r.count(c) for c in cats] for r in rows], dtype=float)
    p_i = ((counts ** 2).sum(1) - m) / (m * (m - 1))
    p_j = counts.sum(0) / counts.sum()
    pe = (p_j ** 2).sum()
    return float((p_i.mean() - pe) / (1 - pe)) if pe < 1 else 1.0


def pair_main(gold: list[str], pred: dict) -> tuple[str, str]:
    """(gold, predicted) main tags. With several gold tags, the gold side is
    the one the model chose if it is among them, else the first."""
    p = pred['tags'][0]
    return (p if p in gold else gold[0]), p


def score_one(uids: list[str], preds: dict[str, dict], boot: int = 1000) -> dict[str, Any]:
    it = _by_uid()
    uids = [u for u in uids if u in preds]
    if not uids:
        return {'n': 0}
    pairs = [pair_main(it.loc[u].gold, preds[u]) for u in uids]
    hit = np.array([g == p for g, p in pairs], dtype=float)
    rng = np.random.default_rng(SEED)
    accs, kaps = [], []
    for _ in range(boot):
        idx = rng.integers(0, len(pairs), len(pairs))
        accs.append(hit[idx].mean())
        kaps.append(cohen_kappa([pairs[i][0] for i in idx], [pairs[i][1] for i in idx]))
    return {'n': len(uids), 'acc': hit.mean(), 'acc_ci': np.nanpercentile(accs, [2.5, 97.5]),
            'kappa': cohen_kappa(*map(list, zip(*pairs))),
            'kappa_ci': np.nanpercentile(kaps, [2.5, 97.5])}


def per_tag(uids: list[str], preds: dict[str, dict]) -> pd.DataFrame:
    it = _by_uid()
    pairs = [pair_main(it.loc[u].gold, preds[u]) for u in uids if u in preds]
    rows = []
    for t in tag_names():
        tp = sum(g == t and p == t for g, p in pairs)
        fp = sum(g != t and p == t for g, p in pairs)
        fn = sum(g == t and p != t for g, p in pairs)
        rows.append({'tag': t, 'gold_n': tp + fn, 'pred_n': tp + fp,
                     'precision': tp / (tp + fp) if tp + fp else np.nan,
                     'recall': tp / (tp + fn) if tp + fn else np.nan})
    return pd.DataFrame(rows)


def is_trivial_agreement(uid: str) -> bool:
    return _by_uid().loc[uid].gold == ['agreement']


# --- Commands --------------------------------------------------------------

def _uids(split_name: str) -> list[str]:
    it = items()
    if split_name == 'all':
        return list(it[it.split.isin(['dev', 'test'])].uid)
    return list(it[it.split == split_name].uid)


def _tokens(s: str) -> float:
    return len(s) / 3.6   # rough; Claude's tokenizer gives ~1.6x more for this text


def cmd_splits(_a) -> None:
    it = items()
    print(it.split.value_counts().to_string(), '\n')
    g = it[it.split.isin(['dev', 'test'])]
    print(g.assign(tag=g.gold.str[0]).groupby(['tag', 'split']).size().unstack(fill_value=0)
          .to_string())


def cmd_show(a) -> None:
    print('=== SYSTEM ===\n' + system_prompt(a.variant))
    print('\n=== USER ===\n' + user_prompt(a.variant, row(a.uid)))
    print('\n=== SCHEMA ===\n' + json.dumps(schema(a.variant)))


def cmd_estimate(_a) -> None:
    groups = {'dev': _uids('dev'), 'test': _uids('test'), 'unlabelled': _uids('unlabelled')}
    rows = []
    for v in VARIANTS:
        sys_t = _tokens(system_prompt(v))
        sample = {g: u[:: max(1, len(u) // 60)] for g, u in groups.items()}
        for g, uids in groups.items():
            if not uids:
                continue
            user_t = np.mean([_tokens(user_prompt(v, row(u))) for u in sample[g]])
            out_t = 200 if _spec(v)['rationale'] else 90
            for m in MODELS:
                p = PRICES[m]
                per = sys_t * p['cache_read'] + user_t * p['in'] + out_t * p['out']
                rows.append({'variant': v, 'split': g, 'model': m, 'n': len(uids),
                             'system_tok': round(sys_t), 'user_tok': round(user_t),
                             'usd_per_call': per, 'usd_total': per * len(uids)})
    pd.set_option('display.width', 200)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print('\nRough: Claude counts ~1.6x more tokens than this estimate (its cache keeps the '
          'cost close).')


def cmd_ping(a) -> None:
    it = items()
    dev = it[it.split == 'dev']
    if dev.empty:
        raise ConfigError('no dev comments: the notes label nothing in these PDFs.')
    res = asyncio.run(run(a.model, 'P2', [dev.iloc[0].uid], sample=99, budget=a.budget))
    print(json.dumps(res, indent=1))
    print(f'spent on {a.model}: ${spent(a.model):.4f}')


def cmd_run(a) -> None:
    require_key(a.model)
    uids = _uids(a.split)
    if a.limit:
        uids = random.Random(SEED).sample(uids, min(a.limit, len(uids)))
    asyncio.run(run(a.model, a.variant, uids, sample=a.sample, budget=a.budget))


def cmd_score(a) -> None:
    uids = _uids(a.split)
    rows = []
    for m in MODELS:
        for v in VARIANTS:
            p = answers_for(m, v, uids)
            if not p:
                continue
            nontriv = [u for u in uids if not is_trivial_agreement(u)]
            s, s2 = score_one(uids, p), score_one(nontriv, p)
            rows.append({'model': m, 'variant': v, 'n': s['n'],
                         'acc': s['acc'], 'kappa': s['kappa'],
                         'acc_ci': '%.2f-%.2f' % tuple(s['acc_ci']),
                         'n_excl_agree': s2['n'], 'acc_excl_agree': s2.get('acc'),
                         'kappa_excl_agree': s2.get('kappa')})
    if not rows:
        print(f'no cached answers for split {a.split!r}; run `mlr-kg label run` first.')
        return
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f'{x:.3f}'))


def cmd_export(a) -> None:
    it = _by_uid()
    keep = it.split.isin(['unlabelled', 'test']) if a.only_needed else it.split.notna()
    uids = list(it.index[keep])
    preds = answers_for(a.model, a.variant, uids)
    if not preds:
        raise ConfigError(f'no cached {a.model} {a.variant} answers; run `mlr-kg label run` first.')
    # The other model's tags, so disagreements can be reviewed first.
    other = [m for m in MODELS if m != a.model][0]
    second = answers_for(other, a.variant, uids)
    tmp = CFG.llm_labels.with_name(CFG.llm_labels.name + '.tmp')
    with tmp.open('w') as f:
        for u in uids:
            if u not in preds:
                continue
            ans = preds[u]
            f.write(json.dumps({'target': u, 'model': a.model, 'variant': a.variant,
                                'tags': ans['tags'], 'note': ans['note'],
                                'confidence': ans['confidence'],
                                'suggested_new_tag': ans.get('suggested_new_tag'),
                                f'{other}_tags': second.get(u, {}).get('tags'),
                                'my_tags': list(it.loc[u].gold),
                                'split': it.loc[u].split}, ensure_ascii=False) + '\n')
    tmp.replace(CFG.llm_labels)     # whole file or nothing
    missing = len(uids) - len(preds)
    print(f'{len(preds)} labels -> {CFG.llm_labels}'
          + (f'  ({missing} comments have no cached answer)' if missing else ''))


def add_commands(sub) -> None:
    """The `label` subcommands, added to the CLI's `label` parser."""
    p = sub.add_parser('splits', help='free: how many comments are dev / test / unlabelled')
    p.set_defaults(label_fn=cmd_splits)
    p = sub.add_parser('show', help='free: print the prompt for one comment')
    p.add_argument('variant', choices=VARIANTS)
    p.add_argument('uid')
    p.set_defaults(label_fn=cmd_show)
    p = sub.add_parser('estimate', help='free: token and cost estimate per variant')
    p.set_defaults(label_fn=cmd_estimate)
    for name, f, help_ in (('ping', cmd_ping, 'one paid call, to test a model and key'),
                           ('run', cmd_run, 'paid: label a split (cached answers are free)')):
        p = sub.add_parser(name, help=help_)
        p.add_argument('--model', choices=MODELS, required=True)
        p.add_argument('--budget', type=float, default=BUDGET_USD,
                       help=f'stop once this model\'s total spend would pass this (USD, default {BUDGET_USD:g})')
        if name == 'run':
            p.add_argument('--variant', choices=VARIANTS, required=True)
            p.add_argument('--split', choices=['dev', 'test', 'all', 'unlabelled'], required=True)
            p.add_argument('--sample', type=int, default=0,
                           help='repeat number; a new number re-asks (for self-consistency)')
            p.add_argument('--limit', type=int, help='label only this many (seeded sample)')
        p.set_defaults(label_fn=f)
    p = sub.add_parser('score', help='free: agreement with the human labels, from cached answers')
    p.add_argument('--split', choices=['dev', 'test', 'all'], default='dev')
    p.set_defaults(label_fn=cmd_score)
    p = sub.add_parser('export', help='free: write cached answers to --llm-labels')
    p.add_argument('--model', choices=MODELS, required=True)
    p.add_argument('--variant', choices=VARIANTS, required=True)
    p.add_argument('--all-comments', dest='only_needed', action='store_false',
                   help='include dev comments too (default: unlabelled and test only)')
    p.add_argument('--overwrite', action='store_true', help='replace an existing --llm-labels file')
    p.set_defaults(label_fn=cmd_export)
