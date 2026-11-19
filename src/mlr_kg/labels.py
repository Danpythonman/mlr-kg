"""Comment labels: the human ones from notes.jsonl, the LLM ones from
llm_labels.jsonl. Both files are only ever read.

notes.jsonl is an append-only event log, one JSON object per line:

    {"event": "add",    "note_id", "at", "by", "target", "text", "tags", "snapshot"}
    {"event": "edit",   "note_id", "at", "by", "text", "tags"}
    {"event": "delete", "note_id", "at", "by"}
    {"event": "torn_line", ...}     (a marker for a write cut off by a crash)

The current notes are the log replayed in order. `target` is a comment uid
(see annotations.uid). Tags are merged onto the tags in taxonomy.yaml.

Human labels ("gold") cover every comment in a PDF that has at least one
note: noted comments get their merged tags; comments in those PDFs left
without a note get `no tag`. Gold comments are split into dev and test
halves, stratified by first tag, with a fixed seed.
"""

from __future__ import annotations

import json
import random
from functools import cache
from pathlib import Path

import pandas as pd
import yaml

BUNDLED_TAXONOMY = Path(__file__).with_name('taxonomy.yaml')
TAXONOMY = BUNDLED_TAXONOMY      # replaced by use_taxonomy() for --taxonomy
SEED = 0
TEST_FRACTION = 0.5


def use_taxonomy(path: Path | None) -> None:
    global TAXONOMY
    TAXONOMY = path or BUNDLED_TAXONOMY
    taxonomy.cache_clear()


@cache
def taxonomy() -> dict:
    return yaml.safe_load(TAXONOMY.read_text())['tags']


def tag_names() -> list[str]:
    return list(taxonomy())


def merge_map() -> dict[str, str]:
    return {old: new for new, t in taxonomy().items() for old in t['from']}


def replay_notes(path: Path) -> pd.DataFrame:
    """Current notes: note_id, target, text, tags, created, updated, by."""
    notes: dict[str, dict] = {}
    lines = path.read_text(encoding='utf-8').split('\n')
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            nxt = lines[i + 1] if i + 1 < len(lines) else ''
            if i == len(lines) - 1 or '"torn_line"' in nxt:
                continue               # a torn write
            raise SystemExit(f'mlr-kg: error: {path}: line {i + 1} is not valid JSON') from None
        ev = e.get('event')
        if ev == 'torn_line':
            continue
        nid = e['note_id']
        if ev == 'add':
            notes[nid] = {'note_id': nid, 'target': e['target'], 'text': e.get('text', ''),
                          'tags': list(e.get('tags', [])), 'created': e['at'],
                          'updated': e['at'], 'by': e.get('by')}
        elif ev == 'edit' and nid in notes:
            notes[nid].update(text=e['text'], tags=list(e['tags']), updated=e['at'])
        elif ev == 'delete':
            notes.pop(nid, None)
    cols = ['note_id', 'target', 'text', 'tags', 'created', 'updated', 'by']
    return pd.DataFrame(list(notes.values()), columns=cols)


def original_tags(notes: pd.DataFrame) -> dict[str, list[str]]:
    """Per comment, the tags as written in the notes (before merging)."""
    out = {}
    for uid, g in notes.sort_values('created').groupby('target'):
        out[uid] = list(dict.fromkeys(t for ts in g.tags for t in ts))
    return out


def gold(ann: pd.DataFrame, notes: pd.DataFrame) -> pd.DataFrame:
    """uid, gold (merged tags), my_note, split for every human-labelled comment."""
    mm = merge_map()
    unknown = {t for ts in notes.tags for t in ts} - set(mm)
    if unknown:
        raise SystemExit(f'mlr-kg: error: tags in the notes that taxonomy.yaml does not '
                         f'list under any `from`: {sorted(unknown)}')
    rows = []
    for uid, g in notes.sort_values('created').groupby('target', sort=False):
        tags = list(dict.fromkeys(mm[t] for ts in g.tags for t in ts))
        texts = [t for t in g.text if t.strip()]
        rows.append({'uid': uid, 'gold': tags, 'my_note': ' | '.join(texts)})
    df = pd.DataFrame(rows, columns=['uid', 'gold', 'my_note'])
    noted_files = set(ann[ann.uid.isin(df.uid)].file)
    skipped = ann[ann.file.isin(noted_files) & ~ann.uid.isin(df.uid)].uid
    df = pd.concat([df, pd.DataFrame({'uid': skipped, 'gold': [['no tag']] * len(skipped),
                                      'my_note': ''})], ignore_index=True)
    df['split'] = _split(df)
    df.loc[df.gold.map(len) == 0, 'split'] = 'text-only'
    return df


def _split(g: pd.DataFrame) -> pd.Series:
    rng = random.Random(SEED)
    out = {}
    for _, grp in g.groupby(g.gold.str[0]):
        uids = sorted(grp.uid)
        rng.shuffle(uids)
        k = int(round(len(uids) * TEST_FRACTION)) if len(uids) > 1 else 0
        out.update({u: 'test' for u in uids[:k]})
        out.update({u: 'dev' for u in uids[k:]})
    return g.uid.map(out)


def items(ann: pd.DataFrame, notes_path: Path) -> pd.DataFrame:
    """Every comment with its thread links and human labels (if any)."""
    df = ann.copy()
    by_xref = {(f, x): u for f, x, u in zip(df.file, df.xref, df.uid)}
    df['parent_uid'] = [by_xref.get((f, int(x))) if pd.notna(x) else None
                        for f, x in zip(df.file, df.in_reply_to)]
    df['root_uid'] = df.parent_uid.fillna(df.uid)
    df = df.merge(gold(ann, replay_notes(notes_path)), on='uid', how='left')
    df['split'] = df.split.fillna('unlabelled')
    df['gold'] = df.gold.map(lambda g: list(g) if isinstance(g, list) else [])
    df['my_note'] = df.my_note.fillna('')
    return df.sort_values(['file', 'page', 'created', 'uid'], kind='stable').reset_index(drop=True)


def llm_labels(path: Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True, dtype=False)
