"""Knowledge graph of the MLR material: passages, claims, comments, concerns.

Built from the source PDFs and the comment labels, with no LLM calls:

    Brand -[HAS_PIECE]-> Piece -[HAS_VERSION]-> Version -[NEXT]-> Version
    Version -[HAS_PASSAGE]-> Passage -[HAS_CLAIM]-> Claim
    Comment -[ON]-> Claim            where its mark (or nearest text) is
    Comment -[ON_PAGE]-> Version     marks with no text near them
    Reply -[REPLIES_TO]-> Comment    plus AGREES / DISAGREES from the tags
    Comment -[BY]-> Person
    Comment -[RAISES]-> Concern      for issue tags only (see CONCERN_TAGS)
    Comment -[PROPOSES]-> Edit -[APPLIED_IN | NOT_APPLIED_IN]-> Version
    Claim -[SAME_AS | REVISED_TO]-> Claim   in the next version
    Claim -[CITES]-> Reference       inline [n/Source] citations and
                                     reference-link annotations
    Comment -[MENTIONS]-> Reference  job codes such as AB12CDEF34567
    Comment -[ABOUT_REFERENCE]-> Reference   its mark covers a citation
    Comment -[RESOLVES]-> Comment    an author's "changed X as requested"

Definitions:
- A Passage is a PyMuPDF text block, read from words with doubled text
  layers removed (as in app/raw.py). A Claim is a sentence of a passage;
  inline citations are cut out of its text and become CITES edges.
- ON edges carry `method` (`mark`: words under the mark; `nearest`: the
  closest word, for carets, sticky notes and boxes without text inside)
  and `share` (fraction of the marked words in that claim).
- A comment's tags are the human ones where it was labelled by hand
  (--notes), otherwise the LLM's (--llm-labels). Concern statements are, for now, the LLM's
  one-line note on the comment (`statement_source = llm_note`).
- Claims are matched to the next version by fuzzy text similarity:
  SAME_AS at >= SAME_SCORE, REVISED_TO at >= REVISED_SCORE. Each claim
  gets `next_status` (unchanged / revised / removed / last version) and
  `prev_status` (added when nothing in the previous version matches).
- Edits come from strikeouts (delete, or replace when the comment holds
  replacement text) and carets (insert). Whether an edit was applied is
  read off the matched claim in the next version.

Output in --out: nodes.parquet (id, kind, label, text, file, page, x0..y1,
attrs) and edges.parquet (src, rel, dst, attrs); attrs are JSON strings.

Run through the `mlr-kg` command (see cli.py and the README).
"""

from __future__ import annotations

import functools
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pymupdf as fitz
from rapidfuzz import fuzz, process

from . import annotations as raw
from . import labels
from .config import Config

CFG: Config = None  # type: ignore[assignment]   set by configure()


def configure(cfg: Config) -> None:
    global CFG
    CFG = cfg
    raw.configure(cfg)
    labels.use_taxonomy(cfg.taxonomy)
    items.cache_clear()


@functools.cache
def items() -> pd.DataFrame:
    return labels.items(raw.load_annotations(), CFG.notes)

CONCERN_TAGS = {'wording', 'overclaiming', 'adding text', 'removing text',
                'information quality', 'reference request', 'reference problem'}
SAME_SCORE = 95
REVISED_SCORE = 70
MIN_MATCH_WORDS = 3       # shorter claims are too generic to match across versions
NEAREST_PT = 60           # nearest-word anchoring gives up beyond this distance

ABBREV = {'e.g.', 'i.e.', 'etc.', 'vs.', 'dr.', 'mr.', 'mrs.', 'ms.', 'no.', 'p.', 'pp.',
          'fig.', 'al.', 'approx.', 'inc.', 'ltd.', 'st.', 'u.s.'}
BULLETS = {'●', '•', '▪', '◦', '-', '–', '○', '■', '*'}
CITATION = re.compile(r'^\[\d+/')
REF_LINK = re.compile(r'^\s*(.+?)\s*\(v\d+(?:\.\d+)?\)', re.S)
JOB_CODE = re.compile(r'\b(?:[A-Z]{2}\d{2}[A-Z]{3,4}\d{5,6}|[A-Z]{2,4}-[A-Z]-\d{6})\b')
VERSION = re.compile(r'\bv(\d+)\.(\d+)', re.I)


def _id(*parts) -> str:
    return hashlib.sha1('#'.join(map(str, parts)).encode()).hexdigest()[:12]


def _norm(s: str) -> str:
    s = s.lower().replace('(r)', '®').replace('(tm)', '™').replace('®', ' ').replace('™', ' ')
    return ' '.join(re.sub(r'[^\w\s]', ' ', s).split())


# --- Passages and claims ---------------------------------------------------

QUOTES = str.maketrans({'’': "'", '‘': "'", '“': '"', '”': '"'})
DUP_BLOCK_SHARE = 0.6


def page_words(page: fitz.Page) -> list[tuple]:
    """Words block by block in reading order, without duplicate text layers.

    Some PDFs carry a second, slightly offset copy of the text. Dropping
    duplicate words one at a time mixes the two copies, so a whole block is
    dropped when most of its words repeat an earlier block's words.
    """
    blocks: dict[int, list[tuple]] = defaultdict(list)
    for w in page.get_text('words'):
        blocks[w[5]].append(w)
    seen: dict[str, list[fitz.Rect]] = defaultdict(list)
    kept: list[list[tuple]] = []
    for bno in sorted(blocks):
        ws = blocks[bno]
        dup = sum(any((fitz.Rect(w[:4]) & b).get_area() > 0.5 * fitz.Rect(w[:4]).get_area()
                      for b in seen[w[4].translate(QUOTES)]) for w in ws)
        if dup >= DUP_BLOCK_SHARE * len(ws):
            continue
        for w in ws:
            seen[w[4].translate(QUOTES)].append(fitz.Rect(w[:4]))
        kept.append(sorted(ws, key=lambda w: (w[6], w[7])))
    # Reading order: top to bottom, then left to right, by block.
    kept.sort(key=lambda ws: (round(min(w[1] for w in ws)), min(w[0] for w in ws)))
    return [w for ws in kept for w in ws]


def _ends_sentence(word: str, nxt: str | None, depth: int) -> bool:
    if depth > 0 or nxt is None:
        return nxt is None
    w = word.rstrip('"\'”’)')
    if not w.endswith(('.', '!', '?', ':')) or w.lower() in ABBREV:
        return False
    nxt = nxt.replace('\u200b', '')
    if w.endswith(':'):
        return nxt[:1] in BULLETS
    return nxt[:1].isupper() or nxt[:1].isdigit() or nxt[:1] in BULLETS or nxt[:1] in '["“'


def _is_bullet(w: tuple) -> bool:
    """A bullet glyph opening a line (a dash mid-line is punctuation)."""
    return w[4].replace('\u200b', '') in BULLETS and (len(w) < 8 or w[7] == 0)


def split_sentences(words: list[tuple]) -> list[list[int]]:
    """Word-index groups, one per sentence; bullets start a new sentence.
    Citations right after a sentence's end stay with that sentence."""
    out, cur, depth, ending = [], [], 0, False
    for i, w in enumerate(words):
        t = w[4]
        if _is_bullet(w) and cur and depth == 0:
            out.append(cur)
            cur, ending = [], False
        cur.append(i)
        depth = max(depth + t.count('[') - t.count(']'), 0)
        nxt = words[i + 1][4] if i + 1 < len(words) else None
        if depth == 0 and (ending or _ends_sentence(t, nxt, depth)):
            if nxt is not None and CITATION.match(nxt):
                ending = True        # wait for the citations to close
                continue
            out.append(cur)
            cur, ending = [], False
    if cur:
        out.append(cur)
    return out


def _cut_citations(tokens: list[str]) -> tuple[list[str], list[str]]:
    """Remove [n/Source …] groups from a token list; return (text, citations)."""
    text, cites, buf, depth = [], [], [], 0
    for t in tokens:
        if not buf and CITATION.match(t):
            depth = 0
        if buf or CITATION.match(t):
            buf.append(t)
            depth += t.count('[') - t.count(']')
            if depth <= 0:
                cites.append(' '.join(buf))
                buf = []
            continue
        text.append(t)
    if buf:
        cites.append(' '.join(buf))
    return text, cites


def citation_source(c: str) -> tuple[str, str]:
    """'[1/Product Leaflet – Dosing p.2 ¶3-4]' -> ('product leaflet dosing', 'p.2 ¶3-4')."""
    body = c.strip('[] ').split('/', 1)[-1]
    m = re.search(r'\s(p\.\s?\d|¶)', body)
    title, loc = (body[:m.start()], body[m.start():].strip()) if m else (body, '')
    return _norm(title), loc


# --- Build -----------------------------------------------------------------

class Graph:
    def __init__(self):
        self.nodes: dict[str, dict] = {}
        self.edges: list[dict] = []

    def node(self, nid: str, kind: str, label: str, **kw) -> str:
        if nid not in self.nodes:
            attrs = kw.pop('attrs', {})
            self.nodes[nid] = {'id': nid, 'kind': kind, 'label': label, **kw, 'attrs': attrs}
        return nid

    def edge(self, src: str, rel: str, dst: str, **attrs) -> None:
        self.edges.append({'src': src, 'rel': rel, 'dst': dst, 'attrs': attrs})


def version_order(files: list[str]) -> list[str]:
    def key(f):
        m = VERSION.search(Path(f).name)
        return (int(m[1]), int(m[2]), f) if m else (99, 99, f)
    return sorted(files, key=key)


def comment_tags() -> tuple[dict[str, list[str]], dict[str, str]]:
    """Tags per comment, and whose they are: 'mine' (human, from --notes)
    where it was labelled by hand, else 'llm' (from --llm-labels)."""
    tags, source = {}, {}
    for line in CFG.llm_labels.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            tags[r['target']], source[r['target']] = r['tags'], 'llm'
    it = items()
    for u, g in zip(it.uid, it.gold):
        if g:
            tags[u], source[u] = list(g), 'mine'
    return tags, source


def llm_notes() -> dict[str, str]:
    out = {}
    for line in CFG.llm_labels.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r['target']] = r['note']
    return out


def build() -> Graph:
    g = Graph()
    ann = raw.load_annotations()
    (tags, tag_source), notes = comment_tags(), llm_notes()
    my_notes = dict(zip(items().uid, items().my_note))

    # Structure: brand / piece / version.
    files_by_piece = defaultdict(list)
    for p in raw.pdf_files():
        rel = p.relative_to(raw.ROOT).as_posix()
        files_by_piece[Path(rel).parent.as_posix()].append(rel)
    version_of: dict[str, str] = {}
    next_file: dict[str, str] = {}
    for piece, files in files_by_piece.items():
        brand = piece.split(' veeva examples')[0]
        b = g.node('brand:' + _id(brand), 'Brand', brand)
        pc = g.node('piece:' + _id(piece), 'Piece', piece.split('/', 1)[-1], file=piece)
        g.edge(b, 'HAS_PIECE', pc)
        ordered = version_order(files)
        for i, f in enumerate(ordered):
            m = VERSION.search(Path(f).name)
            v = g.node('version:' + _id(f), 'Version', Path(f).stem, file=f,
                       attrs={'version': f'v{m[1]}.{m[2]}' if m else None, 'order': i})
            version_of[f] = v
            g.edge(pc, 'HAS_VERSION', v)
            if i:
                g.edge(version_of[ordered[i - 1]], 'NEXT', v)
                next_file[ordered[i - 1]] = f

    # Passages, claims, and a word -> claim index per page for anchoring.
    claims_by_file: dict[str, list[dict]] = defaultdict(list)
    page_index: dict[tuple[str, int], tuple[list[tuple], list[str | None]]] = {}
    refs: dict[str, str] = {}

    def reference(key: str, label: str, kind: str) -> str:
        if key not in refs:
            refs[key] = g.node('ref:' + _id(kind, key), 'Reference', label, attrs={'ref_kind': kind})
        return refs[key]

    for f in version_of:
        with fitz.open(raw.ROOT / f) as doc:
            for page in doc:
                pno = page.number + 1
                words = page_words(page)
                owner: list[str | None] = [None] * len(words)
                blocks = defaultdict(list)
                for i, w in enumerate(words):
                    blocks[w[5]].append(i)
                for bno, idxs in sorted(blocks.items(), key=lambda kv: kv[1][0]):
                    bw = [words[i] for i in idxs]
                    text = ' '.join(w[4] for w in bw)
                    r = fitz.Rect(min(w[0] for w in bw), min(w[1] for w in bw),
                                  max(w[2] for w in bw), max(w[3] for w in bw))
                    psg = g.node('passage:' + _id(f, pno, bno), 'Passage', text[:80], text=text,
                                 file=f, page=pno, x0=r.x0, y0=r.y0, x1=r.x1, y1=r.y1)
                    g.edge(version_of[f], 'HAS_PASSAGE', psg)
                    for k, sent in enumerate(split_sentences(bw)):
                        toks = [bw[j][4] for j in sent]
                        body, cites = _cut_citations(toks)
                        stext = ' '.join(body).replace('\u200b', '').strip()
                        sw = [bw[j] for j in sent]
                        sr = fitz.Rect(min(w[0] for w in sw), min(w[1] for w in sw),
                                       max(w[2] for w in sw), max(w[3] for w in sw))
                        if not stext:
                            # Citations alone: they belong to the sentence before.
                            prev = claims_by_file[f][-1] if claims_by_file[f] else None
                            for j in sent:
                                owner[idxs[j]] = prev['id'] if prev else None
                            if prev:
                                for c in cites:
                                    src, loc = citation_source(c)
                                    g.edge(prev['id'], 'CITES', reference(src, c.split('/', 1)[-1].split(' p.')[0].strip('] '), 'citation'),
                                           locator=loc, how='inline')
                            continue
                        cid = g.node('claim:' + _id(f, pno, bno, k), 'Claim', stext[:80], text=stext,
                                     file=f, page=pno, x0=sr.x0, y0=sr.y0, x1=sr.x1, y1=sr.y1)
                        g.edge(psg, 'HAS_CLAIM', cid)
                        claims_by_file[f].append({'id': cid, 'text': stext, 'norm': _norm(stext)})
                        for j in sent:
                            owner[idxs[j]] = cid
                        for c in cites:
                            src, loc = citation_source(c)
                            g.edge(cid, 'CITES', reference(src, c.split('/', 1)[-1].split(' p.')[0].strip('] '), 'citation'),
                                   locator=loc, how='inline')
                page_index[(f, pno)] = (words, owner)

    # Claims across versions.
    next_claim: dict[str, tuple[str, float]] = {}
    matched_next: set[str] = set()
    for f, nf in next_file.items():
        a = [c for c in claims_by_file[f] if len(c['norm'].split()) >= MIN_MATCH_WORDS]
        b = [c for c in claims_by_file[nf] if len(c['norm'].split()) >= MIN_MATCH_WORDS]
        if not a or not b:
            continue
        scores = process.cdist([c['norm'] for c in a], [c['norm'] for c in b],
                               scorer=fuzz.ratio, workers=-1)
        for i, c in enumerate(a):
            j = int(np.argmax(scores[i]))
            s = float(scores[i, j])
            if s >= REVISED_SCORE:
                rel = 'SAME_AS' if s >= SAME_SCORE else 'REVISED_TO'
                g.edge(c['id'], rel, b[j]['id'], score=round(s, 1))
                next_claim[c['id']] = (b[j]['id'], s)
                matched_next.add(b[j]['id'])
    for f, cl in claims_by_file.items():
        for c in cl:
            n = g.nodes[c['id']]
            if f not in next_file:
                n['attrs']['next_status'] = 'last version'
            elif c['id'] in next_claim:
                n['attrs']['next_status'] = 'unchanged' if next_claim[c['id']][1] >= SAME_SCORE else 'revised'
            elif len(c['norm'].split()) >= MIN_MATCH_WORDS:
                n['attrs']['next_status'] = 'removed'
            if f in next_file.values() and len(c['norm'].split()) >= MIN_MATCH_WORDS:
                n['attrs']['prev_status'] = 'matched' if c['id'] in matched_next else 'added'

    # Comments.
    by_xref = {(f, x): u for f, x, u in zip(ann.file, ann.xref, ann.uid)}
    anchor: dict[str, list[str]] = {}
    for a in ann.itertuples():
        reply_to = by_xref.get((a.file, int(a.in_reply_to))) if pd.notna(a.in_reply_to) else None
        t = tags.get(a.uid, [])
        c = g.node(a.uid, 'Reply' if reply_to else 'Comment',
                   (a.content or f'({a.type}, no text)')[:80], text=a.content, file=a.file,
                   page=int(a.page), x0=a.x0, y0=a.y0, x1=a.x1, y1=a.y1,
                   attrs={'type': a.type, 'author': a.author, 'created': str(a.created),
                          'marked_text': a.marked_text, 'tags': t,
                          'tags_by': tag_source.get(a.uid),
                          'my_note': my_notes.get(a.uid, '')})
        if a.author:
            g.edge(c, 'BY', g.node('person:' + _id(a.author), 'Person', a.author))
        if reply_to:
            g.edge(c, 'REPLIES_TO', reply_to)
            if t and t[0] == 'agreement':
                g.edge(c, 'AGREES', reply_to)
            if 'disagreement' in t:
                g.edge(c, 'DISAGREES', reply_to)
        else:
            anchor[a.uid] = anchor_comment(a, page_index.get((a.file, int(a.page))))
            hits = anchor[a.uid]
            if hits and hits[0][0] is not None:
                for cid, share, how in hits:
                    g.edge(c, 'ON', cid, method=how, share=round(share, 2))
            else:
                g.edge(c, 'ON_PAGE', version_of[a.file], page=int(a.page))
        # A mark over a citation marker is about that reference.
        for cit in re.findall(r'\[\d+/[^\]]+\]?', a.marked_text or ''):
            key, _ = citation_source(cit)
            if key in refs:
                g.edge(c, 'ABOUT_REFERENCE', refs[key])
        concern_tags = [x for x in t if x in CONCERN_TAGS]
        if concern_tags:
            k = g.node('concern:' + a.uid, 'Concern', notes.get(a.uid) or a.content[:80],
                       text=notes.get(a.uid, ''),
                       attrs={'type': concern_tags[0], 'types': concern_tags,
                              'statement_source': 'llm_note' if a.uid in notes else 'comment'})
            g.edge(c, 'RAISES', k)
        content = a.content or ''
        m = REF_LINK.match(content)
        if m and (not t or t[0] in ('references', 'no tag')):
            ref = reference(_norm(m[1]), m[1].strip(), 'citation')
            for cid, _, how in anchor.get(a.uid, []):
                if cid is not None:
                    g.edge(cid, 'CITES', ref, how='annotation', annotation=a.uid,
                           passage=content[m.end():].strip()[:500])
        for code in set(JOB_CODE.findall(content)):
            g.edge(c, 'MENTIONS', reference(code, code, 'job code'))

    add_edits(g, ann, anchor, next_claim, next_file, version_of)
    add_resolutions(g, ann, tags, anchor, next_claim)
    merge_references(g)
    return g


def merge_references(g: Graph) -> None:
    """One node per source: citations name a source in slightly different
    ways (cut-off titles, 'ScienceDirect' vs 'Science Direct')."""
    refs = [n for n in g.nodes.values()
            if n['kind'] == 'Reference' and n['attrs']['ref_kind'] == 'citation']
    key = {n['id']: _norm(n['label']).replace(' ', '') for n in refs}
    canon: dict[str, str] = {}
    for n in sorted(refs, key=lambda n: -len(key[n['id']])):
        k = key[n['id']]
        for c in dict.fromkeys(canon.values()):
            ck = key[c]
            if ck.startswith(k) or fuzz.ratio(k, ck) >= 90:
                canon[n['id']] = c
                g.nodes[c]['attrs'].setdefault('aliases', []).append(n['label'])
                break
        else:
            canon[n['id']] = n['id']
    for e in g.edges:
        e['src'] = canon.get(e['src'], e['src'])
        e['dst'] = canon.get(e['dst'], e['dst'])
    for nid, c in canon.items():
        if nid != c:
            del g.nodes[nid]


def anchor_comment(a, index) -> list[tuple[str | None, float, str]]:
    """[(claim id, share of marked words, method)] for a top-level comment."""
    if index is None:
        return []
    words, owner = index
    if a.type in raw.MARKUP_TYPES:
        with fitz.open(raw.ROOT / a.file) as doc:
            page = doc[int(a.page) - 1]   # keep the page alive while reading its annots
            annot = next((x for x in page.annots() if x.xref == a.xref), None)
            v = annot.vertices if annot else None
        rects = [fitz.Quad(v[i:i + 4]).rect for i in range(0, len(v), 4)] if v else [
            fitz.Rect(a.x0, a.y0, a.x1, a.y1)]
    else:
        rects = [fitz.Rect(a.x0, a.y0, a.x1, a.y1)]
    inside = [i for i, w in enumerate(words)
              if any(r.x0 <= (w[0] + w[2]) / 2 <= r.x1 and r.y0 <= (w[1] + w[3]) / 2 <= r.y1 for r in rects)]
    counts = defaultdict(int)
    for i in inside:
        counts[owner[i]] += 1
    counts.pop(None, None)
    if counts:
        n = sum(counts.values())
        return [(cid, k / n, 'mark') for cid, k in sorted(counts.items(), key=lambda kv: -kv[1])]
    # Nothing under the mark: the nearest word, if it is close.
    r = rects[0]
    cx, cy = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
    best, dist = None, NEAREST_PT
    for i, w in enumerate(words):
        if owner[i] is None:
            continue
        dx = max(w[0] - cx, 0, cx - w[2])
        dy = max(w[1] - cy, 0, cy - w[3])
        d = (dx * dx + dy * dy) ** 0.5
        if d < dist:
            best, dist = i, d
    return [(owner[best], 1.0, 'nearest')] if best is not None else [(None, 0.0, 'none')]


def add_edits(g: Graph, ann, anchor, next_claim, next_file, version_of) -> None:
    """Strikeout / caret edits and whether the next version shows them."""
    for a in ann[ann.in_reply_to.isna() & ann.type.isin(['StrikeOut', 'Caret'])].itertuples():
        parsed = parse_edit(a.type, a.content or '', a.marked_text or '')
        if parsed is None:
            continue
        kind, old, new, how = parsed
        e = g.node('edit:' + a.uid, 'Edit', f'{kind}: {old[:30]!r} -> {new[:30]!r}',
                   attrs={'edit_kind': kind, 'old': old, 'new': new, 'parsed_by': how})
        g.edge(a.uid, 'PROPOSES', e)
        nf = next_file.get(a.file)
        claims = [cid for cid, _, how in anchor.get(a.uid, []) if cid]
        if not nf or not claims:
            g.nodes[e]['attrs']['applied'] = 'no next version' if not nf else 'unanchored'
            continue
        status = applied_status(kind, old, new, claims, next_claim, g)
        g.nodes[e]['attrs']['applied'] = status
        if status in ('applied', 'partly applied', 'not applied'):
            g.edge(e, 'NOT_APPLIED_IN' if status == 'not applied' else 'APPLIED_IN',
                   version_of[nf], status=status)


PROSE = re.compile(
    r"\?|\b(i|i'm|i've|we|we're|we've|our|us|my|this is|is this|shouldn't|suggest|don't|"
    r"isn't|aren't|can't|not needed|not relevant|not the same|unnecessary|per last round|"
    r"previous|consistency|required|chosen|reference|marketing|medical|legal|delete|remove|"
    r"strike|above|below|paragraph|version|comment|appears|wordy|confusing|messaging|"
    r"include)\b", re.I)
DELETE_ASK = re.compile(r"^(delete|strike|remove|can delete|not needed|unnecessary)\b", re.I)
PREFIX = re.compile(r"^(revise to|add|insert|\[?suggest(?:ed)?\]?)\s*:?\s*", re.I)
CHANGE = re.compile(r'^change\s+"(.+?)"\s+to\s+"(.+?)"', re.I)


def parse_edit(atype: str, content: str, marked: str) -> tuple[str, str, str, str] | None:
    """(kind, old, new, parsed_by) for a strikeout or caret, or None when the
    comment is prose about the text rather than the replacement itself.
    Only the first line counts: later lines are usually a note on the edit."""
    first = content.strip().split('\n', 1)[0].strip()
    m = CHANGE.match(first)
    if m:
        return 'replace', m[1], m[2], 'change "x" to "y"'
    if atype == 'StrikeOut' and not first:
        return 'delete', marked, '', 'empty strikeout'
    if atype == 'StrikeOut' and DELETE_ASK.match(first):
        return 'delete', marked, '', 'delete instruction'
    how = 'literal'
    pm = PREFIX.match(first)
    if pm and pm.end() < len(first):
        first, how = first[pm.end():].strip().strip('"'), f'prefix {pm[1].lower()!r}'
    elif (PROSE.search(first) and len(first.split()) > 4
          and not (first.startswith('(') and first.endswith(')'))):
        return None
    if '?' in first or not first:
        return None
    first = first.strip('"“”')
    return ('insert', '', first, how) if atype == 'Caret' else ('replace', marked, first, how)


def applied_status(kind, old, new, claims, next_claim, g) -> str:
    nxt = [next_claim[c][0] for c in claims if c in next_claim]
    if not nxt:
        # The sentence is gone in the next version: a deletion went through.
        return 'applied' if kind in ('delete', 'replace') else 'claim removed'
    after = _norm(' '.join(g.nodes[n]['text'] for n in nxt))
    before = _norm(' '.join(g.nodes[c]['text'] for c in claims))
    o, n = _norm(old), _norm(new)
    if kind == 'insert':
        if n and n in before:
            return 'already present'
        if n and n in after:
            return 'applied'
        # Some of the inserted words arrived (e.g. "two or more" became
        # "two").
        added = set(n.split()) - set(before.split())
        got = added & set(after.split())
        return 'partly applied' if added and len(got) >= len(added) / 2 else 'not applied'
    if kind == 'delete':
        return 'not applied' if o and o in after else 'applied'
    gone = not o or o not in after or o in n
    return 'applied' if (n in after and gone) else 'not applied'


def add_resolutions(g: Graph, ann, tags, anchor, next_claim) -> None:
    """An author's record ("added X as requested") resolves the comments on the
    previous version's claim that turned into the one it marks."""
    prev_of = defaultdict(list)
    for c, (n, _) in next_claim.items():
        prev_of[n].append(c)
    comments_on = defaultdict(list)
    for e in g.edges:
        if e['rel'] == 'ON':
            comments_on[e['dst']].append(e['src'])
    for a in ann.itertuples():
        if 'record' not in tags.get(a.uid, []):
            continue
        for cid, _, _ in anchor.get(a.uid, []):
            for p in prev_of.get(cid, []) if cid else []:
                for target in comments_on.get(p, []):
                    if set(tags.get(target, [])) & CONCERN_TAGS:
                        g.edge(a.uid, 'RESOLVES', target, how='record on revised claim')


# --- Save, load, check -----------------------------------------------------

def save(g: Graph) -> None:
    CFG.out.mkdir(parents=True, exist_ok=True)
    cols = ['id', 'kind', 'label', 'text', 'file', 'page', 'x0', 'y0', 'x1', 'y1']
    n = pd.DataFrame(list(g.nodes.values()))
    for c in cols:
        if c not in n:
            n[c] = None
    n['attrs'] = n['attrs'].map(lambda d: json.dumps(d, ensure_ascii=False, default=str))
    n[cols + ['attrs']].to_parquet(CFG.nodes)
    e = pd.DataFrame(g.edges)
    e['attrs'] = e['attrs'].map(lambda d: json.dumps(d, ensure_ascii=False, default=str))
    e.to_parquet(CFG.edges)
    print(f'{len(n)} nodes, {len(e)} edges -> {CFG.out}')


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    n, e = pd.read_parquet(CFG.nodes), pd.read_parquet(CFG.edges)
    n['attrs'] = n['attrs'].map(json.loads)
    e['attrs'] = e['attrs'].map(json.loads)
    return n, e


def to_networkx() -> nx.MultiDiGraph:
    n, e = load()
    G = nx.MultiDiGraph()
    for r in n.itertuples():
        G.add_node(r.id, kind=r.kind, label=r.label, **{k: v for k, v in r.attrs.items()})
    for r in e.itertuples():
        G.add_edge(r.src, r.dst, key=r.rel, rel=r.rel, **r.attrs)
    return G


def check() -> None:
    n, e = load()
    pd.set_option('display.width', 200)
    print(n.kind.value_counts().to_string(), '\n')
    print(e.rel.value_counts().to_string(), '\n')
    roots = n[n.kind == 'Comment']
    on = e[e.rel == 'ON']
    meth = on.drop_duplicates('src').assign(m=lambda d: d['attrs'].map(lambda a: a['method']))
    print('top-level comments:', len(roots))
    print('  anchored to a claim:', meth.m.value_counts().to_dict(),
          '| page only:', (e.rel == 'ON_PAGE').sum())
    per = on.groupby('src').size()
    print('  claims per anchored comment:', per.describe()[['mean', '50%', 'max']].round(2).to_dict())
    claims = n[n.kind == 'Claim']
    st = claims['attrs'].map(lambda a: a.get('next_status'))
    print('claims:', len(claims), '| next-version status:', st.value_counts().to_dict())
    edits = n[n.kind == 'Edit']
    print('edits:', len(edits), edits['attrs'].map(lambda a: a['applied']).value_counts().to_dict())
    print('RESOLVES:', (e.rel == 'RESOLVES').sum())


VIEWER_TEMPLATE = Path(__file__).with_name('viewer.html')
KIND_COLOURS = {'Brand': '#1d2125', 'Piece': '#6b6f76', 'Version': '#8a8f98', 'Claim': '#E69F00',
                'Comment': '#0072B2', 'Reply': '#56B4E9', 'Concern': '#D55E00', 'Person': '#009E73',
                'Reference': '#CC79A7', 'Edit': '#BCBD22', 'Passage': '#cccccc'}
OVERVIEW_KINDS = ['Brand', 'Piece', 'Version', 'Claim', 'Comment', 'Reply', 'Concern', 'Person',
                  'Reference', 'Edit']


def _brand(file) -> str | None:
    return file.split(' veeva examples')[0] if isinstance(file, str) and file else None


def _s(v) -> str:
    """'' for missing values (parquet gives NaN for empty text)."""
    return v if isinstance(v, str) else ''


def export() -> None:
    """kg_overview.html (interactive, the review part of the graph) and
    kg_full.graphml (everything, for Gephi or Cytoscape), in --out."""
    VIEWER, GRAPHML = CFG.out / 'kg_overview.html', CFG.out / 'kg_full.graphml'
    n, e = load()
    # GraphML wants flat scalar attributes.
    G = nx.MultiDiGraph()
    for r in n.itertuples():
        flat = {k: (v if isinstance(v, (int, float, str, bool)) else json.dumps(v, ensure_ascii=False))
                for k, v in r.attrs.items() if v is not None}
        G.add_node(r.id, kind=r.kind, label=_s(r.label), text=_s(r.text) or '', file=_s(r.file),
                   page=int(r.page) if pd.notna(r.page) else -1, **flat)
    for r in e.itertuples():
        G.add_edge(r.src, r.dst, key=r.rel, rel=r.rel,
                   **{k: v if isinstance(v, (int, float, str, bool)) else json.dumps(v)
                      for k, v in r.attrs.items() if v is not None})
    nx.write_graphml(G, GRAPHML)

    # Overview: drop passages and sentences nobody commented on, but keep the
    # other-version counterparts of commented sentences.
    commented = set(e[e.rel == 'ON'].dst)
    links = e[e.rel.isin(['SAME_AS', 'REVISED_TO'])]
    counterparts = set(links[links.src.isin(commented)].dst) | set(links[links.dst.isin(commented)].src)
    keep = set(n[n.kind.isin(set(OVERVIEW_KINDS) - {'Claim'})].id) | commented | counterparts
    nodes = n[n.id.isin(keep)]
    edges = e[e.src.isin(keep) & e.dst.isin(keep)]
    # Sentences hang off their version directly, since passages are left out.
    version_of = {r.file: r.id for r in n[n.kind == 'Version'].itertuples()}
    claim_in = pd.DataFrame([{'src': r.id, 'rel': 'IN', 'dst': version_of[r.file], 'attrs': {}}
                             for r in nodes[nodes.kind == 'Claim'].itertuples()])
    edges = pd.concat([edges, claim_in], ignore_index=True)
    degree = pd.concat([edges.src, edges.dst]).value_counts()
    by_id = n.set_index('id')

    def brand_of(r) -> str | None:
        if r.kind == 'Concern':
            src = e[(e.rel == 'RAISES') & (e.dst == r.id)].src
            return _brand(by_id.loc[src.iloc[0]].file) if len(src) else None
        if r.kind == 'Brand':
            return r.label
        return _brand(r.file)

    piece_of_file = {}
    for p in n[n.kind == 'Piece'].itertuples():
        for f in n[(n.kind == 'Version') & n.file.str.startswith(p.file + '/')].file:
            if f.rsplit('/', 1)[0] == p.file:
                piece_of_file[f] = p.id
    pieces = sorted(set(piece_of_file.values()))
    rng = np.random.default_rng(0)
    out_nodes = []
    for r in nodes.itertuples():
        a = r.attrs
        if r.kind == 'Claim':
            label = r.text
        elif r.kind in ('Comment', 'Reply'):
            label = f'{a.get("author") or "(no author)"}: {_s(r.text) or "(" + a.get("type", "") + ", no text)"}'
        elif r.kind == 'Concern':
            label = f'{a["type"]}: {_s(r.text) or r.label}'
        else:
            label = _s(r.label)
        label = ' '.join(str(label).split())
        info = {'file': r.file, 'page': int(r.page) if pd.notna(r.page) else None}
        info.update({k: v for k, v in a.items() if k not in ('marked_text',) or v})
        # Start each piece's nodes near one point so the layout settles faster.
        f = r.file if isinstance(r.file, str) else ''
        k = pieces.index(piece_of_file[f]) if f in piece_of_file else len(pieces)
        ang = 2 * np.pi * k / (len(pieces) + 1)
        x, y = 100 * np.cos(ang) + rng.normal(0, 15), 100 * np.sin(ang) + rng.normal(0, 15)
        out_nodes.append({'key': r.id, 'attributes': {
            'label': label[:60] + ('…' if len(label) > 60 else ''), 'full_label': label[:300],
            'text': (_s(r.text) or '')[:1500], 'kind': r.kind, 'brand': brand_of(r),
            'color': KIND_COLOURS.get(r.kind, '#777'), 'x': float(x), 'y': float(y),
            'size': float(2 + 1.6 * np.sqrt(degree.get(r.id, 0))),
            'search': ' '.join([label, _s(r.text) or '', str(a.get('author') or '')]).lower()[:2000],
            'info': info}})
    out_edges = [{'source': r.src, 'target': r.dst, 'attributes': {'rel': r.rel}}
                 for r in edges.itertuples()]
    data = {'nodes': out_nodes, 'edges': out_edges, 'colours': KIND_COLOURS, 'kinds': OVERVIEW_KINDS,
            'brands': sorted({b for b in (x['attributes']['brand'] for x in out_nodes) if b})}
    html = VIEWER_TEMPLATE.read_text().replace(
        '/*DATA*/null', json.dumps(data, ensure_ascii=False, default=str).replace('</', '<\\/'))
    VIEWER.write_text(html)
    print(f'{len(out_nodes)} nodes, {len(out_edges)} edges -> {VIEWER}')
    print(f'{G.number_of_nodes()} nodes, {G.number_of_edges()} edges -> {GRAPHML}')
