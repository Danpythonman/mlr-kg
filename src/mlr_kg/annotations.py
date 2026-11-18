"""Raw annotations read straight from the source PDFs.

Every field is what the PDF stores, except two mechanical additions:
`marked_text` (the page words under a markup annotation's quads, or under
its rectangle for other types) and parsed datetimes. Nothing is inferred.
"""

import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pymupdf as fitz

# Set by configure() from the --pdfs option; nothing works until it is.
ROOT: Path = None  # type: ignore[assignment]
CACHE: Path = None  # type: ignore[assignment]


def configure(cfg) -> None:
    global ROOT, CACHE
    ROOT = cfg.pdfs
    CACHE = cfg.cache / 'annotations.parquet'

# Popups are viewer UI attached to another annotation, not content.
SKIP_TYPES = {'Popup'}
MARKUP_TYPES = {'Highlight', 'StrikeOut', 'Underline', 'Squiggly'}

_PDF_DATE = re.compile(r"D:(\d{14})(?:([+-Z])(\d{2})?'?(\d{2})?'?)?")


def parse_pdf_date(s: str) -> pd.Timestamp:
    m = _PDF_DATE.match(s or '')
    if not m:
        return pd.NaT
    dt = datetime.strptime(m[1], '%Y%m%d%H%M%S')
    if m[2] in ('+', '-'):
        off = timedelta(hours=int(m[3] or 0), minutes=int(m[4] or 0))
        dt = dt.replace(tzinfo=timezone(off if m[2] == '+' else -off))
    else:
        dt = dt.replace(tzinfo=timezone.utc)
    return pd.Timestamp(dt).tz_convert('UTC')


def pdf_files() -> list[Path]:
    return sorted(ROOT.rglob('*.pdf'))


def _words_in(words: list, rect: fitz.Rect) -> list:
    """Page words whose centre lies inside rect, in reading order."""
    return [w for w in words
            if rect.x0 <= (w[0] + w[2]) / 2 <= rect.x1 and rect.y0 <= (w[1] + w[3]) / 2 <= rect.y1]


def _marked_text(words: list, annot: fitz.Annot) -> str:
    if annot.type[1] in MARKUP_TYPES and annot.vertices:
        v = annot.vertices
        rects = [fitz.Quad(v[i:i + 4]).rect for i in range(0, len(v), 4)]
    else:
        rects = [annot.rect]
    # Some PDFs carry the same text layer twice, offset by a point or two; a
    # word overlapping an already-taken copy of itself by >50% is that copy.
    taken: list[tuple[fitz.Rect, str]] = []
    for r in rects:
        for w in _words_in(words, r):
            box = fitz.Rect(w[:4])
            if not any(t == w[4] and (box & b).get_area() > 0.5 * box.get_area()
                       for b, t in taken):
                taken.append((box, w[4]))
    return ' '.join(t for _, t in taken)


# Bump when the extracted columns change, so old caches are rebuilt.
CACHE_VERSION = 2


def uid(file: str, annot_id: str) -> str:
    """Persistent id: the PDF's own annotation id (Veeva's /NM, unique within a
    file but copied between files with pasted text boxes) plus the file path.
    Survives re-extraction and re-saving; changes only if the file is renamed."""
    return 'a' + hashlib.sha1(f'{file}#{annot_id}'.encode()).hexdigest()[:10]


def _signature() -> str:
    return json.dumps([CACHE_VERSION] + [(p.relative_to(ROOT).as_posix(), p.stat().st_size,
                        p.stat().st_mtime_ns) for p in pdf_files()])


def load_annotations() -> pd.DataFrame:
    """Cached extraction; re-reads the PDFs whenever any of them changes."""
    sig = _signature()
    sig_file = CACHE.with_suffix('.sig')
    if CACHE.exists() and sig_file.exists() and sig_file.read_text() == sig:
        return pd.read_parquet(CACHE)
    df = extract_annotations()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE)
    sig_file.write_text(sig)
    return df


def extract_annotations() -> pd.DataFrame:
    rows = []
    for path in pdf_files():
        rel = path.relative_to(ROOT)
        with fitz.open(path) as doc:
            for page in doc:
                words = None
                # A page's /Annots array can list the same object twice.
                seen = set()
                for a in page.annots():
                    kind = a.type[1]
                    if kind in SKIP_TYPES or a.xref in seen:
                        continue
                    seen.add(a.xref)
                    if words is None:
                        words = page.get_text('words', sort=True)
                    info = a.info
                    r = a.rect
                    rows.append({
                        'file': rel.as_posix(),
                        'folder': rel.parent.as_posix(),
                        'page': page.number + 1,
                        'xref': a.xref,
                        'in_reply_to': a.irt_xref or None,
                        'type': kind,
                        'author': info.get('title', ''),
                        'content': info.get('content', ''),
                        'subject': info.get('subject', ''),
                        'marked_text': _marked_text(words, a),
                        'created': parse_pdf_date(info.get('creationDate')),
                        'modified': parse_pdf_date(info.get('modDate')),
                        'x0': r.x0, 'y0': r.y0, 'x1': r.x1, 'y1': r.y1,
                        'annot_id': info.get('id', ''),
                    })
    df = pd.DataFrame(rows)
    df.insert(0, 'uid', [uid(f, i) for f, i in zip(df.file, df.annot_id, strict=True)])
    assert df.uid.is_unique, 'uid collision'
    df['in_reply_to'] = df['in_reply_to'].astype('Int64')
    return df
