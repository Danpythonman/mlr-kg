"""One flat table of every comment: its labels, who made them, and where it
sits in the PDFs (file, page, mark, sentence, passage).

Written to <out>/handoff/comments_labelled.{parquet,csv}. Needs a built graph.
"""

from __future__ import annotations

import shutil

import pandas as pd

from . import kg, labels


def export() -> None:
    cfg = kg.CFG
    items = kg.items()
    llm = labels.llm_labels(cfg.llm_labels).set_index('target')
    raw_tags = labels.original_tags(labels.replay_notes(cfg.notes))
    nodes, edges = kg.load()
    nodes = nodes.set_index('id')

    # Sentence each comment is on: the ON edge with the largest share of the mark.
    on = edges[edges.rel == 'ON'].copy()
    on['share'] = on['attrs'].map(lambda a: a.get('share', 0))
    on['method'] = on['attrs'].map(lambda a: a.get('method'))
    n_sent = on.groupby('src').size()
    on = on.sort_values('share', ascending=False).drop_duplicates('src').set_index('src')
    on_page = set(edges[edges.rel == 'ON_PAGE'].src)
    passage_of = edges[edges.rel == 'HAS_CLAIM'].set_index('dst').src

    def get(L, k, empty=None):
        if L is None or k not in L or not isinstance(L[k], (str, list)):
            return empty
        return list(L[k]) if isinstance(L[k], list) else L[k]

    rows = []
    for r in items.itertuples():
        if r.split in ('dev', 'test'):
            source = 'human' if r.uid in raw_tags else 'human (no note = no tag)'
        elif r.split == 'text-only':
            source = 'human (note, no tag)'
        else:
            source = 'llm'
        L = llm.loc[r.uid] if r.uid in llm.index else None
        human = source.startswith('human')
        anchor = r.uid if (r.uid in on.index or r.uid in on_page) else r.root_uid
        row = {
            'uid': r.uid, 'label_source': source,
            'tags': r.gold if human else get(L, 'tags', []),
            'human_tags': r.gold if human else [],
            'human_tags_original': raw_tags.get(r.uid, []),
            'human_note': r.my_note,
            'llm_model': get(L, 'model'), 'llm_prompt': get(L, 'variant'),
            'llm_tags': get(L, 'tags', []), 'llm_note': get(L, 'note'),
            'llm_confidence': get(L, 'confidence'),
            'llm_suggested_new_tag': get(L, 'suggested_new_tag'),
            'gpt4o_tags': get(L, 'gpt-4o_tags', []),
            'split': r.split,
            'brand_folder': r.file.split('/')[0], 'file': r.file, 'page': int(r.page),
            'annot_type': r.type, 'author': r.author, 'created': r.created,
            'is_reply': isinstance(r.parent_uid, str),
            'parent_uid': r.parent_uid, 'root_uid': r.root_uid,
            'comment_text': r.content, 'marked_text': r.marked_text,
            'mark_x0': r.x0, 'mark_y0': r.y0, 'mark_x1': r.x1, 'mark_y1': r.y1,
            'anchored_via': 'self' if anchor == r.uid else 'root comment',
        }
        if anchor in on.index:
            c = on.loc[anchor].dst
            s, p = nodes.loc[c], nodes.loc[passage_of[c]]
            row |= {'anchor': 'sentence', 'anchor_method': on.at[anchor, 'method'],
                    'anchor_share': round(float(on.at[anchor, 'share']), 2),
                    'n_sentences_marked': int(n_sent[anchor]),
                    'sentence_id': c, 'sentence': s.text,
                    'sent_x0': s.x0, 'sent_y0': s.y0, 'sent_x1': s.x1, 'sent_y1': s.y1,
                    'passage_id': p.name, 'passage': p.text}
        else:
            row |= {'anchor': 'page only' if anchor in on_page else 'none'}
        rows.append(row)

    df = pd.DataFrame(rows)
    out = cfg.out / 'handoff'
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / 'comments_labelled.parquet', index=False)
    csv = df.copy()
    for c in ('tags', 'human_tags', 'human_tags_original', 'llm_tags', 'gpt4o_tags'):
        csv[c] = csv[c].map('; '.join)
    csv.to_csv(out / 'comments_labelled.csv', index=False)
    shutil.copy(labels.TAXONOMY, out / 'taxonomy.yaml')
    print(f'{len(df)} rows -> {out}')
    print(df.label_source.value_counts().to_string(), '\n')
    print(pd.crosstab(df.label_source, df.anchor).to_string())
