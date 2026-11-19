"""mlr-kg: build the MLR review knowledge graph, and label comments with an LLM.

Graph:
    mlr-kg build   --pdfs DIR --notes FILE --llm-labels FILE --out DIR
    mlr-kg check   --out DIR
    mlr-kg export  --out DIR
    mlr-kg handoff --pdfs DIR --notes FILE --llm-labels FILE --out DIR

LLM labelling (writes llm_labels.jsonl, which the graph reads):
    mlr-kg label {splits,show,estimate,ping,run,score,export} --pdfs DIR --notes FILE --out DIR
    (label export also needs --llm-labels FILE, the file to write)

Every location can also come from an environment variable (MLR_PDF_DIR,
MLR_NOTES, MLR_LLM_LABELS, MLR_OUT). There are no defaults.
"""

from __future__ import annotations

import argparse
import sys

from . import autolabel, config, handoff, kg

COMMANDS = {
    'build': ('read the PDFs and labels, write the graph, then check and export', True),
    'check': ('print counts and anchoring statistics for a built graph', False),
    'export': ('write kg_overview.html and kg_full.graphml from a built graph', False),
    'handoff': ('write the flat comments table (needs a built graph)', True),
}
LABEL_INPUTS = ('pdfs', 'notes', 'out')


def _locations(p: argparse.ArgumentParser, inputs, llm_help: str | None, label: bool) -> None:
    if 'pdfs' in inputs:
        p.add_argument('--pdfs', metavar='DIR', help='folder of reviewed PDFs [MLR_PDF_DIR]')
    if 'notes' in inputs:
        p.add_argument('--notes', metavar='FILE', help='human labels, notes.jsonl [MLR_NOTES]')
    if llm_help:
        p.add_argument('--llm-labels', metavar='FILE', help=llm_help + ' [MLR_LLM_LABELS]')
    p.add_argument('--out', metavar='DIR', help='output folder [MLR_OUT]')
    if 'pdfs' in inputs:
        p.add_argument('--taxonomy', metavar='FILE',
                       help='tag scheme to use instead of the bundled taxonomy.yaml')
    if label:
        p.add_argument('--glossary', metavar='FILE',
                       help='text file of names the model should know, added to every prompt')


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog='mlr-kg', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True, metavar='COMMAND')
    for name, (help_, needs_inputs) in COMMANDS.items():
        p = sub.add_parser(name, help=help_, description=help_)
        inputs = config.GRAPH_INPUTS if needs_inputs else ('out',)
        _locations(p, inputs, 'LLM labels, llm_labels.jsonl' if needs_inputs else None, False)
    lp = sub.add_parser('label', help='label comments with an LLM (see `mlr-kg label -h`)',
                        description=autolabel.__doc__,
                        formatter_class=argparse.RawDescriptionHelpFormatter)
    lsub = lp.add_subparsers(dest='label_cmd', required=True, metavar='LABEL_COMMAND')
    autolabel.add_commands(lsub)
    for name, p in lsub.choices.items():
        _locations(p, LABEL_INPUTS, 'file to write the labels to' if name == 'export' else None,
                   True)
    return ap


def main(argv: list[str] | None = None) -> None:
    try:
        run(parser().parse_args(argv))
    except config.ConfigError as e:
        print(f'mlr-kg: error: {e}', file=sys.stderr)
        sys.exit(2)


def run(a: argparse.Namespace) -> None:
    if a.cmd == 'label':
        export = a.label_cmd == 'export'
        cfg = config.resolve(a.pdfs, a.notes, getattr(a, 'llm_labels', None), a.out,
                             need=LABEL_INPUTS + (('llm_labels',) if export else ()),
                             llm_labels_out=export, overwrite=getattr(a, 'overwrite', False),
                             taxonomy=a.taxonomy, glossary=a.glossary)
        autolabel.configure(cfg)
        if not cfg.glossary and a.label_cmd in ('show', 'ping', 'run', 'estimate'):
            print('note: no --glossary given; prompts will not explain any names.',
                  file=sys.stderr)
        a.label_fn(a)
        return

    needs_inputs = COMMANDS[a.cmd][1]
    if needs_inputs:
        cfg = config.resolve(a.pdfs, a.notes, a.llm_labels, a.out, taxonomy=a.taxonomy)
    else:
        cfg = config.resolve(out=a.out, need_inputs=False)
    kg.configure(cfg)
    if a.cmd == 'build':
        kg.save(kg.build())
        kg.check()
        kg.export()
    elif a.cmd == 'check':
        kg.check()
    elif a.cmd == 'export':
        kg.export()
    elif a.cmd == 'handoff':
        if not cfg.nodes.is_file():
            raise config.ConfigError(f'--out {cfg.out}: no graph there; run `mlr-kg build` first.')
        handoff.export()


if __name__ == '__main__':
    main()
