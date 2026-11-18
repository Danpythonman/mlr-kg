"""mlr-kg: build and export the MLR review knowledge graph.

    mlr-kg build   --pdfs DIR --notes FILE --llm-labels FILE --out DIR
    mlr-kg check   --out DIR
    mlr-kg export  --out DIR
    mlr-kg handoff --pdfs DIR --notes FILE --llm-labels FILE --out DIR

Every location can also come from an environment variable (MLR_PDF_DIR,
MLR_NOTES, MLR_LLM_LABELS, MLR_OUT). There are no defaults.
"""

from __future__ import annotations

import argparse
import sys

from . import config, handoff, kg

COMMANDS = {
    'build': ('read the PDFs and labels, write the graph, then check and export', True),
    'check': ('print counts and anchoring statistics for a built graph', False),
    'export': ('write kg_overview.html and kg_full.graphml from a built graph', False),
    'handoff': ('write the flat comments table (needs a built graph)', True),
}


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog='mlr-kg', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True, metavar='COMMAND')
    for name, (help_, needs_inputs) in COMMANDS.items():
        p = sub.add_parser(name, help=help_, description=help_)
        if needs_inputs:
            p.add_argument('--pdfs', metavar='DIR', help='folder of reviewed PDFs [MLR_PDF_DIR]')
            p.add_argument('--notes', metavar='FILE', help='human labels, notes.jsonl [MLR_NOTES]')
            p.add_argument('--llm-labels', metavar='FILE',
                           help='LLM labels, llm_labels.jsonl [MLR_LLM_LABELS]')
        p.add_argument('--out', metavar='DIR', help='output folder [MLR_OUT]')
    return ap


def main(argv: list[str] | None = None) -> None:
    try:
        run(parser().parse_args(argv))
    except config.ConfigError as e:
        print(f'mlr-kg: error: {e}', file=sys.stderr)
        sys.exit(2)


def run(a: argparse.Namespace) -> None:
    needs_inputs = COMMANDS[a.cmd][1]
    if needs_inputs:
        cfg = config.resolve(a.pdfs, a.notes, a.llm_labels, a.out)
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
