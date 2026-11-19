"""The label commands' extra rules: export target, glossary, API keys. No calls."""

import json

import pytest

from mlr_kg import autolabel, cli
from mlr_kg.config import ConfigError, resolve

LABEL = ('pdfs', 'notes', 'out')


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for v in ('MLR_PDF_DIR', 'MLR_NOTES', 'MLR_LLM_LABELS', 'MLR_OUT',
              'ANTHROPIC_API_KEY', 'OPENAI_API_KEY'):
        monkeypatch.delenv(v, raising=False)


@pytest.fixture
def data(tmp_path):
    pdfs = tmp_path / 'pdfs' / 'Brand veeva examples' / 'Piece'
    pdfs.mkdir(parents=True)
    (pdfs / 'v0.1 x.pdf').write_bytes(b'%PDF-1.4\n')
    notes = tmp_path / 'notes.jsonl'
    notes.write_text(json.dumps({'event': 'add', 'note_id': 'n1', 'target': 'a1', 'tags': []}) + '\n')
    return {'pdfs': str(tmp_path / 'pdfs'), 'notes': str(notes), 'out': str(tmp_path / 'out')}


def test_label_commands_do_not_need_llm_labels(data):
    resolve(**data, need=LABEL)


def test_label_commands_need_pdfs_notes_out():
    with pytest.raises(ConfigError) as e:
        resolve(need=LABEL)
    msg = str(e.value)
    assert '--pdfs' in msg and '--notes' in msg and '--out' in msg and '--llm-labels' not in msg


def test_export_needs_a_target(data):
    with pytest.raises(ConfigError, match='--llm-labels'):
        resolve(**data, need=LABEL + ('llm_labels',), llm_labels_out=True)


def test_export_will_not_overwrite(data, tmp_path):
    target = tmp_path / 'llm_labels.jsonl'
    target.write_text('keep me\n')
    with pytest.raises(ConfigError, match='--overwrite'):
        resolve(**data, llm_labels=str(target), need=LABEL + ('llm_labels',), llm_labels_out=True)
    resolve(**data, llm_labels=str(target), need=LABEL + ('llm_labels',), llm_labels_out=True,
            overwrite=True)
    assert target.read_text() == 'keep me\n'     # resolving never writes


def test_export_target_folder_must_exist(data, tmp_path):
    with pytest.raises(ConfigError, match='does not exist'):
        resolve(**data, llm_labels=str(tmp_path / 'nope' / 'x.jsonl'),
                need=LABEL + ('llm_labels',), llm_labels_out=True)


def test_export_cannot_target_the_notes(data):
    with pytest.raises(ConfigError, match='same file'):
        resolve(**data, llm_labels=data['notes'], need=LABEL + ('llm_labels',),
                llm_labels_out=True, overwrite=True)


def test_missing_glossary_or_taxonomy(data):
    with pytest.raises(ConfigError, match='--glossary') as e:
        resolve(**data, need=LABEL, glossary='/no/such.txt', taxonomy='/no/such.yaml')
    assert '--taxonomy' in str(e.value)


def test_glossary_joins_wrapped_lines(data, tmp_path):
    g = tmp_path / 'glossary.txt'
    g.write_text('Names you will see: ACME is\nthe agency.\n\nSecond   paragraph.\n')
    autolabel.configure(resolve(**data, need=LABEL, glossary=str(g)))
    assert autolabel.glossary() == 'Names you will see: ACME is the agency.\n\nSecond paragraph.'


def test_paid_calls_need_a_key(data):
    autolabel.configure(resolve(**data, need=LABEL))
    with pytest.raises(ConfigError, match='ANTHROPIC_API_KEY'):
        autolabel.require_key('claude-sonnet-5-5')
    with pytest.raises(ConfigError, match='OPENAI_API_KEY'):
        autolabel.require_key('gpt-4o')


def test_cli_label_without_locations_exits_2(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(['label', 'splits'])
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert 'MLR_PDF_DIR' in err and 'MLR_LLM_LABELS' not in err
