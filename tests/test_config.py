"""Every data location must be given; a missing or wrong one stops the run."""

import json

import pytest

from mlr_kg import cli
from mlr_kg.config import ConfigError, resolve

ENV = ['MLR_PDF_DIR', 'MLR_NOTES', 'MLR_LLM_LABELS', 'MLR_OUT']


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for v in ENV:
        monkeypatch.delenv(v, raising=False)


@pytest.fixture
def data(tmp_path):
    pdfs = tmp_path / 'pdfs' / 'Brand veeva examples' / 'Piece'
    pdfs.mkdir(parents=True)
    (pdfs / 'v0.1 x.pdf').write_bytes(b'%PDF-1.4\n')     # only its presence is checked
    notes = tmp_path / 'notes.jsonl'
    notes.write_text(json.dumps({'event': 'add', 'note_id': 'n1', 'target': 'a1', 'tags': []}) + '\n')
    llm = tmp_path / 'llm_labels.jsonl'
    llm.write_text(json.dumps({'target': 'a1', 'tags': ['no tag'], 'note': ''}) + '\n')
    return {'pdfs': tmp_path / 'pdfs', 'notes': notes, 'llm_labels': llm, 'out': tmp_path / 'out'}


def test_nothing_given_names_all_four():
    with pytest.raises(ConfigError) as e:
        resolve()
    for opt in ('--pdfs', '--notes', '--llm-labels', '--out'):
        assert opt in str(e.value)


def test_all_given(data):
    cfg = resolve(**{k: str(v) for k, v in data.items()})
    assert cfg.pdfs == data['pdfs'].resolve()


def test_environment_variables(data, monkeypatch):
    for v, k in zip(ENV, ['pdfs', 'notes', 'llm_labels', 'out']):
        monkeypatch.setenv(v, str(data[k]))
    assert resolve().notes == data['notes'].resolve()


@pytest.mark.parametrize('bad, msg', [
    ({'pdfs': '/no/such/dir'}, 'no such folder'),
    ({'notes': '/no/such.jsonl'}, 'no such file'),
])
def test_missing_paths(data, bad, msg):
    args = {k: str(v) for k, v in data.items()} | bad
    with pytest.raises(ConfigError, match=msg):
        resolve(**args)


def test_folder_without_pdfs(data, tmp_path):
    empty = tmp_path / 'empty'
    empty.mkdir()
    with pytest.raises(ConfigError, match='no PDF files'):
        resolve(**{k: str(v) for k, v in data.items()} | {'pdfs': str(empty)})


def test_files_swapped(data):
    args = {k: str(v) for k, v in data.items()}
    args['notes'], args['llm_labels'] = args['llm_labels'], args['notes']
    with pytest.raises(ConfigError, match='does not look like'):
        resolve(**args)


def test_inputs_inside_out(data):
    args = {k: str(v) for k, v in data.items()} | {'out': str(data['notes'].parent)}
    with pytest.raises(ConfigError, match='inside --out'):
        resolve(**args)


def test_check_needs_a_built_graph(tmp_path):
    with pytest.raises(ConfigError, match='run `mlr-kg build` first'):
        resolve(out=str(tmp_path), need_inputs=False)


def test_cli_exits_2_with_message(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(['build'])
    assert e.value.code == 2
    assert 'MLR_PDF_DIR' in capsys.readouterr().err
