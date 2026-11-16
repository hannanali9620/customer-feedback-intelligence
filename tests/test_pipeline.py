import importlib.util, io
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
SAMPLE_CSV = ROOT_DIR / 'V1' / 'sample_data' / 'sample_feedback.csv'

def load_analyzer(version):
    spec = importlib.util.spec_from_file_location(f'{version}_analyzer', ROOT_DIR / version / 'analyzer.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

@pytest.fixture(autouse=True)
def force_offline_env(monkeypatch):
    monkeypatch.setenv('ENABLE_DISTILBERT', 'false')
    monkeypatch.setenv('AZURE_LANGUAGE_ENDPOINT', '')
    monkeypatch.setenv('AZURE_LANGUAGE_KEY', '')
    monkeypatch.setenv('GEMINI_API_KEY', '')
    monkeypatch.setenv('OLLAMA_BASE_URL', 'http://127.0.0.1:19999')

@pytest.mark.parametrize('version', ['V1', 'V2', 'V3'])
def test_missing_column_raises_schema_error(version):
    mod = load_analyzer(version)
    with pytest.raises(mod.SchemaValidationError, match='Missing required column'):
        mod.analyze_feedback(io.StringIO('customer_id,product\nCUST-1,WidgetA\n'))

@pytest.mark.parametrize('version', ['V1', 'V2', 'V3'])
def test_empty_feedback_raises_schema_error(version):
    mod = load_analyzer(version)
    with pytest.raises(mod.SchemaValidationError, match='no non-empty feedback'):
        mod.analyze_feedback(io.StringIO('customer_id,product,feedback\nCUST-1,WidgetA,   \n'))

def test_cross_version_kpi_and_contract_parity():
    res = {v: load_analyzer(v).analyze_feedback(SAMPLE_CSV) for v in ('V1', 'V2', 'V3')}
    assert res['V1']['kpis'] == res['V2']['kpis'] == res['V3']['kpis']
    assert res['V1']['complaint_distribution'] == res['V2']['complaint_distribution'] == res['V3']['complaint_distribution']
    for r in res.values():
        assert set(r['backends'].keys()) == {'sentiment', 'keyphrases', 'report'}
        assert 'VADER' in r['backends']['sentiment']
        assert 'Template-Based' in r['backends']['report']
        assert len(r['limitations']) >= 3
