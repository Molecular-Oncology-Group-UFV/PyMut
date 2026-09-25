# Register analysis fixtures globally
pytest_plugins = [
    'tests.unit.analysis.fixtures.pfam_annotation_fixtures',
    'tests.unit.analysis.fixtures.mutational_signature_fixtures',
    'tests.unit.fixtures.core_fixtures',
]

def pytest_ignore_collect(path):
    """Ignore the visualization oncoplot tests as per issue instruction."""
    try:
        basename = path.basename
    except Exception:
        try:
            from pathlib import Path
            basename = Path(str(path)).name
        except Exception:
            return False
    return basename == 'test_oncoplot.py'