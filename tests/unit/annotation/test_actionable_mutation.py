import types

import numpy as np
import pandas as pd
import pytest

from src.pyMut.annotate import actionable_mutation as am
from src.pyMut.annotate.actionable_mutation import (
    ActionableMutationMixin,
    _clean_empty_annotations,
)


# ---------------------------------------------------------------------------
# _clean_empty_annotations (pure function)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value,expected_nan", [
    ([], True),
    ((), True),
    ({}, True),
    ("", True),
    ("   ", True),
    ("[]", True),
    ("{}", True),
    ("x", False),
    (5, False),
    (0.5, False),
    (None, False),     # None passes through unchanged (not NaN)
])
def test_clean_empty_annotations(value, expected_nan):
    out = _clean_empty_annotations(value)
    if expected_nan:
        assert pd.isna(out)
    else:
        assert out is value or out == value


# ---------------------------------------------------------------------------
# Doubles + fake requests infrastructure (no network, ever)
# ---------------------------------------------------------------------------

class DummyPM(ActionableMutationMixin):
    def __init__(self, df, assembly="37"):
        self.data = df
        self.metadata = types.SimpleNamespace(assembly=assembly)


@pytest.fixture
def fake_requests(monkeypatch):
    """Replace the requests module inside actionable_mutation with a fake.

    Returns (calls, responses, Resp): append payloads or prebuilt Resp objects
    to `responses` BEFORE calling the method under test.
    """
    calls = []
    responses = []

    class Resp:
        def __init__(self, status, payload, text=""):
            self.status_code = status
            self._payload = payload
            self.text = text

        def json(self):
            return self._payload

    class _Session:
        def __init__(self):
            self._queue = list(responses)

        def mount(self, *a, **k):
            pass

        def post(self, url, headers=None, json=None, timeout=None):
            calls.append({"url": url, "payload": json, "headers": headers})
            item = self._queue.pop(0)
            return item if isinstance(item, Resp) else Resp(200, item)

    class _FakeRequestException(Exception):
        """Mirrors requests.exceptions.RequestException: ONLY transport errors
        are caught by the retry handler. API-error ValueErrors raised inside
        the try block (e.g. the 401 'Invalid OncoKB token') must propagate
        untouched, exactly as with the real requests library."""

    fake = types.SimpleNamespace(
        Session=_Session,
        exceptions=types.SimpleNamespace(RequestException=_FakeRequestException),
    )
    monkeypatch.setattr(am, "requests", fake)
    monkeypatch.setattr(am.time, "sleep", lambda s: None)  # batch pacing
    return calls, responses, Resp


def _two_variant_df():
    return pd.DataFrame({
        "CHROM": ["chr7", "chr12"],
        "POS": [140453136, 25398285],
        "REF": ["A", "G"],
        "ALT": ["T", "C"],
    })


# ---------------------------------------------------------------------------
# actionable_mutations_oncokb
# ---------------------------------------------------------------------------

def test_annotates_data_with_oncokb_columns(fake_requests):
    calls, responses, _ = fake_requests
    responses.append([
        {"geneExist": True, "oncogenic": "Oncogenic",
         "highestSensitiveLevel": "LEVEL_1", "dataVersion": "v3.5"},
        {"geneExist": True, "oncogenic": "Unknown",
         "highestSensitiveLevel": None, "hotspot": True},
    ])
    pm = DummyPM(_two_variant_df())
    out = pm.actionable_mutations_oncokb(token="fake-token")

    # columns created and values placed at the right rows
    assert out.loc[0, "oncokb_highestSensitiveLevel"] == "LEVEL_1"
    assert pd.isna(out.loc[1, "oncokb_highestSensitiveLevel"])
    assert out.loc[1, "oncokb_hotspot"] == True
    assert out.loc[0, "oncokb_dataVersion"] == "v3.5"

    # request: single batch with the right payload format and chr stripped
    assert len(calls) == 1
    payload = calls[0]["payload"]
    assert len(payload) == 2
    assert payload[0]["referenceGenome"] == "GRCh37"
    assert payload[0]["genomicLocation"] == "7,140453136,140453137,A,T"
    assert payload[1]["genomicLocation"] == "12,25398285,25398286,G,C"
    assert calls[0]["headers"]["Authorization"] == "Bearer fake-token"


def test_batches_and_preserves_order(fake_requests):
    calls, responses, _ = fake_requests
    df = pd.DataFrame({
        "CHROM": ["1", "1", "1"],
        "POS": [100, 200, 300],
        "REF": ["A", "C", "G"],
        "ALT": ["G", "T", "A"],
    })
    responses.extend([[{"highestSensitiveLevel": f"L{i}"}] for i in range(3)])
    pm = DummyPM(df)
    out = pm.actionable_mutations_oncokb(token="t", batch_size=1)

    assert len(calls) == 3                       # one POST per variant
    assert out["oncokb_highestSensitiveLevel"].tolist() == ["L0", "L1", "L2"]


def test_rows_with_nulls_are_dropped(fake_requests):
    calls, responses, _ = fake_requests
    df = pd.DataFrame({
        "CHROM": ["1", "1"],
        "POS": [100, 200],
        "REF": ["A", "C"],
        "ALT": ["G", None],          # second row has null ALT -> removed
    })
    responses.append([{"highestSensitiveLevel": "LEVEL_2"}])
    pm = DummyPM(df)
    out = pm.actionable_mutations_oncokb(token="t")

    assert len(calls) == 1
    assert len(calls[0]["payload"]) == 1
    assert out.loc[0, "oncokb_highestSensitiveLevel"] == "LEVEL_2"
    assert pd.isna(out.loc[1, "oncokb_highestSensitiveLevel"])


@pytest.mark.parametrize("missing", ["CHROM", "POS", "REF", "ALT"])
def test_missing_required_column_raises(fake_requests, missing):
    _, responses, _ = fake_requests
    df = _two_variant_df().drop(columns=[missing])
    pm = DummyPM(df)
    with pytest.raises(ValueError, match=missing):
        pm.actionable_mutations_oncokb(token="t")
    assert len(responses) == 0                   # no request was ever sent


def test_invalid_reference_genome_raises(fake_requests):
    _, responses, _ = fake_requests
    pm = DummyPM(_two_variant_df(), assembly="19")
    with pytest.raises(ValueError, match="Invalid reference genome"):
        pm.actionable_mutations_oncokb(token="t")


def test_authentication_error_raises(fake_requests):
    _, responses, Resp = fake_requests
    responses.append(Resp(401, [], text="Unauthorized"))
    pm = DummyPM(_two_variant_df())
    with pytest.raises(ValueError, match="Invalid OncoKB token"):
        pm.actionable_mutations_oncokb(token="bad-token")
