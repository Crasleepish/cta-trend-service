from __future__ import annotations

import base64
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.bucket_reco.beta.stability import (
    UnstableRule,
    beta_uncertainty_score,
    decode_cov,
    extract_var_2,
    fetch_beta_rows,
    filter_unstable,
    l2_normalize,
    robust_scale_mad,
)
from tests.helpers.db import create_fund_beta_table


def _encode_cov_lower_triangle(cov: np.ndarray) -> bytes:
    values = []
    for i in range(6):
        for j in range(i + 1):
            values.append(cov[i, j])
    return np.asarray(values, dtype=np.float32).tobytes()


def test_decode_cov_success_and_symmetry() -> None:
    cov = np.eye(6, dtype=float)
    cov[1, 1] = -1.0
    P_bin = _encode_cov_lower_triangle(cov)
    decoded = decode_cov(P_bin)
    assert decoded.shape == (6, 6)
    assert np.allclose(decoded, decoded.T)
    assert decoded[1, 1] == 0.0


def test_decode_cov_invalid_length_raises() -> None:
    with pytest.raises(ValueError, match="length"):
        decode_cov(b"short")


def test_extract_var_and_uncertainty_score() -> None:
    cov = np.eye(6, dtype=float)
    cov[1, 1] = 4.0
    cov[3, 3] = 16.0
    var_smb, var_qmj = extract_var_2(cov)
    assert var_smb == 4.0
    assert var_qmj == 16.0
    assert beta_uncertainty_score(var_smb, var_qmj) == 3.0


def test_filter_unstable_absolute_and_quantile() -> None:
    U = pd.Series({"A": 0.1, "B": 0.3, "C": 0.9})
    absolute = filter_unstable(U, UnstableRule(mode="absolute", value=0.3))
    assert list(absolute) == ["A", "B"]
    quantile = filter_unstable(U, UnstableRule(mode="quantile", value=0.5))
    assert set(quantile) == {"A", "B"}


def test_robust_scale_and_l2_normalize() -> None:
    beta_2 = pd.DataFrame(
        {"SMB": [1.0, 1.0, 1.0], "QMJ": [1.0, 2.0, 3.0]},
        index=["F1", "F2", "F3"],
    )
    scaled, scales = robust_scale_mad(beta_2, eps=1e-6)
    assert np.all(scales > 0)
    normalized = l2_normalize(scaled, eps=1e-6)
    norms = np.linalg.norm(normalized.to_numpy(), axis=1)
    assert np.allclose(norms, np.ones_like(norms), atol=1e-6)
    assert np.isfinite(normalized.to_numpy()).all()


def test_fetch_beta_rows_latest_as_of() -> None:
    cov = np.eye(6, dtype=float)
    P_bin = _encode_cov_lower_triangle(cov)
    rows = pd.DataFrame(
        [
            {
                "code": "F1",
                "date": pd.Timestamp("2026-01-01"),
                "MKT": 0.1,
                "SMB": 0.2,
                "HML": 0.3,
                "QMJ": 0.4,
                "const": 0.0,
                "gamma": 0.0,
                "P_bin": P_bin,
            },
            {
                "code": "F1",
                "date": pd.Timestamp("2026-01-05"),
                "MKT": 0.1,
                "SMB": 0.25,
                "HML": 0.35,
                "QMJ": 0.45,
                "const": 0.0,
                "gamma": 0.0,
                "P_bin": P_bin,
            },
            {
                "code": "F2",
                "date": pd.Timestamp("2026-01-03"),
                "MKT": 0.2,
                "SMB": 0.1,
                "HML": 0.2,
                "QMJ": 0.3,
                "const": 0.0,
                "gamma": 0.0,
                "P_bin": P_bin,
            },
        ]
    )
    out = fetch_beta_rows(["F1", "F2"], date(2026, 1, 4), rows=rows)
    assert set(out["code"]) == {"F1", "F2"}
    f1 = out[out["code"] == "F1"].iloc[0]
    assert f1["date"] == pd.Timestamp("2026-01-01")


def test_fetch_beta_rows_from_db_latest_as_of(pg_engine) -> None:
    table = create_fund_beta_table(pg_engine, schema="public")
    p_bin_b64 = (
        "3h2WOgiPObmyVbo6RAgoOmy46jiqPP064MrtOU+ZJzsgdZ04OnGLPP2ZdLWn03i1"
        "WbQ4NQ1MHbcRo7A2KwcMMc82Zq89qAOxZngHMkcCGDIMuwM6"
    )
    p_bin = base64.b64decode(p_bin_b64)
    with pg_engine.begin() as conn:
        conn.execute(
            table.insert(),
            [
                {
                    "code": "F161604.OF",
                    "date": date(2025, 12, 1),
                    "MKT": -4.569951805518303e-05,
                    "SMB": 9.406573218280596e-05,
                    "HML": -6.8381833463019354e-06,
                    "QMJ": 0.0002932633433694149,
                    "const": -1.084520377220939e-07,
                    "gamma": -1.4800733615840022e-06,
                    "P_bin": p_bin,
                },
                {
                    "code": "F161604.OF",
                    "date": date(2025, 12, 31),
                    "MKT": -4.569951805518303e-05,
                    "SMB": 9.406573218280596e-05,
                    "HML": -6.8381833463019354e-06,
                    "QMJ": 0.0002932633433694149,
                    "const": -1.084520377220939e-07,
                    "gamma": -1.4800733615840022e-06,
                    "P_bin": p_bin,
                },
            ],
        )
    out = fetch_beta_rows(["F161604.OF"], date(2025, 12, 31), engine=pg_engine)
    assert len(out) == 1
    assert out.iloc[0]["date"] == date(2025, 12, 31)
