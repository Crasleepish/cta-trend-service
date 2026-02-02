from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sqlalchemy import Column, Date, Float, LargeBinary, MetaData, String, Table, select
from sqlalchemy.engine import Engine
from sqlalchemy.sql import func

FACTOR_ORDER = ["MKT", "SMB", "HML", "QMJ", "const", "gamma"]
SMB_INDEX = 1
QMJ_INDEX = 3


@dataclass(frozen=True)
class UnstableRule:
    mode: str
    value: float


def fetch_beta_rows(
    codes: Sequence[str],
    t0: date,
    *,
    rows: pd.DataFrame | None = None,
    engine: Engine | None = None,
    schema: str = "public",
) -> pd.DataFrame:
    if rows is not None:
        return _fetch_from_frame(rows, codes, t0)
    if engine is None:
        raise ValueError("engine or rows must be provided")
    return _fetch_from_db(engine, schema, codes, t0)


def decode_cov(P_bin: bytes, *, strict: bool = True) -> NDArray[np.float64]:
    expected_len = 21 * 4
    if len(P_bin) != expected_len:
        if strict:
            raise ValueError("P_bin length invalid")
        return np.full((6, 6), np.nan, dtype=np.float64)
    data = np.frombuffer(P_bin, dtype=np.float32)
    if data.size != 21:
        if strict:
            raise ValueError("P_bin decode invalid")
        return np.full((6, 6), np.nan, dtype=np.float64)

    cov: NDArray[np.float64] = np.zeros((6, 6), dtype=np.float64)
    idx = 0
    for i in range(6):
        for j in range(i + 1):
            cov[i, j] = float(data[idx])
            idx += 1
    cov = cov + cov.T - np.diag(np.diag(cov))
    diag = np.clip(np.diag(cov), 0.0, None)
    np.fill_diagonal(cov, diag)
    return cov


def extract_var_2(cov: np.ndarray) -> tuple[float, float]:
    if cov.shape != (6, 6):
        raise ValueError("cov must be 6x6")
    return float(cov[SMB_INDEX, SMB_INDEX]), float(cov[QMJ_INDEX, QMJ_INDEX])


def beta_uncertainty_score(var_SMB: float, var_QMJ: float) -> float:
    return float((np.sqrt(var_SMB) + np.sqrt(var_QMJ)) / 2.0)


def filter_unstable(U: pd.Series, rule: UnstableRule) -> pd.Index:
    if rule.mode == "absolute":
        threshold = rule.value
    elif rule.mode == "quantile":
        threshold = float(U.quantile(rule.value))
    else:
        raise ValueError("rule.mode must be 'absolute' or 'quantile'")
    return U.index[U <= threshold]


def robust_scale_mad(
    beta_2: pd.DataFrame,
    *,
    eps: float = 1e-12,
) -> tuple[pd.DataFrame, pd.Series]:
    median = beta_2.median(axis=0)
    mad = (beta_2 - median).abs().median(axis=0)
    scales = mad + eps
    scaled = beta_2 / scales
    return scaled, scales


def l2_normalize(beta_scaled: pd.DataFrame, *, eps: float = 1e-12) -> pd.DataFrame:
    norms = np.linalg.norm(beta_scaled.to_numpy(), axis=1)
    norms = norms + eps
    normalized = beta_scaled.div(norms, axis=0)
    return normalized


def _fetch_from_frame(rows: pd.DataFrame, codes: Sequence[str], t0: date) -> pd.DataFrame:
    subset = rows[rows["code"].isin(codes) & (rows["date"] <= pd.Timestamp(t0))].copy()
    if subset.empty:
        return subset
    idx = subset.groupby("code")["date"].idxmax()
    return subset.loc[idx].reset_index(drop=True)


def _fetch_from_db(
    engine: Engine,
    schema: str,
    codes: Sequence[str],
    t0: date,
) -> pd.DataFrame:
    metadata = MetaData(schema=schema)
    fund_beta = Table(
        "fund_beta",
        metadata,
        Column("code", String(64)),
        Column("date", Date),
        Column("MKT", Float),
        Column("SMB", Float),
        Column("HML", Float),
        Column("QMJ", Float),
        Column("const", Float),
        Column("gamma", Float),
        Column("P_bin", LargeBinary),
    )

    row_number = func.row_number().over(
        partition_by=fund_beta.c.code,
        order_by=fund_beta.c.date.desc(),
    )
    subquery = (
        select(
            fund_beta.c.code,
            fund_beta.c.date,
            fund_beta.c.MKT,
            fund_beta.c.SMB,
            fund_beta.c.HML,
            fund_beta.c.QMJ,
            fund_beta.c.const,
            fund_beta.c.gamma,
            fund_beta.c.P_bin,
            row_number.label("rn"),
        )
        .where(fund_beta.c.code.in_(codes))
        .where(fund_beta.c.date <= t0)
        .subquery()
    )
    stmt = select(subquery).where(subquery.c.rn == 1)
    with engine.connect() as connection:
        result = connection.execute(stmt)
        return pd.DataFrame([dict(row._mapping) for row in result])
