from __future__ import annotations

import base64
from datetime import date

import numpy as np
import pandas as pd

from src.bucket_reco.beta.stability import decode_cov
from src.bucket_reco.proxy.composite import build_composite_proxy
from src.bucket_reco.runner import run_bucket_asset_recommender
from src.core.config import AppConfig, BucketRecoConfig, DbConfig, LoggingConfig
from src.utils.series import build_index_from_returns
from tests.helpers.db import create_fund_beta_table, create_fund_hist_table, create_index_hist_table


def _encode_cov_lower_triangle(cov: np.ndarray) -> bytes:
    values = []
    for i in range(6):
        for j in range(i + 1):
            values.append(cov[i, j])
    return np.asarray(values, dtype=np.float32).tobytes()


def _index_series() -> dict[str, pd.Series]:
    data = {
        "000300.SH": [
            ("2025-12-01", 4576.4856),
            ("2025-12-02", 4554.3347),
            ("2025-12-03", 4531.0486),
            ("2025-12-04", 4546.5664),
            ("2025-12-05", 4584.5368),
            ("2025-12-08", 4621.7545),
            ("2025-12-09", 4598.2232),
            ("2025-12-10", 4591.8273),
            ("2025-12-11", 4552.1848),
            ("2025-12-12", 4580.95),
            ("2025-12-15", 4552.0596),
            ("2025-12-16", 4497.5545),
            ("2025-12-17", 4579.875),
            ("2025-12-18", 4552.7926),
            ("2025-12-19", 4568.1781),
            ("2025-12-22", 4611.6235),
            ("2025-12-23", 4620.7341),
            ("2025-12-24", 4634.0586),
            ("2025-12-25", 4642.5357),
            ("2025-12-26", 4657.24),
            ("2025-12-29", 4639.3721),
            ("2025-12-30", 4651.2818),
            ("2025-12-31", 4629.9395),
        ],
        "000905.SH": [
            ("2025-12-01", 7101.8323),
            ("2025-12-02", 7040.3008),
            ("2025-12-03", 6996.3552),
            ("2025-12-04", 7012.8061),
            ("2025-12-05", 7097.8389),
            ("2025-12-08", 7172.3671),
            ("2025-12-09", 7121.3256),
            ("2025-12-10", 7155.9944),
            ("2025-12-11", 7082.8932),
            ("2025-12-12", 7169.791),
            ("2025-12-15", 7113.962),
            ("2025-12-16", 7001.3188),
            ("2025-12-17", 7137.8339),
            ("2025-12-18", 7100.8359),
            ("2025-12-19", 7169.5511),
            ("2025-12-22", 7255.6607),
            ("2025-12-23", 7256.7864),
            ("2025-12-24", 7352.0396),
            ("2025-12-25", 7410.7119),
            ("2025-12-26", 7458.8386),
            ("2025-12-29", 7430.6147),
            ("2025-12-30", 7458.936),
            ("2025-12-31", 7465.5742),
        ],
        "000852.SH": [
            ("2025-12-01", 7386.6811),
            ("2025-12-02", 7313.178),
            ("2025-12-03", 7248.274),
            ("2025-12-04", 7248.6573),
            ("2025-12-05", 7342.4924),
            ("2025-12-08", 7423.0462),
            ("2025-12-09", 7380.5835),
            ("2025-12-10", 7408.2441),
            ("2025-12-11", 7311.9952),
            ("2025-12-12", 7370.9416),
            ("2025-12-15", 7309.0775),
            ("2025-12-16", 7181.62),
            ("2025-12-17", 7288.7384),
            ("2025-12-18", 7272.4024),
            ("2025-12-19", 7329.8057),
            ("2025-12-22", 7408.3488),
            ("2025-12-23", 7392.4204),
            ("2025-12-24", 7506.3814),
            ("2025-12-25", 7579.3791),
            ("2025-12-26", 7605.5299),
            ("2025-12-29", 7594.1575),
            ("2025-12-30", 7597.299),
            ("2025-12-31", 7595.2845),
        ],
    }
    series = {}
    for code, rows in data.items():
        idx = pd.DatetimeIndex([pd.Timestamp(d) for d, _ in rows])
        series[code] = pd.Series([v for _, v in rows], index=idx, name=code)
    return series


def _fund_beta_rows() -> dict[str, dict[str, float]]:
    return {
        "000008.OF": {
            "MKT": 1.046949401125729,
            "SMB": -0.02281007818708613,
            "HML": -0.11994933205210927,
            "QMJ": -0.010992640792896016,
            "const": 0.0025855544990838674,
            "gamma": -0.006501107062073596,
        },
        "000042.OF": {
            "MKT": 1.0147696909328812,
            "SMB": -0.28212752491479337,
            "HML": 0.2708817577950781,
            "QMJ": 0.007820901704774043,
            "const": -0.0021717243328353937,
            "gamma": -0.009119794891214837,
        },
        "000051.OF": {
            "MKT": 0.9393313514725707,
            "SMB": -0.34815816381567655,
            "HML": 0.13548355939287998,
            "QMJ": 0.06659426538016536,
            "const": -0.001966792242481278,
            "gamma": -0.006722961750302142,
        },
        "000055.OF": {
            "MKT": -0.08139666674802902,
            "SMB": 0.16174640342218546,
            "HML": -0.14192351349294996,
            "QMJ": -0.16995317776555222,
            "const": -0.0064474906854276775,
            "gamma": -0.011206028392605377,
        },
        "000059.OF": {
            "MKT": 0.6819980640080352,
            "SMB": 0.22805275692028548,
            "HML": 0.031184594653791066,
            "QMJ": 0.43105806115646716,
            "const": -0.002499250542558986,
            "gamma": -0.007420796327490305,
        },
        "000071.OF": {
            "MKT": 0.772289215549752,
            "SMB": -0.15451437786358027,
            "HML": 0.06578550702767523,
            "QMJ": -0.015476154742322237,
            "const": -0.005768879343166561,
            "gamma": -0.0077972097203896865,
        },
        "000075.OF": {
            "MKT": 0.7745189252313336,
            "SMB": -0.16189924256361052,
            "HML": 0.06884020861820467,
            "QMJ": -0.017215966682974426,
            "const": -0.005108773251741312,
            "gamma": -0.007795016654469695,
        },
        "000076.OF": {
            "MKT": 0.7745189252313336,
            "SMB": -0.16189924256361052,
            "HML": 0.06884020861820467,
            "QMJ": -0.017215966682974426,
            "const": -0.005108773251741312,
            "gamma": -0.007795016654469695,
        },
        "000176.OF": {
            "MKT": 0.902964444305244,
            "SMB": -0.3113117517989113,
            "HML": 0.14440262545992427,
            "QMJ": 0.18891513617317018,
            "const": -0.0022239536354578537,
            "gamma": -0.00786479456173882,
        },
        "003184.OF": {
            "MKT": 0.0,
            "SMB": 0.0,
            "HML": 0.0,
            "QMJ": 0.0,
            "const": 0.0,
            "gamma": 0.0,
        },
    }


def test_run_bucket_asset_recommender_end_to_end(pg_engine) -> None:
    index_table = create_index_hist_table(pg_engine)
    fund_hist_table = create_fund_hist_table(pg_engine)
    fund_beta_table = create_fund_beta_table(pg_engine)
    index_series = _index_series()
    index_rows = []
    for code, series in index_series.items():
        for dt, value in series.items():
            index_rows.append({"index_code": code, "date": dt.date(), "close": float(value)})
    with pg_engine.begin() as conn:
        conn.execute(index_table.insert(), index_rows)

    proxy_returns, proxy_index = build_composite_proxy(
        index_series, {code: 1 / 3 for code in index_series}
    )

    good = ["000008.OF", "000042.OF", "000051.OF", "000055.OF", "000059.OF", "000071.OF"]
    beta_neg = ["000075.OF", "000076.OF"]
    trend_bad = ["000176.OF", "003184.OF"]
    nav_rows = []
    noise = np.sin(np.linspace(0, 3.14, len(proxy_index)))
    for idx, code in enumerate(good):
        nav = proxy_index * (1 + 0.001 * noise * (idx + 1))
        for dt, value in nav.items():
            nav_rows.append({"fund_code": code, "date": dt.date(), "net_value": float(value)})
    for code in beta_neg:
        neg_returns = -0.6 * proxy_returns
        nav = build_index_from_returns(neg_returns)
        for dt, value in nav.items():
            nav_rows.append({"fund_code": code, "date": dt.date(), "net_value": float(value)})
    constant_nav = float(proxy_index.iloc[0])
    for code in trend_bad:
        nav = pd.Series([constant_nav] * len(proxy_index), index=proxy_index.index, name=code)
        for dt, value in nav.items():
            nav_rows.append({"fund_code": code, "date": dt.date(), "net_value": float(value)})

    with pg_engine.begin() as conn:
        conn.execute(fund_hist_table.insert(), nav_rows)

    beta_rows = _fund_beta_rows()
    base_b64 = (
        "3h2WOgiPObmyVbo6RAgoOmy46jiqPP064MrtOU+ZJzsgdZ04OnGLPP2ZdLWn03i1"
        "WbQ4NQ1MHbcRo7A2KwcMMc82Zq89qAOxZngHMkcCGDIMuwM6"
    )
    base_cov = decode_cov(base64.b64decode(base_b64))

    beta_insert = []
    for code, row in beta_rows.items():
        adjusted = dict(row)
        if code in {"000008.OF", "000042.OF"}:
            adjusted["SMB"] += 0.05
            adjusted["QMJ"] += 0.05
        if code in {"000051.OF", "000071.OF"}:
            adjusted["SMB"] -= 0.05
            adjusted["QMJ"] += 0.05
        if code in {"000055.OF", "000059.OF"}:
            adjusted["SMB"] += 0.05
            adjusted["QMJ"] -= 0.05

        cov = base_cov.copy()
        if code == "000059.OF":
            cov[1, 1] = cov[1, 1] * 400
            cov[3, 3] = cov[3, 3] * 400
        p_bin = _encode_cov_lower_triangle(cov)
        beta_insert.append(
            {
                "code": code,
                "date": date(2025, 12, 31),
                "MKT": adjusted["MKT"],
                "SMB": adjusted["SMB"],
                "HML": adjusted["HML"],
                "QMJ": adjusted["QMJ"],
                "const": adjusted["const"],
                "gamma": adjusted["gamma"],
                "P_bin": p_bin,
            }
        )

    with pg_engine.begin() as conn:
        conn.execute(fund_beta_table.insert(), beta_insert)

    base_u = float((np.sqrt(base_cov[1, 1]) + np.sqrt(base_cov[3, 3])) / 2.0)
    config = BucketRecoConfig.model_validate(
        {
            "proxy": {"weight_mode": "equal"},
            "trend": {"short_window": 3, "long_window": 6, "vol_window": 3, "eps": 1e-12},
            "consistency": {
                "min_points": 3,
                "min_coverage": 0.3,
                "windows": [{"label": "1M", "months": 1}],
            },
            "score": {
                "w_rho": 1.0,
                "w_h": 0.0,
                "lambdas": {"1M": 1.0},
                "top_k": 10,
                "score_threshold": 0.0,
                "min_count": 1,
            },
            "beta": {"u_mode": "absolute", "u_value": base_u * 2.0},
            "convex_hull": {
                "n": 3,
                "epsilon": 0.0,
                "M": 256,
                "rng_seed": 7,
                "topk_per_iter": 32,
                "violation_tol": 1e-9,
            },
        }
    )
    dsn = pg_engine.url.render_as_string(hide_password=False)
    app_config = AppConfig(
        env="test",
        db=DbConfig(dsn=dsn, schema_in="public", schema_out="cta"),
        logging=LoggingConfig(),
        bucket_reco=config,
    )
    result = run_bucket_asset_recommender(
        "GROWTH",
        ["000300.SH", "000905.SH", "000852.SH"],
        date(2025, 12, 31),
        app_config=app_config,
        config=config,
    )

    step1_candidates = set(result.step1["candidates"])
    assert step1_candidates == set(good)

    selected = result.step2["selected"]
    assert len(selected) == 3
    assert set(selected).issubset(set(good))

    details = result.step2["details"]
    assert details
    for item in details:
        assert "fund" in item
        assert "SMB" in item
        assert "QMJ" in item
        assert "p_SMB" in item
        assert "p_QMJ" in item
