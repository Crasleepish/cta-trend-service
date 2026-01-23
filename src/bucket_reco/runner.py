from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sqlalchemy import Column, Date, Float, MetaData, String, Table, create_engine, select
from sqlalchemy.engine import Engine

from src.bucket_reco.beta.convex_hull import select_representatives
from src.bucket_reco.beta.ols import ols_beta
from src.bucket_reco.beta.stability import (
    UnstableRule,
    decode_cov,
    extract_var_2,
    fetch_beta_rows,
    filter_unstable,
    robust_scale_mad,
)
from src.bucket_reco.proxy.composite import (
    build_composite_proxy,
    compute_vol_annualized,
    compute_weights_equal,
    compute_weights_inv_vol,
)
from src.bucket_reco.score.step1 import (
    CandidateConstraints,
    WindowScore,
    aggregate_scores,
    score_window,
    select_candidates,
)
from src.bucket_reco.trend.consistency import hit_ratio, trend_corr, window_slices
from src.bucket_reco.trend.trend_score import trend_score
from src.core.config import AppConfig, BucketRecoConfig, BucketRecoWindowSpec, load_app_config
from src.core.logging import setup_logging
from src.utils.series import align_series, log_return


@dataclass(frozen=True)
class BucketRecoResult:
    proxy: dict[str, Any]
    step1: dict[str, Any]
    step2: dict[str, Any]
    recommendations: list[str]


def run_bucket_asset_recommender(
    bucket_name: str,
    list_of_proxy_index: Sequence[str],
    as_of_date: date,
    *,
    app_config: AppConfig | None = None,
    config: BucketRecoConfig | None = None,
) -> BucketRecoResult:
    logger = logging.getLogger(__name__)
    app_config = app_config or load_app_config()
    cfg = config or app_config.bucket_reco

    engine = create_engine(app_config.db.dsn)
    try:
        logger.info(
            "bucket_reco start",
            extra={
                "bucket": bucket_name,
                "proxy_index": list(list_of_proxy_index),
                "as_of_date": as_of_date.isoformat(),
            },
        )
        assets = _fetch_universe_codes(engine, app_config.db.schema_in, as_of_date)
        if not assets:
            raise ValueError("no fund universe found for as_of_date")
        logger.info("fund universe loaded", extra={"count": len(assets)})

        max_months = max(spec.months for spec in cfg.consistency.windows)
        start_date = (pd.Timestamp(as_of_date) - pd.DateOffset(months=max_months)).date()

        proxy_prices = _fetch_index_prices(
            engine, app_config.db.schema_in, list_of_proxy_index, start_date, as_of_date
        )
        proxy_returns, proxy_index, proxy_weights = _build_proxy(cfg, proxy_prices)
        logger.info(
            "proxy built",
            extra={
                "start_date": proxy_index.index.min().date().isoformat(),
                "end_date": proxy_index.index.max().date().isoformat(),
                "weights": proxy_weights,
            },
        )
        T_G = trend_score(
            proxy_index,
            cfg.trend.short_window,
            cfg.trend.long_window,
            cfg.trend.vol_window,
            eps=cfg.trend.eps,
        )
        windows = window_slices(
            proxy_index.index, [spec.model_dump() for spec in cfg.consistency.windows]
        )

        fund_prices = _fetch_fund_navs(
            engine, app_config.db.schema_in, assets, start_date, as_of_date
        )
        fund_universe = list(fund_prices.keys())
        step1_details, s_fit_scores = _compute_step1(
            fund_prices,
            proxy_returns,
            T_G,
            windows,
            cfg,
        )
        constraints = CandidateConstraints(
            top_k=cfg.score.top_k,
            score_threshold=cfg.score.score_threshold,
            min_count=cfg.score.min_count,
        )
        candidates = _select_step1_candidates(fund_universe, s_fit_scores, constraints)
        logger.info(
            "step1 complete",
            extra={
                "universe": len(fund_universe),
                "candidates": len(candidates),
                "score_threshold": cfg.score.score_threshold,
                "min_count": cfg.score.min_count,
            },
        )

        step2 = _compute_step2(
            engine,
            app_config.db.schema_in,
            candidates,
            as_of_date,
            s_fit_scores,
            cfg,
        )
        logger.info(
            "step2 complete",
            extra={
                "selected": len(step2.get("selected", [])),
                "filtered": len(step2.get("filtered", [])),
            },
        )

        proxy_info = {
            "index_codes": list(list_of_proxy_index),
            "weights": proxy_weights,
            "start_date": proxy_index.index.min().date(),
            "end_date": proxy_index.index.max().date(),
        }
        return BucketRecoResult(
            proxy=proxy_info,
            step1={"candidates": candidates, "details": step1_details},
            step2=step2,
            recommendations=step2["selected"],
        )
    finally:
        engine.dispose()


def _fetch_universe_codes(engine: Engine, schema: str, as_of_date: date) -> list[str]:
    metadata = MetaData(schema=schema)
    fund_beta = Table(
        "fund_beta",
        metadata,
        Column("code", String(64)),
        Column("date", Date),
    )
    stmt = (
        select(fund_beta.c.code)
        .where(fund_beta.c.date <= as_of_date)
        .distinct()
        .order_by(fund_beta.c.code)
    )
    with engine.connect() as conn:
        rows = conn.execute(stmt).all()
    return [str(row[0]) for row in rows]


def _fetch_index_prices(
    engine: Engine,
    schema: str,
    index_codes: Sequence[str],
    start_date: date,
    end_date: date,
) -> dict[str, pd.Series]:
    metadata = MetaData(schema=schema)
    index_hist = Table(
        "index_hist",
        metadata,
        Column("index_code", String(64)),
        Column("date", Date),
        Column("close", Float),
    )
    stmt = (
        select(index_hist.c.index_code, index_hist.c.date, index_hist.c.close)
        .where(index_hist.c.index_code.in_(index_codes))
        .where(index_hist.c.date.between(start_date, end_date))
        .order_by(index_hist.c.index_code, index_hist.c.date)
    )
    rows = []
    with engine.connect() as conn:
        for row in conn.execute(stmt):
            rows.append(row)

    prices: dict[str, list[tuple[pd.Timestamp, float]]] = {}
    for code, dt, close in rows:
        prices.setdefault(code, []).append((pd.Timestamp(dt), float(close)))

    series: dict[str, pd.Series] = {}
    for code, points in prices.items():
        idx = pd.DatetimeIndex([p[0] for p in points])
        series[code] = pd.Series([p[1] for p in points], index=idx, name=code)
    if not series:
        raise ValueError("no index price data found")
    return series


def _fetch_fund_navs(
    engine: Engine,
    schema: str,
    fund_codes: Sequence[str],
    start_date: date,
    end_date: date,
) -> dict[str, pd.Series]:
    metadata = MetaData(schema=schema)
    fund_hist = Table(
        "fund_hist",
        metadata,
        Column("fund_code", String(64)),
        Column("date", Date),
        Column("net_value", Float),
    )
    stmt = (
        select(fund_hist.c.fund_code, fund_hist.c.date, fund_hist.c.net_value)
        .where(fund_hist.c.fund_code.in_(fund_codes))
        .where(fund_hist.c.date.between(start_date, end_date))
        .order_by(fund_hist.c.fund_code, fund_hist.c.date)
    )
    rows = []
    with engine.connect() as conn:
        for row in conn.execute(stmt):
            rows.append(row)

    prices: dict[str, list[tuple[pd.Timestamp, float]]] = {}
    for code, dt, nav in rows:
        prices.setdefault(code, []).append((pd.Timestamp(dt), float(nav)))

    series: dict[str, pd.Series] = {}
    for code, points in prices.items():
        idx = pd.DatetimeIndex([p[0] for p in points])
        series[code] = pd.Series([p[1] for p in points], index=idx, name=code)
    return series


def _build_proxy(
    cfg: BucketRecoConfig,
    price_dict: Mapping[str, pd.Series],
) -> tuple[pd.Series, pd.Series, dict[str, float]]:
    if cfg.proxy.weight_mode == "equal":
        weights_vec = compute_weights_equal(len(price_dict))
    else:
        vols = []
        for series in price_dict.values():
            returns = log_return(series).dropna()
            vols.append(compute_vol_annualized(returns, annualize=cfg.proxy.annualize))
        clip = None
        if cfg.proxy.clip_min is not None or cfg.proxy.clip_max is not None:
            clip = (cfg.proxy.clip_min, cfg.proxy.clip_max)
        weights_vec = compute_weights_inv_vol(np.asarray(vols, dtype=float), clip=clip)
    weights = {code: float(weights_vec[i]) for i, code in enumerate(price_dict.keys())}
    returns, index = build_composite_proxy(price_dict, weights, join=cfg.proxy.join)
    return returns, index, weights


def _compute_step1(
    fund_prices: Mapping[str, pd.Series],
    proxy_returns: pd.Series,
    T_G: pd.Series,
    windows: Mapping[str, pd.DatetimeIndex],
    cfg: BucketRecoConfig,
) -> tuple[list[dict[str, Any]], dict[str, WindowScore]]:
    logger = logging.getLogger(__name__)
    details: list[dict[str, Any]] = []
    scores: dict[str, WindowScore] = {}

    proxy_returns = proxy_returns.rename("proxy")
    for fund, price_series in fund_prices.items():
        T_i = trend_score(
            price_series,
            cfg.trend.short_window,
            cfg.trend.long_window,
            cfg.trend.vol_window,
            eps=cfg.trend.eps,
        )
        fund_returns = log_return(price_series).rename("fund").dropna()
        window_scores: dict[str, WindowScore] = {}
        window_detail: dict[str, dict[str, Any]] = {}
        for label, window_idx in windows.items():
            rho = trend_corr(
                T_i,
                T_G,
                window_idx,
                min_points=cfg.consistency.min_points,
                min_coverage=cfg.consistency.min_coverage,
            )
            hr = hit_ratio(
                T_i,
                T_G,
                window_idx,
                min_points=cfg.consistency.min_points,
                min_coverage=cfg.consistency.min_coverage,
            )
            beta = _compute_beta(fund_returns, proxy_returns, window_idx)
            if rho is None or hr is None or beta is None:
                ws = WindowScore(score=float("nan"), valid=False)
            else:
                ws = score_window(rho, hr, beta, cfg.score.w_rho, cfg.score.w_h)
            window_scores[label] = ws
            window_detail[label] = {
                "rho_T": rho,
                "HR": hr,
                "beta": beta,
                "score": ws.score,
                "valid": ws.valid,
            }
        agg = aggregate_scores(window_scores, cfg.score.lambdas)
        scores[fund] = agg
        details.append(
            {"fund": fund, "windows": window_detail, "S_fit": agg.score, "valid": agg.valid}
        )
    if logger.isEnabledFor(logging.DEBUG):
        valid_scores = [
            (fund, ws.score) for fund, ws in scores.items() if ws.valid and np.isfinite(ws.score)
        ]
        valid_scores.sort(key=lambda x: x[1], reverse=True)
        logger.debug(
            "step1 score distribution",
            extra={
                "valid": len(valid_scores),
                "top": valid_scores[:10],
            },
        )
    return details, scores


def _compute_beta(
    fund_returns: pd.Series,
    proxy_returns: pd.Series,
    window_idx: pd.DatetimeIndex,
) -> float | None:
    aligned = align_series(
        [fund_returns.reindex(window_idx), proxy_returns.reindex(window_idx)], join="inner"
    ).dropna()
    if aligned.empty:
        return None
    y = aligned.iloc[:, 0].to_numpy()
    x = aligned.iloc[:, 1].to_numpy()
    try:
        _, beta = ols_beta(y, x)
    except ValueError:
        return None
    return float(beta)


def _select_step1_candidates(
    funds: Sequence[str],
    scores: Mapping[str, WindowScore],
    constraints: CandidateConstraints,
) -> list[str]:
    return select_candidates(funds, scores, constraints)


def _compute_step2(
    engine: Engine,
    schema: str,
    candidates: Sequence[str],
    as_of_date: date,
    s_fit_scores: Mapping[str, WindowScore],
    cfg: BucketRecoConfig,
) -> dict[str, Any]:
    logger = logging.getLogger(__name__)
    if not candidates:
        return {"selected": [], "representatives": [], "details": [], "filtered": []}

    rows = fetch_beta_rows(candidates, as_of_date, engine=engine, schema=schema)
    if rows.empty:
        return {
            "selected": [],
            "representatives": [],
            "details": [],
            "filtered": list(candidates),
        }

    beta_points = []
    U_values = {}
    filtered = []
    missing_pbin = 0
    decode_errors = 0
    for _, row in rows.iterrows():
        code = str(row["code"])
        p_bin = row.get("P_bin")
        if p_bin is None:
            filtered.append(code)
            missing_pbin += 1
            continue
        try:
            cov = decode_cov(p_bin, strict=cfg.beta.strict_decode)
            var_smb, var_qmj = extract_var_2(cov)
            U_values[code] = float((np.sqrt(var_smb) + np.sqrt(var_qmj)) / 2.0)
        except ValueError:
            filtered.append(code)
            decode_errors += 1
            continue
        beta_points.append({"code": code, "SMB": row["SMB"], "QMJ": row["QMJ"]})

    if not beta_points:
        return {"selected": [], "representatives": [], "details": [], "filtered": filtered}

    beta_df = pd.DataFrame(beta_points).set_index("code")
    U = pd.Series(U_values)
    stable_idx = filter_unstable(U, UnstableRule(cfg.beta.u_mode, cfg.beta.u_value))
    unstable = set(U.index) - set(stable_idx)
    filtered.extend(sorted(unstable))
    logger.info(
        "beta stability filter",
        extra={
            "rows": len(rows),
            "usable": len(beta_df),
            "missing_pbin": missing_pbin,
            "decode_errors": decode_errors,
            "unstable": len(unstable),
        },
    )
    beta_df = beta_df.loc[beta_df.index.intersection(stable_idx)]

    if beta_df.empty:
        return {"selected": [], "representatives": [], "details": [], "filtered": filtered}

    scaled, scales = robust_scale_mad(beta_df, eps=cfg.beta.mad_eps)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("beta scales", extra={"scales": {k: float(v) for k, v in scales.items()}})
    points = scaled
    target_n = min(cfg.convex_hull.n, len(points))
    if target_n <= 0:
        return {"selected": [], "representatives": [], "details": [], "filtered": filtered}

    init_size = min(2, len(points))
    max_iters = cfg.convex_hull.max_iters
    if max_iters is None:
        max_iters = max(target_n - init_size, 0)

    selected_idx = select_representatives(
        points.to_numpy(),
        cfg.convex_hull.epsilon,
        M=cfg.convex_hull.M,
        rng_seed=cfg.convex_hull.rng_seed,
        topk_per_iter=cfg.convex_hull.topk_per_iter,
        violation_tol=cfg.convex_hull.violation_tol,
        max_iters=max_iters,
        clip_rhopow=cfg.convex_hull.clip_rhopow,
        clip_viol=cfg.convex_hull.clip_viol,
        diversity_beta=cfg.convex_hull.diversity_beta,
        nms_cos_thresh=cfg.convex_hull.nms_cos_thresh,
        labels=[str(code) for code in points.index],
        logger=logger if logger.isEnabledFor(logging.DEBUG) else None,
        debug=logger.isEnabledFor(logging.DEBUG),
    )

    selected_codes = [str(points.index[i]) for i in selected_idx][:target_n]
    logger.info(
        "convex_hull selected",
        extra={
            "selected": selected_codes,
            "target_n": target_n,
        },
    )
    details: list[dict[str, Any]] = []
    for code in points.index:
        score = s_fit_scores.get(code)
        details.append(
            {
                "fund": str(code),
                "SMB": float(beta_df.loc[code, "SMB"]),
                "QMJ": float(beta_df.loc[code, "QMJ"]),
                "p_SMB": float(points.loc[code, "SMB"]),
                "p_QMJ": float(points.loc[code, "QMJ"]),
                "S_fit": None if score is None else float(score.score),
                "U": float(U.get(code, np.nan)),
                "selected": str(code) in selected_codes,
            }
        )

    return {
        "selected": selected_codes,
        "representatives": selected_codes,
        "details": details,
        "filtered": filtered,
        "scales": {k: float(v) for k, v in scales.items()},
    }


def _parse_window_specs(text: str | None) -> list[dict[str, Any]] | None:
    if not text:
        return None
    specs = []
    for item in text.split(","):
        label, months = item.split(":")
        specs.append({"label": label.strip(), "months": int(months)})
    return specs


def _apply_overrides(cfg: BucketRecoConfig, args: argparse.Namespace) -> BucketRecoConfig:
    updated = cfg.model_copy(deep=True)
    if args.proxy_weight_mode:
        updated.proxy.weight_mode = args.proxy_weight_mode
    if args.proxy_clip_min is not None:
        updated.proxy.clip_min = args.proxy_clip_min
    if args.proxy_clip_max is not None:
        updated.proxy.clip_max = args.proxy_clip_max
    if args.trend_short_window is not None:
        updated.trend.short_window = args.trend_short_window
    if args.trend_long_window is not None:
        updated.trend.long_window = args.trend_long_window
    if args.trend_vol_window is not None:
        updated.trend.vol_window = args.trend_vol_window
    if args.trend_eps is not None:
        updated.trend.eps = args.trend_eps
    if args.consistency_min_points is not None:
        updated.consistency.min_points = args.consistency_min_points
    if args.consistency_min_coverage is not None:
        updated.consistency.min_coverage = args.consistency_min_coverage
    if args.window_specs:
        updated.consistency.windows = [
            BucketRecoWindowSpec.model_validate(spec)
            for spec in _parse_window_specs(args.window_specs) or []
        ]
    if args.score_w_rho is not None:
        updated.score.w_rho = args.score_w_rho
    if args.score_w_h is not None:
        updated.score.w_h = args.score_w_h
    if args.score_threshold is not None:
        updated.score.score_threshold = args.score_threshold
    if args.score_top_k is not None:
        updated.score.top_k = args.score_top_k
    if args.score_min_count is not None:
        updated.score.min_count = args.score_min_count
    if args.hull_n is not None:
        updated.convex_hull.n = args.hull_n
    if args.hull_epsilon is not None:
        updated.convex_hull.epsilon = args.hull_epsilon
    if args.hull_M is not None:
        updated.convex_hull.M = args.hull_M
    if args.hull_rng_seed is not None:
        updated.convex_hull.rng_seed = args.hull_rng_seed
    if args.hull_topk_per_iter is not None:
        updated.convex_hull.topk_per_iter = args.hull_topk_per_iter
    if args.hull_violation_tol is not None:
        updated.convex_hull.violation_tol = args.hull_violation_tol
    if args.hull_max_iters is not None:
        updated.convex_hull.max_iters = args.hull_max_iters
    if args.hull_clip_rhopow is not None:
        updated.convex_hull.clip_rhopow = args.hull_clip_rhopow
    if args.hull_clip_viol is not None:
        updated.convex_hull.clip_viol = args.hull_clip_viol
    if args.hull_diversity_beta is not None:
        updated.convex_hull.diversity_beta = args.hull_diversity_beta
    if args.hull_nms_cos is not None:
        updated.convex_hull.nms_cos_thresh = args.hull_nms_cos
    return updated


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bucket Tradable Assets recommender")
    parser.add_argument("--bucket-name", required=True)
    parser.add_argument("--proxy-index", required=True, help="comma-separated index codes")
    parser.add_argument("--as-of-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--config", default=None, help="path to app.yaml")
    parser.add_argument("--proxy-weight-mode", choices=["equal", "inv_vol"])
    parser.add_argument("--proxy-clip-min", type=float)
    parser.add_argument("--proxy-clip-max", type=float)
    parser.add_argument("--trend-short-window", type=int)
    parser.add_argument("--trend-long-window", type=int)
    parser.add_argument("--trend-vol-window", type=int)
    parser.add_argument("--trend-eps", type=float)
    parser.add_argument("--consistency-min-points", type=int)
    parser.add_argument("--consistency-min-coverage", type=float)
    parser.add_argument("--window-specs", help="e.g. 3M:3,12M:12")
    parser.add_argument("--score-w-rho", type=float)
    parser.add_argument("--score-w-h", type=float)
    parser.add_argument("--score-threshold", type=float)
    parser.add_argument("--score-top-k", type=int)
    parser.add_argument("--score-min-count", type=int)
    parser.add_argument("--hull-n", type=int)
    parser.add_argument("--hull-epsilon", type=float)
    parser.add_argument("--hull-M", type=int)
    parser.add_argument("--hull-rng-seed", type=int)
    parser.add_argument("--hull-topk-per-iter", type=int)
    parser.add_argument("--hull-violation-tol", type=float)
    parser.add_argument("--hull-max-iters", type=int)
    parser.add_argument("--hull-clip-rhopow", type=float)
    parser.add_argument("--hull-clip-viol", type=float)
    parser.add_argument("--hull-diversity-beta", type=float)
    parser.add_argument("--hull-nms-cos", type=float)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    config_path = Path(args.config) if args.config else None
    app_config = load_app_config(config_path)
    setup_logging(app_config.logging)
    cfg = _apply_overrides(app_config.bucket_reco, args)
    result = run_bucket_asset_recommender(
        args.bucket_name,
        [code.strip() for code in args.proxy_index.split(",") if code.strip()],
        date.fromisoformat(args.as_of_date),
        app_config=app_config,
        config=cfg,
    )
    logger = logging.getLogger(__name__)
    logger.info(
        "bucket reco result",
        extra={"recommendations": result.recommendations},
    )
    logger.debug("bucket reco result details", extra={"result": result})


if __name__ == "__main__":
    main()
