from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import date
from pathlib import Path

from sqlalchemy import create_engine

from ..core.config import load_app_config
from ..repo.inputs import NavRepo, TradeCalendarRepo
from ..repo.outputs import WeightRepo
from ..services.backtest_service import BacktestRequest, BacktestService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run backtest task")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--strategy-id")
    parser.add_argument("--version")
    parser.add_argument("--portfolio-id")
    parser.add_argument("--output-dir")
    parser.add_argument("--buy-fee", type=float)
    parser.add_argument("--sell-fee", type=float)
    parser.add_argument("--slippage", type=float)
    parser.add_argument("--init-cash", type=float)
    parser.add_argument("--cash-sharing", type=str)
    parser.add_argument("--freq")
    return parser.parse_args()


def _parse_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise ValueError("cash-sharing must be true/false")


def main() -> None:
    args = parse_args()
    config = load_app_config()
    engine = create_engine(config.db.dsn)

    service = BacktestService(
        weight_repo=WeightRepo(engine, schema=config.db.schema_out),
        nav_repo=NavRepo(engine, schema=config.db.schema_in),
        calendar_repo=TradeCalendarRepo(engine, schema=config.db.schema_in),
    )

    req = BacktestRequest(
        start_date=date.fromisoformat(args.start_date),
        end_date=date.fromisoformat(args.end_date),
        strategy_id=args.strategy_id or config.strategy.default_strategy_id,
        version=args.version or config.strategy.default_version,
        portfolio_id=args.portfolio_id or config.strategy.default_portfolio_id,
        output_dir=Path(args.output_dir or config.backtest.output_dir),
        buy_fee=args.buy_fee if args.buy_fee is not None else config.backtest.buy_fee,
        sell_fee=args.sell_fee if args.sell_fee is not None else config.backtest.sell_fee,
        slippage=args.slippage if args.slippage is not None else config.backtest.slippage,
        init_cash=args.init_cash if args.init_cash is not None else config.backtest.init_cash,
        cash_sharing=(
            _parse_bool(args.cash_sharing)
            if args.cash_sharing is not None
            else config.backtest.cash_sharing
        ),
        freq=args.freq or config.backtest.freq,
    )

    result = service.run_backtest_task(req)
    print("backtest artifacts:")
    print(asdict(result.artifacts))
    if result.warnings:
        print("warnings:", result.warnings)


if __name__ == "__main__":
    main()
