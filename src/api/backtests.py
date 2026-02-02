from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends

from ..core.config import AppConfig
from ..services.backtest_service import BacktestRequest, BacktestService
from .deps import build_backtest_service, get_config
from .schemas import BacktestRequestBody, BacktestResponse

router = APIRouter(prefix="/backtests", tags=["backtests"])


@router.post("/run", response_model=BacktestResponse)
def run_backtest(
    body: BacktestRequestBody,
    config: AppConfig = Depends(get_config),
    service: BacktestService = Depends(build_backtest_service),
) -> BacktestResponse:
    req = BacktestRequest(
        start_date=body.start_date,
        end_date=body.end_date,
        strategy_id=body.strategy_id or config.strategy.default_strategy_id,
        version=body.version or config.strategy.default_version,
        portfolio_id=body.portfolio_id or config.strategy.default_portfolio_id,
        output_dir=Path(body.output_dir or config.backtest.output_dir),
        buy_fee=body.buy_fee,
        sell_fee=body.sell_fee,
        slippage=body.slippage,
        init_cash=body.init_cash,
        cash_sharing=body.cash_sharing,
        freq=body.freq,
    )
    result = service.run_backtest_task(req)
    return BacktestResponse(
        weights_csv=str(result.artifacts.weights_csv),
        nav_returns_csv=str(result.artifacts.nav_returns_csv),
        equity_curve_html=str(result.artifacts.equity_curve_html),
        report_md=str(result.artifacts.report_md),
        warnings=result.warnings,
        stats=result.stats,
    )
