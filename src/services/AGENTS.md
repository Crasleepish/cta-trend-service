# src/services

## 作用与架构定位

* 职责：核心编排层（Feature/Signal/Portfolio/Backtest 等）。
* DOT 位置：Domain Layer（计算编排）。

## 目录结构（至多两层）

```
.
|-- backtest_service.py
|-- auto_param_service.py
|-- feature_service.py
|-- job_runner.py
|-- portfolio_service.py
|-- contracts.py
|-- run_audit_service.py
`-- signal_service.py
```

## 文件说明

* `backtest_service.py`：回测任务编排与报告输出（调用 backtest engine）。
* `auto_param_service.py`：基于历史数据自动估计参数并写入 `config/auto_params.json`。
* `feature_service.py`：特征计算编排与落库（feature_daily + weekly sample）。
* `signal_service.py`：信号决策与倾斜层产出（signal_weekly）。
* `portfolio_service.py`：组合权重层（只读 signal_weekly）。
* `contracts.py`：运行契约（RunContext/RunRequest/RunResult 等）。
* `job_runner.py`：同步执行编排与审计。
* `run_audit_service.py`：run_id 生命周期管理。

## 本模块约束/规范

* 服务只做编排，不内嵌公式细节。
* 不直接读取 env，配置由 AppConfig 注入。
* 运行失败必须写入 `job_run.error_stack`，不得吞异常。
