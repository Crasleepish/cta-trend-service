# src/services

## 作用与架构定位

* 职责：同步编排与审计框架（不含计算公式）。
* DOT 位置：`Domain Layer (计算编排)`, `RunAuditService`。

## 目录结构（至多两层）

```
.
|-- contracts.py
|-- job_runner.py
|-- feature_service.py
|-- signal_service.py
|-- portfolio_service.py
`-- run_audit_service.py
```

## 文件说明

* `contracts.py`：RunContext/RunRequest/RunResult 与 JobType/RunStatus。
* `feature_service.py`：特征计算编排与落库（feature_daily + weekly sample）。
* `signal_service.py`：信号决策与倾斜层产出（signal_weekly）。
* `portfolio_service.py`：组合权重层（只读 signal_weekly）。
* `run_audit_service.py`：job_run 运行审计（RUNNING/SUCCESS/FAILED）。
* `job_runner.py`：同步任务编排（FEATURE/SIGNAL/PORTFOLIO/FULL）。

## 本模块约束/规范

* 不实现白皮书公式，所有计算细节留给后续服务层。
* 仅负责解析输入、覆盖检查、编排调用、审计落库。
* 错误必须写入 `job_run.error_stack`。
