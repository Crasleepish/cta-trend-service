# src/backtest

## 作用与架构定位

* 职责：回测任务编排与报告输出（使用既定回测引擎）。
* DOT 位置：服务编排的旁路工具层（不参与实时信号/权重计算）。

## 目录结构（至多两层）

```
.
|-- __init__.py
|-- engine.py
|-- runner.py
`-- README.md
```

## 文件说明

* `engine.py`：回测引擎（从 docs/backtest_engine.py 复制，逻辑保持一致）。
* `runner.py`：CLI 入口，解析参数并调用 BacktestService。
* `README.md`：模块说明与注意事项。
* `__init__.py`：模块导出。

## 本模块约束/规范

* 引擎逻辑保持不变，仅做调用与输出编排。
* 只读数据，不写生产数据库。
* 输出文件路径与格式稳定（便于回归对比）。
