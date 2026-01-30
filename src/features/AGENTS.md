# src/features

## 作用与架构定位

* 职责：CTA 特征注册与计算（公式实现 + 采样规则）。
* DOT 位置：Domain Layer / FeatureService 公式实现组件。

## 目录结构（至多两层）

```
.
|-- __init__.py
|-- registry.py
|-- computer.py
`-- sampler.py
```

## 文件说明

* `registry.py`：特征注册表与 DataBundle 定义。
* `computer.py`：日频特征计算（log return / vol / trend / gate）。
* `sampler.py`：周频采样规则（最后交易日）。

## 本模块约束/规范

* 公式必须遵循白皮书（WP §4/§6/§11.1）。
* 保持纯函数与可单测；不做 DB 读写。
* 允许依赖 `pandas` / `numpy` 用于向量化计算。
