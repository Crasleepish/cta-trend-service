# beta

## 作用与架构定位

* 职责：beta 稳定性过滤、鲁棒标准化、Ward 聚类、代表/TopK。
* DOT 位置：因子/风险结构辅助工具。

## 目录结构（至多两层）

```
.
|-- AGENTS.md
|-- clustering.py
|-- ols.py
`-- stability.py
```

## 文件说明

* `clustering.py`：Ward 聚类、最小簇约束、代表/Top-K 选择。
* `ols.py`：OLS beta 估计与正向门控。
* `stability.py`：协方差解码、不稳定度过滤、MAD 缩放、L2 单位化。

## 本模块约束/规范

* 聚类/标准化必须可重复、可审计。
