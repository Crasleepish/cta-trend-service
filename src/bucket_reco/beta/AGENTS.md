# beta

## 作用与架构定位

* 职责：beta 稳定性过滤、鲁棒标准化、凸包代表选择（暴露向量字段可配置）。
* DOT 位置：因子/风险结构辅助工具。

## 目录结构（至多两层）

```
.
|-- AGENTS.md
|-- convex_hull.py
|-- ols.py
`-- stability.py
```

## 文件说明

* `convex_hull.py`：贪心凸包代表选择（与 `docs/greedy_convex_hull_volume.py` 一致）。
* `ols.py`：OLS beta 估计与正向门控。
* `stability.py`：协方差解码、不稳定度过滤、MAD 缩放。

## 本模块约束/规范

* 标准化与凸包选择必须可重复、可审计。
