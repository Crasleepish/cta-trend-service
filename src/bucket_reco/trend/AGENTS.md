# trend

## 作用与架构定位

* 职责：趋势分数、相关性/命中率、窗口管理。
* DOT 位置：Signal/CTA Core 辅助工具。

## 目录结构（至多两层）

```
.
|-- AGENTS.md
|-- trend_score.py
`-- consistency.py
```

## 文件说明

* `trend_score.py`：移动均值、滚动波动率、趋势分数计算。
* `consistency.py`：趋势相关性与命中率、窗口切片工具。

## 本模块约束/规范

* 只输出趋势相关中间量，不写权重生成。
