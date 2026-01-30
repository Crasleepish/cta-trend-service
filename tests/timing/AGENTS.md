# tests/timing

## 作用与架构定位

* 职责：日频计算 / 周频采样的时序契约测试（WP §11.1）。
* DOT 位置：Algorithm Tests / Timing Contract。

## 目录结构（至多两层）

```
.
`-- test_wp11_timing_contract.py
```

## 本模块约束/规范

* 必须显式验证“每周最后一个交易日”的采样规则。
* 允许使用极小日期样本；结果必须可手算。
* 测试使用固定输入（CSV/YAML fixture），不依赖真实 DB。
