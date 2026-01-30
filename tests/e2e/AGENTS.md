# tests/e2e

## 作用与架构定位

* 职责：确定性与幂等性 E2E 测试（mock services + fake repos）。
* DOT 位置：Algorithm Tests / Determinism E2E。

## 目录结构（至多两层）

```
.
|-- test_e2e_determinism_full.py
`-- test_e2e_rerun_idempotent_upsert.py
```

## 本模块约束/规范

* 先用 mock services 验证 determinism + upsert + 审计链路。
* 不实现“迷你公式”；真实公式 E2E 需等 Feature/Portfolio 完成后补充。
* 允许使用 fixtures 中的小型 CSV/YAML 数据作为输入。
