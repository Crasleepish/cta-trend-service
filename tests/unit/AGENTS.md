# tests/unit

## 作用与架构定位

* 职责：纯函数单元测试（不依赖数据库）。
* DOT 位置：Algorithm Tests/单元层。

## 目录结构（至多两层）

```
.
|-- test_beta_ols.py
|-- test_beta_stability.py
|-- test_proxy_composite.py
|-- test_series_utils.py
|-- test_step1_score.py
|-- test_trend_consistency.py
`-- test_trend_score.py
```

## 本模块约束/规范

* 只测逻辑与边界；如需验证 DB 读取逻辑，使用 Testcontainers 的 fake DB（不直连生产库）。
* 使用与生产一致的字段语义（日期索引、价格/NAV 序列）。
