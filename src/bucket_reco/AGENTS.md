# src/bucket_reco

## 作用与架构定位

* 职责：Bucket Tradable Assets 推荐工具的子模块集合。
* DOT 位置：辅助工具（不直接进入主服务分层）。

## 目录结构（至多两层）

```
.
|-- runner.py
|-- proxy/
|-- trend/
|-- score/
|-- beta/
`-- report/
```

## 本模块约束/规范

* `runner.py`：端到端推荐执行入口（读取 fund_beta/fund_hist/index_hist + 编排计算）。
* runner 允许读取数据库；其它模块保持纯计算工具。
