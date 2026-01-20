# src/bucket_reco

## 作用与架构定位

* 职责：Bucket Tradable Assets 推荐工具的子模块集合。
* DOT 位置：辅助工具（不直接进入主服务分层）。

## 目录结构（至多两层）

```
.
|-- proxy/
|-- trend/
|-- score/
|-- beta/
`-- report/
```

## 本模块约束/规范

* 仅提供可复用的分析工具，不引入服务层或 DB 依赖。
