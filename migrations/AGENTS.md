# migrations

## 作用与架构定位

* 职责：数据库迁移（输出表 + bucket 表）。
* DOT 位置：`Repository Layer` 相关的 schema 变更。

## 目录结构（至多两层）

```
.
|-- README
|-- env.py
|-- script.py.mako
`-- versions/
```

## 文件说明

* `env.py`：读取 `config/app.yaml` 作为迁移连接来源。
* `versions/`：迁移脚本集合。

## 本模块约束/规范

* 避免 autogenerate 引入非本服务表的删除。
* 迁移必须幂等、安全（只包含本服务相关表）。
