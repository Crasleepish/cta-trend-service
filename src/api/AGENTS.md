# src/api

## 作用与架构定位

* 职责：FastAPI 路由层，只做请求/响应编排与参数校验，不承载业务逻辑。
* DOT 位置：`API Layer (FastAPI Routes)`。

## 目录结构（至多两层）

```
.
|-- __init__.py
`-- health.py
```

## 文件说明

* `health.py`：`/health` 健康检查路由，返回 `ok`。
* `__init__.py`：保持空，不放逻辑。

## 本模块约束/规范

* 路由层不得直接访问数据库；必须通过 `services/`。
* 不引入计算公式或策略细节；仅负责输入输出边界。
* 如需日志，使用统一的 `logging`，不新增 ad-hoc logger。
