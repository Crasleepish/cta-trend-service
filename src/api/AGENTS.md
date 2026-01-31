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

* `health.py`：`/health` 健康检查路由，返回状态/版本/时间/DB 连通性。
* `jobs.py`：`/jobs/*` 触发同步 job。
* `weights.py`：`/weights*` 权重查询。
* `runs.py`：`/runs*` 运行审计查询。
* `signals.py`：`/signals*` 信号查询。
* `schemas.py`：API 请求/响应模型。
* `errors.py`：API 错误模型与异常映射。
* `deps.py`：API 依赖构建（repos/services/job_runner）。
* `__init__.py`：保持空，不放逻辑。

## 本模块约束/规范

* 路由层不得直接访问数据库；必须通过 `services/`。
* 不引入计算公式或策略细节；仅负责输入输出边界。
* 如需日志，使用统一的 `logging`，不新增 ad-hoc logger。
