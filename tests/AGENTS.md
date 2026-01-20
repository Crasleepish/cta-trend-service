# tests

## 作用与架构定位

* 职责：单元测试与框架测试验证。
* DOT 位置：`Algorithm Tests`（当前仅框架层测试）。

## 目录结构（至多两层）

```
.
|-- conftest.py
|-- test_health.py
|-- test_job_runner.py
|-- test_repos_inputs.py
|-- test_repos_outputs.py
`-- test_run_audit_service.py
```

## 文件说明

* `conftest.py`：PostgresContainer 测试容器与 Engine fixtures。
* `test_health.py`：健康检查基础测试。
* `test_job_runner.py`：JobRunner 框架流程测试（mock services）。
* `test_repos_inputs.py`：输入 repos 读取测试（容器内 Postgres）。
* `test_repos_outputs.py`：输出 repos upsert 测试。
* `test_run_audit_service.py`：RunAuditService 生命周期测试。

## 本模块约束/规范

* 使用 `postgres:17` 的 Testcontainers。
* 若需要新增测试约束（如性能/数据规模），在此补充。
