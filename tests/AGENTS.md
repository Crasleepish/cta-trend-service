# tests

## 作用与架构定位

* 职责：单元测试与框架测试验证。
* DOT 位置：`Algorithm Tests`（当前仅框架层测试）。

## 目录结构（至多两层）

```
.
|-- conftest.py
|-- e2e/
|-- fixtures/
|-- formula/
|-- helpers/
|-- integration/
|-- perf/
|-- regression/
|-- test_health.py
|-- test_job_runner.py
|-- test_repos_inputs.py
|-- test_repos_outputs.py
`-- test_run_audit_service.py
```

## 文件说明

* `conftest.py`：PostgresContainer 测试容器与 Engine fixtures。
* `e2e/`：确定性与幂等性 E2E（mock services + fake repos）。
* `fixtures/`：小型可手算输入与 golden outputs。
* `formula/`：白皮书公式一致性测试（章节名）。
* `helpers/db.py`：测试用 DB 表结构辅助（复用 fund_beta 等表建表逻辑）。
* `integration/`：集成测试（保留占位）。
* `perf/`：性能基线测试（可选）。
* `regression/`：回归快照（可选）。
* `test_health.py`：健康检查基础测试。
* `test_job_runner.py`：JobRunner 框架流程测试（mock services）。
* `test_repos_inputs.py`：输入 repos 读取测试（容器内 Postgres）。
* `test_repos_outputs.py`：输出 repos upsert 测试。
* `test_run_audit_service.py`：RunAuditService 生命周期测试。

## 本模块约束/规范

* 测试代码执行时使用 fake/mocked 数据源；可读取真实数据库样例用于填充假库，禁止对真实数据库写入。
* 真实数据库样例读取需通过 MCP 数据库工具完成，避免直连在测试内执行；开发/设计阶段可访问真实 DB 以理解生产行为。
* 若使用容器数据库，仅限 Testcontainers 本地实例。
* 若需要新增测试约束（如性能/数据规模），在此补充。
