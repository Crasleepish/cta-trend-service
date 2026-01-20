# src/core

## 作用与架构定位

* 职责：核心基础设施（配置、日志、DB 引擎、输出表元数据）。
* DOT 位置：`AppConfig (app.yaml)`, `Local Logger`, `DB session/engine`。

## 目录结构（至多两层）

```
.
|-- __init__.py
|-- config.py
|-- db.py
|-- logging.py
`-- schema.py
```

## 文件说明

* `config.py`：加载 `config/app.yaml`（允许 `APP_CONFIG_PATH` 覆盖）。
* `logging.py`：结构化日志格式、日滚动、gzip、保留清理。
* `db.py`：创建同步 SQLAlchemy Engine + 连接检查。
* `schema.py`：输出表 Core 元数据（feature/signal/weight/job_run）。
* `__init__.py`：保持空，不放逻辑。

## 本模块约束/规范

* 不包含业务逻辑或计算公式。
* 不在 import 时触发副作用（连接/写入/环境读取）。
* 仅通过 `config.py` 读取配置，不从业务层直接读环境变量。
