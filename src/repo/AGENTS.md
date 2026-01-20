# src/repo

## 作用与架构定位

* 职责：数据访问层，负责读输入表与写输出表（无业务逻辑）。
* DOT 位置：`Repository Layer (DB 读写)`。

## 目录结构（至多两层）

```
.
|-- __init__.py
|-- base.py
|-- inputs.py
`-- outputs.py
```

## 文件说明

* `base.py`：共享 DB 执行/查询辅助方法。
* `inputs.py`：只读输入表 repos（bucket/market/factor/nav/beta/aux）。
* `outputs.py`：写输出表 repos（feature/signal/weight/run），包含 upsert。
* `__init__.py`：保持空，不放逻辑。

## 本模块约束/规范

* 只写 SQLAlchemy Core（`Table/select/insert/update`）。
* 读写必须显式，不使用隐式全局状态。
* 读写职责清晰：输入只读、输出 upsert。
