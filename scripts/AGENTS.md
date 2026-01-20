# scripts

## 作用与架构定位

* 职责：本地辅助脚本（读写冒烟、排查）。
* DOT 位置：`CLI Runner (optional)` 周边。

## 目录结构（至多两层）

```
.
`-- read_write_job_run.py
```

## 文件说明

* `read_write_job_run.py`：读取一段数据并写入 `job_run` 的冒烟脚本。

## 本模块约束/规范

* 脚本仅做本地验证，不承载业务逻辑。
* 若脚本需要参数说明，请在文件顶部注明。
