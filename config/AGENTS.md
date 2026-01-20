# config

## 作用与架构定位

* 职责：配置文件存放目录。
* DOT 位置：`AppConfig (app.yaml)`。

## 目录结构（至多两层）

```
.
|-- app.yaml
`-- app.yaml.template
```

## 文件说明

* `app.yaml`：本地环境配置（含敏感信息，不提交）。
* `app.yaml.template`：模板配置（可提交）。

## 本模块约束/规范

* 任何敏感配置只放 `app.yaml`，禁止提交。
* 新增配置需同步更新模板文件。
