# tests/fixtures

## 作用与架构定位

* 职责：算法测试固定输入与期望输出（golden files）。
* DOT 位置：Algorithm Tests / fixtures。

## 目录结构（至多两层）

```
.
|-- mini_market.csv
|-- mini_factors.csv
|-- mini_nav.csv
|-- mini_bucket.yaml
|-- golden/
`-- expected/
```

## 本模块约束/规范

* CSV/YAML 必须排序固定（date 升序、ticker 升序）。
* 数据规模小且可手算；不依赖真实 DB。
* expected/ 用于 golden files（features/weights 等），对齐列名/行数/主键。
* golden/ 用于 signal_weekly / portfolio_weight_weekly 的稳定快照。
