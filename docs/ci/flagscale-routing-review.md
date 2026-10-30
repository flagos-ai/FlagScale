# FlagScale #1315：统一测试策略、两级路由与执行证据

本文描述本轮本地整改。参考 Megatron-LM-FL #203 的单策略入口、规划身份和执行证据理念；不复制其组布局、L1/L2 定义、重试、覆盖率口径或未知源码阻塞策略。

## 改造前的问题与对应修改

| 原问题 | 本轮修改 | 原因 |
| --- | --- | --- |
| shadow 与仓库变量使 PR 默认仍跑全量 | 删除 rollout、shadow/enforce 和变量读取 | 默认正式执行确定性路由；不要求维护者增加开关 |
| 两份手工 JSON 与固定 Python 字典重复维护 | 合入 project.yaml，改为按规则发现测试 | 普通测试新增无需逐文件登记；校验结构和引用，不锁死整个策略值 |
| unit-core 把根级异质测试混为一组 | 按 entrypoint、ci-control、distributed 等用途登记 | 功能归属不等同执行环境；保留 accelerator unit 启动方式 |
| 只有 checkpoint 源码映射 | 增加 backend/train、runner、KERV/VLA、transformations、processor 映射 | 根据真实调用、导入、examples 读取建立一对多影响 |
| 平台路由不识别 functional case 消费者 | 从平台支持清单和本地 Hydra defaults 建立反向索引 | 不根据文件名中厂商字符串猜归属；共享 case 取平台并集 |
| examples 曾整体按文档跳过 | 正确定义文档识别，不再替换特殊假路径 | 运行 YAML/Python 必须进入变更判断 |
| 普通标签也触发全量 | 只处理 platform:* 控制标签 | bug 等标签不改变计划 |
| 上传 JUnit，但 gate 只看 job 状态 | unit/functional 写 result.json，两级 gate 重读报告核对 | 缺失、重复、错目标、旧提交、空跑都不能以 job success 放行 |
| requested 绕过 planner | requested profile 接入相同计划、执行和 gate | 自动与补充测试不保留两套成功判定 |
| 临时性能基线可能与编排绿灯混淆 | 按要求保留已有放宽基线，明确单独验收性能 | 结构通过不能等同于性能退化修复 |

## 当前调用链

```text
PR / main push / Nightly / requested
  → 公共控制检查与项目契约检查
  → 平台选择：paths_watch + functional 消费者反查 + 增量平台标签
  → 测试选择：归属 + 明确的额外受影响组 + functional selectors
  → 与平台 include/exclude、当前 case 支持清单相交
  → plan.json（SHA、配置指纹、理由、缺口、预期执行项、hash）
  → 既有 prepare + _platform_pipeline + all_tests_common
  → 既有 unit torchrun / functional CLI 与比较器
  → 每 rank JUnit/collection、每执行项 result.json
  → 平台 gate + 顶层 gate 下载并核对证据
```

计划的 hash 是内容一致性检查，不是安全签名。这里沿用 GitHub 既有权限、fork approval 和只读测试执行边界，不引入特权候选代码执行。

## 配置职责

- `.flagcicd/project.yaml`：profile、逻辑测试组、源码归属、直接影响展开、functional selectors、少量例外、门禁要求。
- `.github/configs/<platform>.yml`：镜像、runner、volume、安装与运行环境、厂商 paths_watch。
- `tests/test_utils/config/platforms/<platform>.yaml`：设备、原有 unit include/exclude/nproc、functional 支持清单。
- 业务 case YAML：模型参数、启动入口、实际训练拓扑。
- 生成的 JSON/XML：本次运行计划、结果、JUnit、collection 和 coverage；不是额外人工配置源。

新建已有组内的 `test_*.py` 自动发现。全新测试目录需要补归属和实际执行入口；新测试若所有平台均排除，规划失败。新增未知源码不冒充已经具备专门测试，保守执行当前配置全量并记录原因。已禁用任务缺口单独记录，不用其他任务成功冒充 E2E 覆盖。

## 已核实的依赖例子

- `backend_megatron.py` 导入 `runner_train.py` 参数处理；选择 runner、训练及训练工具测试，并匹配 Megatron train/hetero/benchmark cases。
- `backend_native_train.py` 也使用训练 runner；选择 runner、训练工具和 native 相关模型测试，当前 native functional 缺口明确记录。
- `runner_train.py`、公共 launcher、runner utils、平台抽象层作为共享路径，保持完整配置回归。
- KERV 的配置、命令、ops/runtime 测试读取 `examples/kerv`；in-process 测试导入 KERV train/inference 入口。
- transformations 测试直接导入 inference diffusion/runtime_context；inference workflow 关闭不意味着可以省略这些单测。
- VLA 与 KERV 分组不代表全部 FlagScale 模型均由该目录验证。
- Qwen3.5 converter 的已知验证是 checkpoint converter；不无条件带上全部模型训练。

`affects` 是“这份源码修改额外需要哪些测试”，只展开直接命中源码的规则，不把“选中了消费者测试”再当成“消费者源码也修改了”。这与依赖递归的含义不同。

## L0–L4 和 Nightly

本轮不提供 `--level` / `--max-level`，不新增层级继承或发布验收门禁。没有实际消费者的 levels/readiness/exclusive 已从运行配置和计划移除。

单个测试的验证目标与整个流程的累积验收不是一回事。现有内容覆盖静态/控制、单进程、分布式、benchmark，但通用设备 smoke、独立安装验收、长稳/收敛、完整发布验收并未实现。

Nightly 是八个平台当前已配置、已接入、已启用测试的全集，保留平台排除项；目前为 26 个 functional case 实例，另加 unit、CLI 和公共控制检查。它不是所有设想能力已经实现。新 schedule 和评论入口仍需合入默认分支后验证事件链。

## 执行和 gate 的具体含义

- unit 保留原 torchrun 进程数，输出每 rank JUnit 与 collection manifest。
- collection 核对实际 pytest 参数、被收集文件、显式 collection skip、nodeid 数量及 world size。
- functional 保留 `flagscale <task> <model> --config ... --test`，原数值比较器输出带实际 platform/device/task/model/case 的 JUnit。
- 每个执行项输出绑定 spec、run/attempt、exit code、报告指纹的 result.json。
- 平台 gate 核对本平台每个 device/case；顶层 gate 核对计划全部平台与执行项。均不只依赖 job success。
- 缺报告、额外/重复报告、错误 rank、错误 case、旧 SHA、其他 run、报告被修改、全部 skip 或比较失败均不通过。
- 同 run 的 failed-jobs rerun 可以使用旧 attempt 的成功证据；同执行项有更晚失败时不能退回旧成功。

JUnit/collection 证明的是当前测试执行与记录一致，不是对整个源码依赖图完整性的数学证明。仍需依赖审查、硬件执行与全量 Nightly。

## 请求与兼容入口

`/ci run all`、`/ci run <platform>` 产生 requested plan，使用精确 PR SHA、受控 planner 代码、同一流水线和证据 gate；补充结果不替代 CI Complete。结束前重新校验 PR head。

`/ci rerun-job <id>` 保留既有来源、权限和最新失败 attempt 校验。本文不新增多平台命令语法。评论控制器合入默认分支前不会因 PR 里的代码自动生效。

保留空 plan 输入给既有平台手动诊断、镜像验证和依赖触发调用；它们明确属于 legacy full diagnostic，不宣称通过本轮新计划证据契约。自动 PR、Nightly、requested 均传 plan。

## 性能基线与验证边界

按本轮最新要求，Kunlunxin benchmark JSON 保留之前已经提交的临时放宽基线，数值、元数据、tolerance 和比较算法均不再调整。临时放宽的基线单独作为性能验收边界说明，不能以一次结构全绿声称性能退化已经修复。

新增测试包含真实 pytest、Shell runner、CLI 替身和数值比较器替身，验证完整 affected/functional 证据链，以及错误路径、报告缺失、重复和版本不匹配等拒绝场景。硬件启动器与业务 CLI 在本地集成测试中使用替身，因此这些测试不等于八平台硬件验收。

本轮不提交、不推送、不更新 #1315。当前代码仍含 #1312 的依赖恢复和授权等前置改动；这些与本轮路由整改应按不同评审范围解释。
