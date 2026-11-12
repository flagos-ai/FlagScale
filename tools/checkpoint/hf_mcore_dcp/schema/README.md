# HF ↔ MCore DCP 映射规则

本文对应当前 `schema_version: 2`。建议按 **模型 `mapping.yaml` → `parameters.yaml` → `args.yaml`** 的顺序阅读。

| 文件 | 职责 |
| --- | --- |
| 模型目录下的 `mapping.yaml` | 选择结构、绑定名称和层范围，声明模型自己的配置及 adapter。 |
| [parameters.yaml](parameters.yaml) | 定义可复用的权重结构，以及拼接、拆分、重排等布局规则。 |
| [args.yaml](args.yaml) | 定义 HF config 与 MCore args 的字段关系、默认值和计算表达式。 |
| [mapping.py](mapping.py) | 组合结构和参数组，处理覆盖及权重名称展开。 |
| [config.py](config.py) | 执行配置字段转换和表达式计算。 |

这里的 `parameters` 指权重张量，`args` 指配置参数。转换器依据这些声明处理已有 checkpoint，不实例化模型。

## 1. `args.yaml`：配置参数组

文件按具名参数组组织：

```yaml
schema_version: 2
groups:
  gqa:
    config_map:
      num_key_value_heads: num_query_groups
      head_dim: kv_channels
    mcore_args:
      group_query_attention: true
```

选择 `gqa` 组后，应用两条双向字段映射，并在 HF→MCore 时设置 `group_query_attention=true`。

| 组内字段 | 格式 | 生效方向 |
| --- | --- | --- |
| `config_map` | `{HF字段: MCore字段或详细定义}` | 双向。 |
| `mcore_args` | `{MCore字段: 常量或表达式}` | 仅 HF→MCore。 |
| `hf_config` | `{HF字段: 常量或表达式}` | 仅 MCore→HF。 |

### 字段映射的三种写法

简单对应：

```yaml
config_map:
  num_hidden_layers: num_layers
```

正向读取 HF 的 `num_hidden_layers`，写入 MCore 的 `num_layers`；反向交换读写方向。

MCore 别名列表：

```yaml
config_map:
  rms_norm_eps: [norm_epsilon, layernorm_epsilon]
```

正向生成配置时写入第一个名称；目标配置中已经存在的其他别名也会同步。反向读取非 `null` 的别名值，多个别名同时存在时要求值一致。

详细定义：

```yaml
config_map:
  num_nextn_predict_layers:
    mcore: mtp_num_layers
    hf_default: 0
    reverse_default: 0
    normalize: int_or_zero
```

| 字段 | 含义 |
| --- | --- |
| `mcore` | 必填，MCore 字段名或别名列表。 |
| `hf_default` | 正向转换时，HF 缺少该字段所使用的值。 |
| `reverse_default` | 反向转换时，MCore 没有非 `null` 值所使用的值。 |
| `required` | 默认 `false`。设为 `true` 时，缺少必需值直接报错，优先于默认值处理。正向检查字段是否存在，反向检查是否有非 `null` 别名值。 |
| `normalize` | 当前支持 `int_or_zero`，双向执行 `int(value or 0)`。 |

`hf_default` 不会替换已经存在的 `null`。没有默认值且非必填时，缺失字段通常不写入目标配置。

字段名中的点表示嵌套配置：

```yaml
config_map:
  vision_config.hidden_size: vision_hidden_size
```

对应 `HF config["vision_config"]["hidden_size"]` 与 `MCore args["vision_hidden_size"]`。

### 配置引用和表达式

```yaml
mcore_args:
  normalization: RMSNorm
  padded_vocab_size: $hf.vocab_size
  moe_shared_expert_intermediate_size:
    op: mul
    args: [$hf.n_shared_experts, $hf.moe_intermediate_size]
hf_config:
  n_shared_experts:
    op: exactdiv
    args: [$mcore.moe_shared_expert_intermediate_size, $mcore.moe_ffn_hidden_size]
```

| 写法 | 含义 |
| --- | --- |
| 普通数字、字符串、布尔值、`null` | 常量。 |
| `$hf.xxx` | 读取 HF 配置，支持点分隔的嵌套路径。 |
| `$mcore.xxx` | 读取 MCore 配置。 |
| `{op: 操作名, args: [操作数...]}` | 配置计算，操作数可以继续使用引用或嵌套表达式。 |

表达式字典必须恰好包含 `op`、`args` 两个键。当前计算操作如下：

| 操作 | 含义 |
| --- | --- |
| `add`、`sub`、`mul` | 加、减、乘，两个操作数。 |
| `div`、`floordiv` | 普通除法、向下取整除法，两个操作数。 |
| `exactdiv` | 整除，除数必须非零，且不能有余数。 |
| `not` | 布尔取反，一个操作数。 |
| `int` | 转为整数，一个操作数。 |

这些操作不执行任意 Python。引用不存在的配置值会报错；反向公式需要单独声明，不会自动推导。

先执行 `config_map`，再按声明顺序执行目标侧的 `mcore_args` 或 `hf_config`。引用目标侧字段时，只能依赖已经生成的字段，不会自动进行依赖排序。

## 2. `parameters.yaml`：共享权重结构

```yaml
schema_version: 2
bindings:
  mlp: ffn
  gate_proj: w1
  up_proj: w3
structures:
  dense_swiglu:
    args: [dense_mlp, swiglu]
    rules:
      - id: fc1
        scope: layer
        hf:
          - "{mlp}.{gate_proj}.weight"
          - "{mlp}.{up_proj}.weight"
        mcore: [mlp.linear_fc1.weight]
        op: swiglu
        extra_state: [mlp.linear_fc1._extra_state/shard_0_1]
```

| 字段 | 含义 |
| --- | --- |
| `bindings` | 结构名称占位符的默认值，值为字符串。 |
| `structures` | 具名权重结构目录。 |
| `structures.<名称>.args` | 选择结构时，自动引入的 `args.yaml` 参数组。 |
| `structures.<名称>.rules` | 构成该结构的权重规则列表。 |

上述规则把 HF 的 gate/up 权重沿第 0 维拼接为 MCore `linear_fc1.weight`；反向则均分为 gate/up。

### 权重规则字段

| 字段 | 格式与含义 |
| --- | --- |
| `id` | 字符串，规则标识，也是覆盖规则时使用的键。组合内必须唯一。 |
| `hf` | HF 权重名称模板列表，即使只有一个名称也必须写成列表。名称展开后就是预期的 HF 序列化输出名称。 |
| `mcore` | MCore DCP **全局键**模板列表，不是各 TP rank 的运行时参数名。 |
| `scope` | 使用模型 `scopes` 中哪组前缀；省略时直接使用规则名称。 |
| `op` | 布局操作，默认 `identity`。 |
| `params` | 布局操作参数字典，支持配置引用和表达式。 |
| `collect` | `{编号轴: 数量}`，把该轴的多个权重收集到一次布局操作。数量可以引用配置。 |
| `optional` | 默认 `false`。该规则没有任何源权重匹配时，是否允许跳过。 |
| `extra_state` | MCore 额外状态对象键列表，常用于 TE；有 scope 时自动加上 MCore 前缀。写入的是空状态占位，而非模型权重。 |
| `shape_config` | `{配置路径: 维度编号}`，从规则中第一个逻辑 HF 张量的形状回填配置；路径以 `hf.` 或 `mcore.` 开头。 |
| `export_slice` | `{axis: 维度, size: 保留长度}`，仅在 MCore→HF 时裁剪，当前仅适用于 `identity`。 |
| `generate` | 常量张量生成声明，当前执行路径用于 MCore→HF 导出。 |

普通规则的 `hf`、`mcore` 都必须是非空列表。`optional: true` 不允许缺少已匹配规则的配套输入，例如只找到 Q 却缺少对应 K/V，仍会报错。它也不负责验证模型应该有多少层。

### `op` 与 `params`

| `op` | HF→MCore 行为 | `params` |
| --- | --- | --- |
| `identity` | 一对一原样对应。 | 无。 |
| `concat` | 沿已有维度拼接。 | `axis` 默认 0；`sizes` 声明各段长度，反向拆分需要它。 |
| `split` | 将一个张量拆分为多个。 | `axis` 默认 0；`sizes` 必填。 |
| `stack` | 新增一个维度，堆叠多个张量。 | `axis` 默认 0。 |
| `swiglu` | 沿第 0 维拼接两个等形状的 gate/up。 | 无。 |
| `expert_swiglu` | 每个专家先拼接 gate/up，再沿第 0 维堆叠专家。 | 无。 |
| `reshape` | 改变形状，元素数保持不变。 | `hf_shape`、`mcore_shape`；形状遵循 PyTorch reshape 规则。 |
| `permute` | 置换维度。 | `dims`，必须是全部维度编号 `0...ndim-1` 的排列。 |
| `qkv` | 将三个 HF 输入按查询组组织成融合 QKV。 | `heads`、`groups` 必填；`order` 默认 `[q, k, v]`；`output_gate` 默认 `false`。 |
| `packed_qkv` | 将一个已拼接的 QKV，从按 Q/K/V 排列改为按 head 分组。 | `heads` 必填，适用于等宽 Q/K/V。 |

这些操作自动按转换方向执行对应逆操作。拼接、拆分、堆叠的 `axis` 可使用负数。

门控 QKV 使用 `output_gate: true` 时，HF 第一个输入为逐 head 排列的 `[q, gate]`，MCore 为逐查询组排列的 `[all q, all gate, k, v]`；此时 `order` 必须为 `[q, k, v]`。

注意两种不同的 `op`：规则的 `op: qkv` 是张量布局操作；`params` 内的 `{op: mul, args: [...]}` 是计算布局参数值的表达式。例如：

```yaml
op: split
params:
  axis: 0
  sizes:
    - {op: mul, args: [$hf.linear_key_head_dim, $hf.linear_num_key_heads]}
    - {op: mul, args: [$hf.linear_key_head_dim, $hf.linear_num_key_heads]}
    - {op: mul, args: [$hf.linear_value_head_dim, $hf.linear_num_value_heads]}
```

### 专家收集、裁剪和生成

```yaml
collect:
  expert: $hf.n_routed_experts
```

`{expert}` 会遍历 `0...n_routed_experts-1`，按专家编号收集。多个 HF 模板按“专家优先、模板其次”展开，例如 `gate0, up0, gate1, up1`。`extra_state` 中还可使用 `{expert_count}` 表示该轴的总数。

从权重形状回填词表大小：

```yaml
shape_config:
  hf.vocab_size: 0
  mcore.vocab_size: 0
  mcore.padded_vocab_size: 0
  mcore.actual_vocab_size: 0
```

`shape_config` 在该规则的形状推导完成后执行，双向都读取逻辑 HF 张量的形状：正向是源 HF 形状，反向是目标 HF 形状。结果覆盖配置中的同名值，并可供后续规则引用；多个规则声明同一配置路径时，形状值必须一致。

共享 embedding 规则据此保留完整词表行数，output 规则也声明 `hf.vocab_size: 0`，以确保输出头和 embedding 一致。随附的 DSv4 映射不裁剪词表或 hash-router 表，不需要 tokenizer 的有效词数，也没有 `--vocab-size` 参数。已有的 `make_vocab_size_divisible_by` 通过配置映射保留，缺失时不推导、不补默认值；训练者自行选择与 `padded_vocab_size` 相容的 TP 和对齐参数。

通用 `export_slice` 仍可用于明确要求裁剪的自定义规则，例如沿第 1 维保留前 16 列：

```yaml
export_slice:
  axis: 1
  size: 16
```

它仅在反向执行，且只适用于 `identity`；当前模型预设不使用该操作。

常量生成：

```yaml
- id: generated_bias
  hf: [model.aux.bias]
  mcore: []
  generate:
    side: hf
    shape: [16]
    dtype: F32
    value: 0
```

`shape` 是形状列表，`dtype` 使用内部编码，如 `BF16`、`F16`、`F32`、`I32`、`I64`，`value` 是填充值。生成规则不消费源权重。当前生成分支不按源权重展开数字占位符，需要给出具体目标名称；动态生成多个层的规则由 adapter 完成。

## 3. 模型 `mapping.yaml`：组合和特化

| 顶层字段 | 含义 |
| --- | --- |
| `schema_version` | 必填，当前为 `2`。 |
| `profile` | 映射名称，用于报告和标识，省略时使用通用名称。 |
| `adapter` | 可选 Python 模块路径，例如 `tools.checkpoint.deepseek_v4.adapter`。 |
| `parameters` | 共享结构选择列表。 |
| `args` | 额外选择的配置参数组列表。结构自带的组会自动引入。 |
| `bindings` | 覆盖共享名称占位符的默认值。 |
| `scopes` | `{scope名称: 前缀对列表}`。也可以由 adapter 动态生成。 |
| `config_map`、`mcore_args`、`hf_config` | 补充或覆盖配置关系，格式同参数组。 |
| `rules` | 模型自己的完整规则，或按 `id` 替换、删除已引入的规则。 |
| `quantization` | 当前使用 `mxfp4` 名称模式列表，识别输入编码。 |
| `ignore` | 按 `hf`、`mcore` 分别声明源权重忽略模式列表。 |
| `zero_checks` | 当前通过 `hf` 模式列表声明必须全零才能忽略的源权重。 |
| `naming` | 可选的 HF 输入名称规范化；不用于推断输出名称。 |

### 选择结构

```yaml
parameters:
  - dense_swiglu
  - use: gdn
    scope: gdn
  - use: gated_gqa
    scope: gqa
```

字符串 `dense_swiglu` 等价于 `{use: dense_swiglu}`。字典选择项支持：

| 字段 | 含义 |
| --- | --- |
| `use` | 必填，`parameters.yaml` 中的结构名称。 |
| `scope` | 覆盖该结构所有规则的 scope。 |
| `optional` | 覆盖该结构所有规则的可选设置。 |
| `bindings` | 仅对本次结构选择生效的名称绑定。 |

绑定优先级为：共享默认值 → 模型 bindings → 选择项 bindings。

### 名称和层范围展开

```yaml
bindings:
  mlp: mlp
  gate_proj: gate_proj
  up_proj: up_proj
scopes:
  layer:
    - hf: model.layers.{layer}.
      mcore: decoder.layers.{layer}.
```

前面的 `fc1` 规则在第 3 层展开为：

```text
HF:
  model.layers.3.mlp.gate_proj.weight
  model.layers.3.mlp.up_proj.weight
MCore:
  decoder.layers.3.mlp.linear_fc1.weight
```

| 写法 | 处理方式 |
| --- | --- |
| `{mlp}` | 在组合结构时，按字符串 bindings 替换。 |
| `{layer}` | 在匹配源权重时提取数字编号；HF 和 MCore 共用该编号。 |
| `{expert}` 配合 `collect` | 按声明数量枚举编号并收集。 |
| `$hf.hidden_size` | 在配置/布局参数求值时读取配置值。 |

scope 前缀直接与规则名称拼接，因此通常以 `.` 结尾。每个前缀对包含 `hf`、`mcore`，还可设置 `optional: true`；scope 或规则任一允许可选，该展开规则就允许可选。

同一 scope 可列出多个位置，也可使用字面量层编号选择非连续层。HF 与 MCore 层编号不一致时，应提供对应的具体前缀，不会自动计算编号偏移。

```yaml
scopes:
  gdn:
    - hf: model.layers.0.
      mcore: decoder.layers.0.
    - hf: model.layers.2.
      mcore: decoder.layers.2.
  vision: []
```

空 scope 不展开权重规则，但不会撤销结构已经引入的配置参数组。选择不存在的 scope 会报错。

当前结构中的 rule ID 必须唯一，不能直接重复选择同一结构并期待自动生成独立 ID。相同规则用于多个位置时，优先使用包含多个前缀对的 scope。

### 输入名称、忽略和量化声明

```yaml
naming:
  strip_prefix: wrapper.
  segments:
    backbone: model
  replace:
    .attention.: .self_attn.
ignore:
  hf: [model.rotary_emb.inv_freq]
zero_checks:
  hf: [model.aux.bias]
quantization:
  mxfp4: [layers.*.ffn.experts.*.w*.weight]
```

`naming` 依次执行去前缀、点分隔名称段替换、字符串替换。它只把输入别名归一到规则声明的 HF 名称，导出始终使用规则声明的名称。没有 `naming.use` 或历史命名快照恢复机制。

`ignore`、`zero_checks` 和 `quantization.mxfp4` 使用 glob 模式，不是正则表达式；HF 模式针对规范化后的名称。普通 `hf`/`mcore` 权重规则使用数字占位符，不支持用 `*` 代替层编号。

`ignore` 不检查权重数值。`zero_checks.hf` 会在实际转换时验证已匹配的权重全零；dry-run 只记录待检查项。`quantization.mxfp4` 用于识别哪些输入字节张量是 MXFP4，不会把 BF16 权重新量化成 MXFP4。

## 4. 合并与覆盖顺序

1. 按 `parameters` 顺序选择结构，并收集其 `args` 组。
2. 加上模型显式声明的 `args` 组，按首次出现顺序去重、应用。
3. 应用模型本地配置字段和规则。
4. 按命令行顺序应用每个 `--mapping-override` 文件。

配置字典按直接字段键合并，后者覆盖前者，不递归合并字段内部的字典或列表。例如覆盖某个 `config_map` 详细定义时，要重新写出需要保留的默认值等选项；覆盖 `scopes.layer` 时会替换整个前缀列表。

规则按 `id` **整条替换**，新 ID 则新增。删除规则：

```yaml
schema_version: 2
rules:
  - id: output
    disabled: true
```

`disabled` 是组合阶段的删除标记，不是执行阶段的权重规则字段。

bindings 在结构组合时已经替换。后续 override 仅修改 bindings 不会重新绑定先前规则；需要重新选择结构，或覆盖相应完整规则。

## 5. Adapter 与 YAML 的边界

配置外壳、特殊 norm 数值语义、模型层分布等由模型 adapter 处理；通用结构和布局操作仍由 YAML 声明。

| 可选 hook | 作用 |
| --- | --- |
| `normalize_config(config, forward, keys)` | 在通用字段转换前规范化源配置。读取 HF reference 配置时也调用，`keys` 来自可选索引，可能为空。 |
| `finish_config(config, forward)` | 通用字段转换后补充目标配置。 |
| `prepare_mapping(mapping, configs, specs, forward)` | 编译布局前生成实际 scopes 或处理模型特有的规则选择。 |
| `tensor_transforms(configs, hf_names, forward)` | 返回逻辑 HF 名称到具名张量函数的字典；处理不改变形状或 dtype 的数值变换。 |
| `export_config(config)` | 在输出精度确定后，还原最终序列化的 HF 配置外壳。 |

`forward=True` 表示 HF→MCore。`configs` 包含 `hf` 和 `mcore` 两个字典，`specs` 描述源权重元数据。

第一版 MCore→HF 必须显式指定 `--hf-reference`，包括 dry-run 和保留快照的 checkpoint。reference 需要原 HF 配置，导出 tokenizer 时还需其根目录资产，不需要权重。无 reference 的自动配置和资产恢复留待后续版本支持。

例如共享 GDN 参数组引用的 `$hf._linear_pattern` 必须由使用该组的模型 adapter 提供，它不是框架内置变量。adapter 也可以根据实际结构生成 scopes，因此仅展开 YAML 时看不到动态层分布。DSv4 adapter 则处理压缩层配置及 router bias 约定。

张量数值变换在正向布局之前、反向布局之后执行。存在数值变换时只能使用 dense 模式；BF16/FP16 的 norm 加减 1 不保证任意权重逐位往返。

## 6. 完整示例与查看命令

以下是使用默认 RoPE 的通用 dense decoder 组合，展示名称和结构的声明方式，不依赖任何提交外的模型 adapter。接入具体 HF 模型时，还需声明该模型的 `hf_config.model_type` 和 `architectures`：

```yaml
schema_version: 2
profile: dense_decoder
parameters:
  - embedding
  - output
  - final_norm
  - gqa
  - qk_norm
  - dense_swiglu
args: [rope]
bindings:
  embedding: model.embed_tokens
  output: lm_head
  final_norm: model.norm
  attention: self_attn
  attention_output: o_proj
  attention_norm: input_layernorm
  mlp_norm: post_attention_layernorm
  mlp: mlp
  gate_proj: gate_proj
  up_proj: up_proj
  down_proj: down_proj
scopes:
  layer:
    - hf: model.layers.{layer}.
      mcore: decoder.layers.{layer}.
```

以下命令使用随附的 DSv4 profile。从仓库根目录查看组合后的规则和配置关系，不写文件：

```bash
python -m tools.checkpoint.hf_mcore_dcp.convert export-mapping \
  --mapping tools/checkpoint/deepseek_v4/mapping.yaml \
  --output /dev/null --dry-run
```

查看 adapter 处理和源权重匹配后的实际转换计划，不写权重：

```bash
python -m tools.checkpoint.hf_mcore_dcp.convert hf-to-mcore \
  --mapping tools/checkpoint/deepseek_v4/mapping.yaml \
  --source /path/to/hf-checkpoint \
  --output /path/to/planned-output --dry-run
```

随附模型示例：[DeepSeek V4](../../deepseek_v4/mapping.yaml)。其他模型的 profile 和 adapter 作为独立集成提供。转换命令、精度模式和存储边界见[主 README](../README.md)。
