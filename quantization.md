# π₀.₅ 量化方案：从 QDQ Baseline 到部署实践

> 适用代码：`nvidia-trt` 分支（`deployment_scripts/pytorch_to_onnx.py`、`build_engine.sh`）。
> 硬件目标：Jetson AGX Thor（sm_110，Blackwell，FP8/NVFP4 tensor core）。

---

## 1. Baseline：本项目使用的 QDQ 流程

### 1.1 定位与术语

本项目采用 **PTQ（Post-Training Quantization，训练后量化）+ QDQ（Quantize-DeQuantize）**
方案，工具链为 **NVIDIA ModelOpt**（`mtq`）→ ONNX → **TensorRT**。

- **PTQ**：只用少量校准数据确定量化 scale，**不更新任何权重**（区别于 QAT 的伪量化微调）
- **QDQ**：把"量化→反量化"表示为显式节点对。QDQ 不是性能优化，而是**量化误差的
  显式标记点**——PyTorch 阶段用它在浮点计算中模拟量化误差，TensorRT 阶段靠它定位
  低精度 kernel 的插入位置

端到端链路：

```
PyTorch checkpoint (BF16)
   │  mtq.quantize：插入量化器 + 校准定 scale（伪量化，权重不变）
   ▼
量化 PyTorch 模型（QDQ 语义存在于 forward 中）
   │  torch.onnx.export：量化器 trace 成 QDQ 节点
   ▼
ONNX（权重为量化常量 + scale，激活带 Q/DQ 标记）
   │  trtexec --stronglyTyped：QDQ 模式匹配，融进 GEMM kernel
   ▼
TensorRT engine（QDQ 消失，误差保留，开销消除）
```

### 1.2 量化器插入

`pytorch_to_onnx.py:813` 的 `mtq.quantize(model, quant_cfg, forward_loop)` 把每个
`nn.Linear` 替换为 `QuantLinear`，附挂三个 `TensorQuantizer`：

| 量化器 | 作用 | 本项目状态 |
|---|---|---|
| `input_quantizer` | 激活量化 | FP8 层静态 / NVFP4 层动态 |
| `weight_quantizer` | 权重量化 | 全部静态（scale 固化） |
| `output_quantizer` | 输出量化 | **全部关闭**，GEMM 输出以 FP16 回到残差流 |

前向变为 `dequant(quant(x)) @ dequant(quant(W))`——计算仍是 BF16/FP16，但数值先经过
量化往返，使模型在校准阶段就"感受"到量化误差。

非 `nn.Linear` 的 matmul（attention 的 QK^T、AV）无处挂量化器，由
`replace_attention_with_quantized_version()` 插入 `QuantizedMatMul` 模块解决
（`pytorch_to_onnx.py:263`）。

### 1.3 精度分配（哪些层量化成什么）

由 `quant_cfg`（`pytorch_to_onnx.py:763-787`）决定，基准 `FP8_DEFAULT_CFG`（E4M3，
per-tensor 静态 scale）之上做覆盖：

| 模块 | 精度 | 说明 |
|---|---|---|
| LLM（PaliGemma Gemma 18 层）的 q/k/v/o、gate/up/down | **NVFP4** | E2M1 数据 + 每 16 元素一个 FP8(E4M3) block scale；权重静态、激活动态（`TRT_FP4DynamicQuantize`） |
| action expert（gemma_expert）、ViT MLP、action/state 投影 | **FP8 E4M3** | 权重+激活均静态 scale |
| attention QK^T / AV（prefix 块） | **FP8** | `QuantizedMatMul`；suffix 小块保持 FP16（Q/DQ 开销超过 matmul 本身，`pytorch_to_onnx.py:270`） |
| `nn.Conv2d`（ViT patch embedding） | **FP16** | 量化显式关闭 |
| `time_mlp` / `norm.dense`（AdaRMS 时间条件） | **FP16** | 输出只依赖固定 denoise 调度，被常量折叠，插 QDQ 纯属浪费 |
| softmax | **FP32** | 显式 `softmax(dtype=torch.float32)` |

层间传递的 hidden states（残差流）保持 **FP16**——所有 output 量化器关闭，GEMM 输出
以 FP16 写回；每个 Linear 在 GEMM 入口处由 input 量化器把 FP16 激活压到 FP8/NVFP4。
FP4 只存在于 LLM 层内部的 GEMM 里。

### 1.4 校准：scale 如何确定

`mtq.quantize` 的 `forward_loop` 用真实数据驱动校准（默认 32 个 LIBERO 样本，
等间距取样，`calibration_data.py:49`；`--num_calibration_samples` 可调）：

1. 每个量化器在校准前向中统计输入的 **amax（绝对值最大值）**
2. 静态量化器定 scale 并固化：

```
FP8 E4M3 可表示最大值 = 448
scale = amax / 448
quant(x)  = clamp(round(x / scale), -448, 448)
dequant(q) = q * scale
```

3. NVFP4 层的激活量化器标记为 dynamic（`pytorch_to_onnx.py:826`），不在校准时固化——
   运行时每 16 元素现场求 block scale，用一点运行时开销换精度

校准是**唯一引入数据依赖的环节**：样本选择变化 ⇒ scale 变化 ⇒ 引擎行为变化。
等间距取样在数据集不变时是确定性的，同 checkpoint 重导出可复现。

### 1.5 导出：量化器变成 ONNX QDQ 节点

`torch.onnx.export`（legacy tracer，`dynamo=False`）把量化器 trace 为显式节点：

| 对象 | ONNX 形态 | 备注 |
|---|---|---|
| FP8 权重/静态激活 | `QuantizeLinear` → `DequantizeLinear`（`TRT_FP8QDQ`） | scale 为常量 |
| NVFP4 激活 | `TRT_FP4DynamicQuantize` | 运行时产生 scale |
| NVFP4 权重 | `TRT_FP4QDQ` → **2DQ 格式** | `NVFP4QuantExporter` 转换（`pytorch_to_onnx.py:476-489`）：TRT 10.16 的 ONNX 解析器要求用两个 `DequantizeLinear` 表达"FP4 数据 + FP8 块 scale" |

导出后权重以量化常量 + scale 存进外部数据文件（`model_fp8_nvfp4.data`），
模型中不再有浮点权重视图。

### 1.6 TensorRT 消费 QDQ

`trtexec --stronglyTyped`（`build_engine.sh`）构建时做 **QDQ 模式匹配**：

- 识别 `DQ → GEMM → (Q)` 模式，将 DQ 融入 GEMM kernel：FP8 GEMM 走 FP8 tensor core；
  NVFP4 GEMM 在 kernel 内做 FP4→高精度的块反量化
- QDQ 放置有匹配规则（GEMM 两个输入都带 QDQ 才进低精度路径），这正是
  `output_quantizer` 关闭、suffix attention 留 FP16 等决策的约束来源
- 融合后 QDQ 节点消失，残余量化开销只有 NVFP4 激活动态量化算子（逐层 profile 中的
  `*Dyna*` 类）和必要的 cast

### 1.7 精度验证基线

| 指标 | 期望值 | 测量方式 |
|---|---|---|
| compare 模式 cosine similarity | ≈ 0.99 | `pi05_inference.py --inference-mode compare`（同 noise 对照 PyTorch BF16） |
| LIBERO-Long 成功率掉点 | ΔSR 在 ±5% 内判无显著差异 | 三臂评测，见 `handbook.md` Step 6 |
| 推理延迟 | ~49 ms（vs PyTorch BF16 ~130 ms，2.7×） | Thor MAXN，`--num-test-runs 10` |

### 1.8 复现命令

```bash
# ONNX 导出（FP8 + NVFP4，32 样本校准）
python deployment_scripts/pytorch_to_onnx.py \
  --checkpoint_dir $CKPT --output_path $CKPT \
  --config_name pi05_libero \
  --precision fp8 --enable_llm_nvfp4 --quantize_attention_matmul

# 引擎构建
ACTION_HORIZON=10 bash deployment_scripts/build_engine.sh \
  $CKPT/onnx/model_fp8_nvfp4.onnx \
  $CKPT/engine/model_fp8_nvfp4.engine
```

`$CKPT=~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch`

### 1.9 已知边界

- **不支持纯 FP16**：π₀.₅ 原生 BF16（8 位指数），FP16 动态范围不足会在 Gemma
  attention 溢出并随去噪循环累积
- **重导出会覆盖旧 ONNX**（固定文件名，静默覆盖）但不影响已构建引擎；
  重导出后应重建引擎保持产物一致
- 校准样本过少或前向失败过多会导致 scale 失真——关注校准日志中的
  `Calibration batch N forward failed` 警告


---

## 2. PTQ 专题：AWQ / GPTQ / SmoothQuant

第一节的 baseline 是"校准定 scale → QDQ 导出"的直接路径：量化误差被动接受，不
做任何主动补偿。学术界针对 LLM 量化误差提出的三个经典 PTQ 算法，本质上都是在
**量化前对权重或激活做等价变换/补偿，把误差转移到损失更不敏感的地方**。三者都不
更新权重语义（变换数学上恒等或近似恒等），因此都属于 PTQ 范畴。

本节按四个维度对照：**动机**（要解决什么误差）、**缩放对象**（对谁做变换）、
**缩放方向**（放大还是缩小）、**优化的瓶颈**（误差/效率的瓶颈在哪）。

### 2.1 共同前提：量化误差从哪里来

均匀量化 `quant(w) = round(w / Δ)` 的误差上界是 Δ/2，Δ（scale）由被量化张量的
**动态范围**决定。误差失控只有两种原因：

1. **离群值撑大量程**：少数大值把 Δ 顶大，多数正常值的相对误差随之恶化
   （SmoothQuant 针对激活的这类问题）
2. **位宽太低**：4 bit 只有 16 个量化级，即使范围正常，舍入误差本身就不可忽视
   （AWQ / GPTQ 针对权重的这类问题）

三个算法的分歧点在于：**谁出了问题、把难度转移给谁**。

### 2.2 AWQ（Activation-aware Weight Quantization）

**动机**：W4A16 场景（权重 4 bit、激活保持 FP16）下，权重舍入误差是精度的唯一来源。
关键观察：权重通道的重要性**不平等**——约 1% 的"显著通道"对输出贡献远超其余，
而显著性应该用**激活幅值**判断（`|x|` 大的通道，其权重误差被激活放大），而不是
权重自身幅值。只要保护这 1% 的通道，就能挽回大部分精度损失。

**缩放对象**：逐输出通道的权重 `W`（配合激活 `x` 做恒等变换）。利用数等价关系：

```
y = W·x = (W·diag(s)) · (diag(s)⁻¹·x)
```

**缩放方向**：**放大显著权重通道**（`s_j > 1`），激活对应通道对称缩小。原理：分组
量化（group=128）的 Δ 由组内最大值决定；显著通道放大后占住组内更大的动态范围份额，
其有效分辨率提高、相对误差下降——非显著通道误差略增，但它们对输出影响小。`s` 不
是拍出来的，而是在校准数据上网格搜索：

```
s_j = (mean|x_j|)^α，α ∈ [0, 1] 网格搜索，最小化 ||Q(W·s)(x/s) − Wx||²
```

**优化的瓶颈**：权重量化误差（W4 位宽极限）。它不触碰激活量化（激活全程 FP16），
因此面向的是**权重主导误差**的场景；附带收益是权重 4 bit 化后显存/带宽压力减半以上，
对 batch=1 的内存受限推理（如本项目 expert 段的 GEMV）同时是性能优化。

### 2.3 GPTQ

**动机**：W3/W4 极端位宽下，逐通道 round-to-nearest 的误差仍然太大；理论上最优的
逐层补偿法 OBQ（Optimal Brain Quantization）复杂度 O(d⁴) 量级，无法用于 LLM。
GPTQ 是 OBQ 的高效实现：把逐层权重量化表述为**输出重构误差最小化**——

```
min ||W·X − Q(W)·X||²，校准数据 X 的二阶信息 H = 2X·Xᵀ（Hessian）
```

**缩放对象**：**不缩放**。GPTQ 的机制是误差补偿而非等价变换：按固定顺序逐列量化，
每量化一列 `q`，立刻用 Hessian 逆把该列的量化误差**摊销到右侧所有未量化列**：

```
量化第 q 列得误差 err = (w_q − Q(w_q))
未量化列修正：w_j ← w_j − err / [H⁻¹]_qq · [H⁻¹]_jq，j > q
```

**缩放方向**（补偿方向）：误差沿 `H⁻¹` 第 q 行给出的方向分摊给未量化列——直观含义是
"这一列少掉的信息，由后面列在保持层输出不变的前提下补回来"。工程上的三个关键技巧：
任意顺序改固定顺序（所有行共享一次 Cholesky 分解）、Cholesky 求逆替代反复高斯消元、
lazy batch update 降低显存带宽压力——把 175B 模型的量化压到小时级。

**优化的瓶颈**：双重瓶颈——**精度侧**是 W3/W4 的权重舍入误差（用二阶信息补偿到接近
理论极限）；**效率侧**是 OBQ 类算法的计算可扩展性（GPTQ 的核心贡献其实在后者）。

### 2.4 SmoothQuant

**动机**：W8A8（权重激活都 INT8）时，误差几乎全部来自**激活**：LLM 的激活存在
**系统性 outlier 通道**——固定少数通道的幅值可达其他通道的 ~100 倍，per-tensor
激活量化的 Δ 被 outlier 顶大，正常通道的有效分辨率被压垮；而权重分布平坦，量化
余量很大。思路：**把量化难度从激活迁移给权重**。

**缩放对象**：逐通道同时对激活和权重做恒等变换（与 AWQ 同一数学形式，目的相反）：

```
Y = X·W = (X·diag(s)⁻¹) · (diag(s)·W)
```

**缩放方向**：**压缩激活 outlier 通道**（除以 `s_j > 1`），对应权重通道对称放大。
平滑强度由 α 控制：

```
s_j = max|X_j|^α / max|W_j|^(1−α)
α = 0.5 为典型值：激活、权重各承担一半难度
α → 1：难度全推给权重（权重开始出问题）；α → 0：回到不平滑
```

关键工程性质：`diag(s)⁻¹` 可以**吸收进前一层**的 LayerNorm 参数或 Linear 权重
（离线完成），运行时零额外算子。

**优化的瓶颈**：激活 outlier 导致的 per-tensor 激活量化失效——压平 outlier 后，
W8A8 per-tensor 量化从"不可用"变为"接近无损"，使 LLM 推理能整体落在 INT8 tensor
core 上。

### 2.5 三算法对照

| 维度 | AWQ | GPTQ | SmoothQuant |
|---|---|---|---|
| 目标位宽 | W4A16 | W3/W4（A16） | W8A8 |
| 动机 | 1% 显著权重通道决定精度 | 低 bit 权重舍入误差 + OBQ 太慢 | 激活 outlier 拖垮 per-tensor 量化 |
| 变换/补偿对象 | 显著权重通道 | 未量化权重列（吸收误差） | 激活 outlier 通道 |
| 方向 | 放大权重（激活对称缩小） | 误差向右摊销（无缩放） | 压缩激活（权重对称放大） |
| 优化瓶颈 | 权重量化误差 | 权重误差 + 算法复杂度 | 激活量化误差 |
| 校准数据用途 | 网格搜索 s | 估计 Hessian | 统计通道 max |
| 运行时开销 | 零（离线并入权重） | 零 | 零（并入前层参数） |

数学形式上 AWQ 与 SmoothQuant 是**同一恒等变换的两种用法**：AWQ 放大权重保护显著
通道（误差在权重侧），SmoothQuant 压缩激活消灭 outlier（误差在激活侧）——方向相反，
因为它们诊断出的"病灶"相反。GPTQ 则完全不做缩放，走的是二阶误差补偿路线。

### 2.6 与本项目 baseline 的关系

本项目的 ModelOpt baseline（第一节）未使用这三个算法，原因是场景错位：

- **FP8 E4M3 的动态范围远大于 INT8**：8 位浮点（4 位指数）天然容忍激活 outlier，
  SmoothQuant 要解决的问题在 FP8 下大幅弱化——这是 baseline 用 per-tensor 静态 scale
  也能到 cosine≈0.99 的原因
- **NVFP4 的 block-16 缩放是另一条路径**：每 16 元素一个 FP8 块 scale，用硬件支持的
  细粒度动态缩放替代 AWQ 式的软件通道保护——思想相通（让重要/大值区域获得更高有效
  分辨率），但实现在硬件块缩放语义里，零软件变换成本
- **weight-only 算法收益面小**：AWQ/GPTQ 面向 W4A16，本项目是 W4A4（NVFP4）与 W8A8
  （FP8），激活同样量化，仅优化权重一侧不够；且 GPTQ 的二阶补偿对 E2M1 这种非均匀
  浮点网格不直接适用

如果 Step 6 的 LIBERO 掉点评测显示 NVFP4 损失不可接受，可选的升级路径按代价排序：

1. **减小 NVFP4 覆盖范围**（如 expert 层回退 FP8）——改 `quant_cfg` 即可，零新算法
2. **SmoothQuant 式激活平滑**（INT8 化时几乎必需；FP8 下收益有限）
3. **QAT**——伪量化微调数百步，让权重主动适应量化网格（误差最后手段）

### 参考

- GPTQ: Frantar et al., *GPTQ: Accurate Post-Training Quantization for Generative
  Pre-trained Transformers*, [arXiv:2210.17323](https://arxiv.org/abs/2210.17323)
- AWQ: Lin et al., *AWQ: Activation-aware Weight Quantization for LLM Compression
  and Acceleration*, [arXiv:2306.00978](https://arxiv.org/abs/2306.00978)
- SmoothQuant: Xiao et al., *SmoothQuant: Accurate and Efficient Post-Training
  Quantization for Large Language Models*, [arXiv:2211.10438](https://arxiv.org/abs/2211.10438)
