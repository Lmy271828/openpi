# π₀.5 在 Jetson AGX Thor 上的推理性能分析报告

> 对象：openpi π₀.5（pi05\_libero，action horizon 10，action dim 32）
> 平台：Jetson AGX Thor DevKit（JetPack 7.2，MAXN，`jetson\_clocks` 锁定），Docker `openpi-pi0.5:l4t-jp7.2`
> 流程参照：\[OpenPi π₀.₅ on Jetson Thor | Jetson AI Lab](https://www.jetson-ai-lab.com/tutorials/openpi\_on\_thor/) Step 1–12
> 数据：`perf\_data/`（num\_steps 扫描、Eager/torch.compile/TensorRT 三路 nsys、trtexec 逐层 profile），
> 由 `deployment\_scripts/analyze\_perf.py` 汇总为 `perf\_data/analysis.md`

\---

## 0\. 摘要

|后端|时延 (ms)|关键机制|瓶颈画像|
|-|-|-|-|
|PyTorch Eager BF16|\~316|逐算子 launch（\~13.6k 次/帧）|**launch-bound**：GPU 空闲 \~45%|
|PyTorch `torch.compile(max-autotune)`|126.7（model）|Inductor 融合 + **39 张 CUDA graph**/帧|带宽受限的 BF16 kernel 执行|
|TensorRT FP8+NVFP4|47.9（model）|全图 **1 张 CUDA graph** + 量化|expert 循环（52%）与 LLM prefill（37%）|

三个可落地的优化抓手（详见 §5）：

1. **减少 denoise 步数/蒸馏**：实测每步成本 PyTorch 6.24 ms、TRT \~2.5–3.2 ms，10→4 步可省 \~30%。
2. **expert 权重 NVFP4 化**：expert 占 TRT 时延 52%，其权重每帧被读 10 遍（3.1 GB，占总流量 2/3），
量化到 NVFP4 预计再省 4–8 ms。
3. **NVFP4 动态量化算子优化**：实测 LLM 的 `\*Dyna\*` 动态量化链占 TRT profile 的 9.9%（\~6.1 ms），
值得与"LLM 回退 FP8 静态 scale"做 A/B。

\---

## 1\. 基线结果

|后端|Total (ms)|Model (ms)|host 开销 (ms)|相对加速|
|-|-|-|-|-|
|PyTorch BF16 + `torch.compile(mode="max-autotune")`|130.37 ± 0.82|126.72 ± 0.66|\~3.65 (2.8%)|1.0×|
|TensorRT FP8 + NVFP4（`--useCudaGraph`）|48.71 ± 0.11|47.93 ± 0.05|\~0.78 (1.6%)|**2.68× / 2.64×**|

（数据：`media/pytorch\_baseline.png`、`media/tensorrt\_fp8\_nvfp4\_baseline.png`，与教程公布值一致。
精度：compare 模式 cosine similarity ≈ 0.9945。）

两点直接结论：

1. **TRT 的 host 开销已压到 1 ms 以内**（tokenize 缓存 + fast\_infer 绕过 Observation 校验 + CUDA graph replay），
继续优化的空间几乎全在 GPU 侧。
2. **2.64× 的加速主要来自精度下降带来的权重流量缩减与 kernel 融合**，而不是计算量变化——
两条路径的数学计算基本相同（常数折叠除外）。

\---

## 2\. 推理流水线与理论负载（Roofline 分析）

### 2.1 计算图结构

每次 `policy.infer()` 的 GPU 工作分三段（`src/openpi/models\_pytorch/pi0\_pytorch.py:377` `sample\_actions`）：

|阶段|模块|规模|token 数|执行次数/推理|
|-|-|-|-|-|
|① 视觉编码|SigLIP-so400m（27 层，width 1152，patch 14）|\~0.42B 参数|3 视角 × 256 = 768|1|
|② LLM prefix prefill|Gemma 2B（18 层，width 2048，MLP 16384，GQA 8/1，head\_dim 256）|\~1.98B 参数（非嵌入）|768 图像 + ≤208 语言 ≈ **976**|1|
|③ action expert 去噪|Gemma 300M（18 层，width 1024，MLP 4096，adaRMS）|\~0.31B 参数|后缀仅 **11**（1 state + 10 action）|**10（串行）**|

关键点：③ 通过 prefix KV cache 只跑 11 个 token，但要**串行跑 10 次**（flow matching 的 10 步 Euler 积分），
每次都要重读一遍 expert 权重。

### 2.2 算力与显存流量账本（batch = 1）

|阶段|计算量（2·P·T 估算）|BF16 权重流量|FP8/NVFP4 权重流量|
|-|-|-|-|
|① ViT ×3|\~0.65 TFLOPs|0.84 GB|0.42 GB (FP8)|
|② LLM prefill|\~4.0 TFLOPs|3.96 GB|\~1.0 GB (NVFP4)|
|③ expert ×10|\~0.085 TFLOPs|0.62 GB ×10 = 6.2 GB|0.31 GB ×10 = 3.1 GB (FP8)|
|**合计**|**\~4.7 TFLOPs**|**\~11.0 GB**|**\~4.5 GB**|

（prefix KV 因 GQA 只有 1 个 KV head，每层仅 2×976×256×2B ≈ 1 MB，全帧 \~0.18 GB，可忽略。）

Thor 标称（T5000，130W MAXN）：显存带宽 273 GB/s；Tensor Core 峰值约 BF16 260 / FP8 520 / FP4 1035 TFLOPS（dense）。

|后端|权重流量下界 @273 GB/s|实测 model 时延|有效算力|峰值利用率|
|-|-|-|-|-|
|PyTorch BF16|\~41 ms|126.7 ms|\~37 TFLOPS|**\~14%（BF16）**|
|TRT FP8+NVFP4|\~16.5 ms|47.9 ms|\~98 TFLOPS（等效）|**\~19%（FP8）**|

**核心判断：两条路径都远未触及算力峰值，本质都是"显存带宽 + 小算子/串行开销"受限。**
batch=1、expert 每步只有 11 个 token，绝大多数 GEMM 是 GEMV 级别，Tensor Core 喂不饱。
这决定了优化主线：**减流量、减步数、减串行段，而不是堆算力**。

\---

## 3\. PyTorch 瓶颈剖析（Eager / torch.compile 对照）

### 3.1 num\_steps 扫描：固定开销 vs 去噪循环（实测）

用 `T(N) ≈ T\_fixed + N·T\_step` 对 torch.compile 路径做线性分解
（`deployment\_scripts/pi05\_numsteps\_sweep.py`，实例级覆盖 `policy.\_sample\_kwargs`，不改源码）：

|num\_steps|1|2|4|6|8|10|
|-|-|-|-|-|-|-|
|mean (ms)|74.82|80.42|92.99|106.04|118.79|130.27|

* **T\_fixed（ViT + LLM prefill + host）= 68.31 ms**，占 T(10) 的 **52.3%**
* **T\_step（单次 expert denoise）= 6.24 ms**，10 步合计 62.4 ms，占 **47.7%**

R² 极高的线性关系证实：去噪循环是严格的"每步固定成本 × 10"结构，减步数收益完全可预测。

### 3.2 Eager vs torch.compile：launch 开销才是 compile 的主要收益（实测）

Eager 路径（`pi05\_pt.nsys-rep`，NVTX 稳态中位数）与 compile 路径对比：

|阶段|Eager (ms)|compile (ms)|加速|
|-|-|-|-|
|S1 embed\_prefix（ViT ×3 ≈ 30.5）|31.2|—|—|
|S2 LLM prefill|16.4|—|—|
|S3 expert ×10（每步 19.6）|\~196|62.4（每步 6.24）|**3.1×**|
|全帧（S0\_policy\_infer）|**316.4**|**135.0**|**2.34×**|

Eager 的 kernel 画像（13 次推理聚合）：每帧 \~13.6k 次 kernel launch；GPU busy 仅 \~175 ms/帧，
对照 316 ms 的墙钟时间，**GPU 空闲约 45%**——典型 launch-bound。kernel 分类：

|类别|占比|launch 次数（13 帧）|
|-|-|-|
|GEMM (cuBLAS/nvjet)|50.1%|31,018|
|Elementwise/reduction|21.7%|86,437|
|bf16↔fp32 cast/convert|12.7%|46,722|
|Softmax|3.1%|2,574|
|Norm|1.0%|2,158|

* 单 kernel 榜首是 `gemv2T\_kernel\_val`（5070 次，257 ms）——expert 的 GEMV 级矩阵乘；
ViT/LLM 的大 GEMM 走 `nvjet\_sm110`。
* **cast/convert 类占 12.7%**（4.67 万次）：bf16 权重与 fp32 激活/softmax 之间的反复转换，
是 Inductor 融合能消掉的主要对象之一。

### 3.3 torch.compile 稳态 = CUDA graph replay（sqlite 直读实测）

对 `pi05\_ptc.sqlite`（compile 路径）的 test\_i 窗口（135 ms）直接统计 CUDA API：

* `cudaGraphLaunch` × **39**，普通 `cudaLaunchKernel` 仅 × 83；
* 窗口内 CUPTI 可见 kernel 只有 83 个 / 0.24 ms——**稳态计算几乎全部跑在 39 张 Inductor CUDA graph 的 replay 里**
（graph 内 kernel 不被此 nsys 版本归因到 `CUPTI\_ACTIVITY\_KIND\_KERNEL`，故 per-kernel 分析只能用 Eager 对照组）。

这修正了一个常见直觉：**compile 路径的 launch 开销基本已被 cudagraphs 消除**，它的 126.7 ms 几乎全是
BF16 kernel 的真实执行时间。距离 41 ms 的权重流量下界仍有 \~3×，差距来自：GEMV 低利用率、
融合之外的 elementwise/cast、以及串行段间无法重叠。

### 3.4 代码级已确认的问题

1. **attention mask dtype 隐患**：`\_prepare\_attention\_masks\_4d` 产出 fp32 mask 而 Q/K/V 为 bf16，
使 SDPA 无法走 memory-efficient kernel（部署侧 `install\_attention\_mask\_dtype\_fix` 已修补）。
2. **host 侧 3.65 ms**：Observation 校验、tokenizer 每帧重跑。TRT 路径的 `OPENPI\_FAST\_INFER` /
`OPENPI\_TOKENIZE\_CACHE` 两个 hook 与后端无关，PyTorch 路径可直接复用（已验证可压到 <1 ms）。
3. **ViT 三视角串行循环**：PyTorch 路径 `embed\_prefix` 对 3 个相机逐个跑 SigLIP；
TRT 导出路径已改为 view-batching（`vit\_view\_batch`），PyTorch 路径未享受。

\---

## 4\. TensorRT（FP8 + NVFP4）瓶颈剖析（逐层 profile 实测）

### 4.1 阶段归因

来源 `model\_fp8\_nvfp4.engine\_profile.json`（3847 层，执行序边界法归因）。
注意 trtexec 逐层 profile 含层间同步开销，合计 61.7 ms 约为实际 47.9 ms 的 1.29 倍，**看占比不看绝对值**：

|阶段|profile 耗时 (ms)|占比|折算实际 (ms)|
|-|-|-|-|
|Action expert ×10 denoise（FP8）|32.2|**52.1%**|\~25.0|
|LLM prefill（Gemma 2B，NVFP4）|23.0|**37.2%**|\~17.8|
|ViT（SigLIP ×3，FP8）|6.6|**10.7%**|\~5.1|

推算：expert 每步 ≈ 3.22 ms（profile 口径；实际 \~2.5 ms）；LLM 每层 ≈ 1.27 ms。

与 PyTorch 对照（同口径折算）：固定段 68.3 → \~22.9 ms（3.0×），每步 6.24 → \~2.5 ms（2.5×）——
量化+融合对两段都有 2.5–3× 收益，没有哪一段掉队。

### 4.2 已确认的结构特征

1. **48 ms 距 16.5 ms 的权重流量下界仍有 \~3×**：小形状 GEMM 的 Tensor Core 低利用率、
Q/DQ 与 dynquant 算子本身、softmax fp32 计算。
2. **精度配置不对称**：LLM 已 NVFP4（权重 1/4），但 **expert 仍是 FP8**——
而 expert 权重每帧被读 10 遍（3.1 GB，占 TRT 总权重流量的 \~2/3）。
**把 expert 也量化到 NVFP4 是性价比最高的单点优化**（流量下界省 \~8 ms，实际预计 4–8 ms）。
3. **NVFP4 动态量化链实测占 9.9%**：Top-20 layer 几乎被 `\_\_myl\_MulReshDyna\_\*`（每 LLM 层一个，
\~0.28 ms × 17）霸榜，全图 `\*Dyna\*` 类算子合计 6.1 ms。这是 NVFP4 "dynamic block scale"
设计的固有开销，值得与"LLM 回退 FP8 静态 scale（权重流量 1→2 GB，约 +3.7 ms）"做 A/B。
4. **KV cache 流量很小，量化价值有限**：GQA 只有 1 个 KV head，prefix KV 全帧 \~0.18 GB，
量化只能省 \~0.3 ms，不值得做。KV 的真正价值在跨帧复用（§5-A2）。
5. **语言序列被 padding 到 208**：LIBERO 实际 prompt 通常 <30 token；prefill 成本随序列长增长，
976 token 中约 18% 是 padding（对应 prefill 的 \~3 ms）。
6. **suffix FP16 是刻意的**：11-token 的 suffix 分支太小，Q/DQ 开销超过收益
（`pytorch\_to\_onnx.py` `suffix\_attn\_fp16`）——FP8 化在极小 GEMM 上已饱和，
进一步优化应走"减步数"而非"继续压 suffix 精度"。

\---

## 5\. 优化方向（按预期收益排序，收益均已用实测校准）

### A. 算法层（改计算量，收益最大）

1. **减少 denoise 步数 / 步数蒸馏**：实测每步成本 PyTorch 6.24 ms、TRT \~2.5–3.2 ms，
且时延-步数严格线性（§3.1）。10→4 步：PyTorch 省 37.4 ms（-29%），TRT 省 \~15 ms（-31%）。
方向：shortcut/consistency 蒸馏，或直接评估 N=4\~6 的任务成功率损失。
N 是引擎编译期常量，改 N 只需重导 ONNX/重建引擎。
2. **跨帧 prefix 复用（时序冗余）**：同一 episode 内 prompt 不变、场景缓变。
语言前缀 KV 可跨帧缓存，ViT 降频重编码 + 陈旧度门控。
上限 = T\_fixed 的大部分（PyTorch 68.3 ms 中的 ViT+prefill；TRT \~23 ms）——
机器人控制场景特有的创新点。
3. **异步执行（RTC 式推理）**：48 ms 推理产出 200 ms（@50Hz）的控制量，推理与执行天然可流水，
有效感知时延趋近于 0。配合方向 2，prefix 更新与 denoise 可进一步解耦。

### B. 系统层（改流量与 launch）

4. **expert 权重 NVFP4 化**（§4.2-2）：权重流量 3.1→0.8 GB/帧，预计省 4–8 ms（8–17%），
量化管线已支持 per-layer 配置，改动是配置级的。
5. **NVFP4 dynquant 链优化**（§4.2-3）：\~6 ms 的动态量化开销，方向包括
（a）LLM 回退 FP8 静态 scale 做 A/B；（b）合并 18 个 dynquant 算子为更大粒度。
6. **编译期对齐真实 prompt 长度**：按任务实际 token 分布重建引擎（如 opt=64），
或提供 2–3 档长度引擎按 prompt 分发，省 \~3 ms（§4.2-5）。
7. **视觉 token 削减**：ViT 在 TRT 中仅占 10.7%（\~5 ms），token merging/降分辨率的直接收益
≤3 ms，优先级下调；仅当与方向 2（降频重编码）结合时值得做。
8. **PyTorch 路径补齐 TRT 已验证的 hook**：fast\_infer + tokenize cache（省 \~3.65 ms host 开销）；
compile 路径稳态已是 cudagraph replay（§3.3），launch 侧无需再优化。
9. **层级流水：prefill 与首个 denoise step 重叠**（收益有界，见 §7）：实测重叠窗口
**3.22 ms（约占 TRT 实际时延 5–7%）**，实现复杂度高，ROI 中等。

### C. 平台层

10. **统一内存零拷贝管线**：相机帧直接落 pinned/unified 缓冲，前处理上 GPU。
（Eager 实测 HtoD memcpy 每帧 \~4.6 ms，compile/TRT 路径已被 fast\_infer 大幅压缩。）
11. **多实例/多机 batching**：expert GEMM 在 batch=1 时是 GEMV；一台 Thor 拖多台机器人时，
跨请求 in-flight batching 可数倍放大 Tensor Core 利用率（吞吐场景）。

### 不建议的方向（数据已否定）

* 继续压 suffix 分支精度（§4.2-6，太小不受益）；
* prefix KV 量化（§4.2-4，全帧仅 0.18 GB）；
* 纯算力型优化——利用率 \~14–19% 说明瓶颈不在算力；
* DLA 卸载——Thor（T5000）无 DLA 单元。

\---

## 6\. 复现与数据采集附录

### 6.1 已采集数据（`perf\_data/`）

|产物|用途|状态|
|-|-|-|
|`numsteps\_sweep\_pytorch.csv`|§3.1 固定开销/去噪循环分解|✅ 已分析|
|`pi05\_pt\_\*.csv`（Eager nsys stats）|§3.2 Eager kernel 归因、NVTX 阶段计时|✅ 已分析|
|`pi05\_ptc.nsys-rep/.sqlite`（torch.compile）|§3.3 CUDA graph 发现|✅ 已分析（sqlite 直读）|
|`pi05\_trt.nsys-rep/.sqlite`（TensorRT）|§3.3 CUDA graph 对照|✅ 已分析（sqlite 直读）|
|`model\_fp8\_nvfp4.engine\_profile.json` / `\_layers.json`|§4.1 TRT 逐层阶段归因|✅ 已分析|
|`analysis.md`|以上全部分析表|`analyze\_perf.py` 可重现|

两个已知的数采注意事项：

* `pi05\_pt.nsys-rep` 是 **Eager** 路径（命令行证实 `--profile-eager`），与 126.7 ms 的 compile
基线不是同一条路径，只用作 eager 对照组。
* host 侧 nsys 2025.5.2 无法解析 Thor 的 2026.2.1 报告；ptc/trt 的 stats CSV 导出在 Thor 侧
也失败（0 字节）。分析改为 **host 直接查询 `.sqlite` 导出**（标准 SQLite，绕过版本问题），
已固化在 `analyze\_perf.py: section\_graph\_launches`。

### 6.2 采集与分析命令（已全部执行完毕）

```bash
# Thor 容器内：
bash deployment\_scripts/collect\_perf\_data.sh pi05\_pt.nsys-rep
# host 侧：
python deployment\_scripts/analyze\_perf.py --perf-dir perf\_data --output perf\_data/analysis.md
```

§7 的"并行后 SM/DRAM 是否同时饱和"已通过补采回答：`pi05\_trt\_gpu.sqlite`
（`nsys profile --gpu-metrics-devices=all`，容器需 `--cap-add SYS\_ADMIN`），结果见 §7.3。

### 6.3 新增文件清单（均为新增，未修改任何原始脚本）

* `deployment\_scripts/pi05\_numsteps\_sweep.py` —— num\_steps 扫描实验（Thor 侧运行）
* `deployment\_scripts/collect\_perf\_data.sh` —— Thor 侧一键采集（自带 PYTHONPATH/transformers 补丁）
* `deployment\_scripts/analyze\_perf.py` —— host 侧 profiling 解析与表格生成
* `report.md` —— 本报告

\---

## 7\. 专题：层级流水（prefill 与首个 denoise step 重叠）可行性

**想法**：expert 第 L 层只依赖 LLM 第 L 层的 prefix KV，因此第 1 个 denoise step 的 expert
第 L 层可以与 LLM 第 L+1..18 层流水重叠，不必等整个 prefill 完成。

### 7.1 现状：两条路径都没有该优化

* **PyTorch**：`pi0\_pytorch.py:394-420` —— prefix forward 完整跑完 18 层返回 `past\_key\_values`
之后才进入 `while` 去噪循环，模块级严格串行。
* **TensorRT**：导出走同一个计算图；运行时 `trt\_torch.py` 在单条 stream 上 enqueue 整个引擎并
捕获为一张 CUDA graph（实测每帧仅 1 次 `cudaGraphLaunch`）。ONNX 张量级依赖虽允许
expert step-1 第 L 层在 LLM 第 L 层之后启动，但单流执行将其序列化。

### 7.2 收益上限（实测）

* **只有第 1 个 denoise step 能藏进 prefill**；第 2–10 步仍是纯串行。收益上限 = 一个 T\_step。
* 逐层负载实测极不对称：LLM 每层 ≈ 1.27 ms，expert 每层 ≈ 0.18 ms（每步 3.22 ms / 18）——
**LLM 单层耗时约为 expert 单层的 7 倍**，expert step-1 全部 18 层理论上可藏进
2–3 个 LLM 层的时间里。
* 重叠窗口实测：**min(expert 每步 3.22 ms, LLM 层 2–18 合计 21.7 ms) = 3.22 ms**，
约占 TRT 实际时延的 **5–7%**（PyTorch 路径对应为 6.24 ms，\~4.8%）。
* 对比：减步数（方向 1，每步都是收益，10→4 即 -30%）、expert NVFP4（方向 4，8–17%）——
本方向 ROI 中等。

### 7.3 瓶颈互补性判断（GPU metrics 实测）

`pi05\_trt\_gpu.sqlite`（`--gpu-metrics-devices=all`，100 µs 采样，10 次推理高度一致）：

|阶段|SMs Active|SM Issue|Tensor Active|
|-|-|-|-|
|prefill（ViT+LLM，前 47.8% 时间）|\~85%|**\~20%**|**\~20%**|
|expert ×10（后 52.1% 时间）|\~71%|**\~9%**|**\~10%**|

解读：

* 两个阶段 SM 都"有活在干"（Active 71–85%），但 **issue 槽位只用了 9–20%、Tensor Core 只用了
10–20%**——都是延迟/带宽受限，没有一个阶段把计算资源占满。
* 这正是重叠能获益的情形：prefill 期间有 80% 的 issue 槽位和 Tensor 容量闲置，足以吸收
expert step-1 的小 GEMV；反过来 expert 段的闲置更多。两阶段瓶颈互补得到实测确认。
* 唯一未能直接观测的是 DRAM 带宽（此 nsys 指标集不含 DRAM 项）：SM Active 高 + Issue 低说明
两阶段都在等内存，重叠会增加带宽争用；但 expert step-1 每层的权重+KV 流量（\~18 MB）仅为
LLM 每层 NVFP4 权重（\~55 MB）的 1/3，争用增量有限，预期仍能拿到窗口的大部分收益。

**结论更新：层级流水在资源层面可行（prefill 的 SM/Tensor 远未饱和），收益上限 3.22 ms（5–7%），
瓶颈仍在实现复杂度（TRT 需拆引擎多流执行），ROI 评级维持"中等"。**

### 7.4 实现代价

* TRT 无法表达"层间多流"，需拆成两个引擎（prefill 按层分组 + step-1 expert）由 host 多 stream

  * event 驱动，或放弃 TRT 在 PyTorch 侧用双流 + per-layer event 手工流水（每 episode 形状静态，
可整体 CUDA graph 化）。相对 5–7% 的收益，工程量偏大。
* 显存容量不是约束（128 GB 统一内存）；争用在 DRAM 带宽与 L2。

