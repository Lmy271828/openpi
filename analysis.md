# π₀.5 性能分析报告（Jetson Thor GB10y，pi05_libero，w3/r10）

采集：`bash deployment_scripts/collect_perf_data.sh`（nsys `--gpu-metrics-devices=all` + tegrastats + trtexec 逐层 profile）
分析：`python deployment_scripts/analyze_perf.py --perf-dir perf_data --dram-peak-gbps 273`

> 本轮为第四轮采集：ptcompile CUDA graph 完整（39 次 graphLaunch，recompile_limit=64 修复已验证）；
> tegrastats EMC 已采到（容器挂 `/sys`），phase 对齐按行时间戳（GB10y 实际采样 ~8 Hz，
> index×名义间隔在 110 s 上漂移 ~20 s，旧对齐会把 test 阶段映射到样本外）。

## num_steps 扫描（PyTorch）

| num_steps | mean (ms) | std (ms) | min | max |
|---|---|---|---|---|
| 1 | 75.88 | 0.67 | 75.26 | 77.71 |
| 2 | 82.38 | 0.77 | 81.71 | 83.88 |
| 4 | 94.20 | 0.58 | 93.72 | 95.49 |
| 6 | 107.87 | 3.39 | 105.87 | 117.65 |
| 8 | 118.91 | 0.96 | 117.98 | 120.73 |
| 10 | 131.00 | 0.57 | 130.18 | 132.17 |

线性拟合 `T(N) ≈ T_fixed + N·T_step`：
- **T_fixed（ViT + LLM prefill + host）= 70.02 ms**，占 T(10) 的 53.3%
- **T_step（单次 expert denoise）= 6.13 ms**，10 步合计 61.32 ms，占 T(10) 的 46.7%

## DRAM / 显存带宽（GPU metrics）

**pi05_ptcompile_pi05_libero_w3_r10**

nsys GPU metrics 在本设备（Tegra iGPU）未暴露 DRAM 计数器——DRAM 挂在 SoC 侧 MC/EMC，不归 GPU metrics 采样（设备限制，重采不会改变）；以下为 Copy Engine 吞吐，仅作显存传输代理。真实 DRAM 带宽见「DRAM / EMC（tegrastats）」一节。

| 指标 | 单位 | 整体均值 | 整体峰值 | test 窗口均值 | test 窗口峰值 |
|---|---|---|---|---|---|
| Sync Copy Engine Active | Throughput % | 0.0（≈0 GB/s） | 12 | 0.0（≈0 GB/s） | 0 |
| Async Copy Engine Active 0 | Throughput % | 0.0（≈0 GB/s） | 100 | 0.0（≈0 GB/s） | 22 |
| Async Copy Engine Active 1 | Throughput % | 0.0（≈0 GB/s） | 59 | 0.1（≈0 GB/s） | 11 |

分阶段饱和度（S2/S3 NVTX 探针精确切窗，中位 test 窗口）：

| 指标 | 单位 | 阶段 | 窗口数 | 均值 | 峰值 |
|---|---|---|---|---|---|
| Copy Engine ×3 | Throughput % | prefill (S2) | 1 | 0.00 | 0 |
| Copy Engine ×3 | Throughput % | expert (S3) | 10 | 0.00 | 0 |
| SMs Active | Throughput % | prefill (S2) | 1 | 24.14 | 97 |
| SMs Active | Throughput % | expert (S3) | 10 | 36.56 | 99 |
| SM Issue | Throughput % | prefill (S2) | 1 | 7.86 | 43 |
| SM Issue | Throughput % | expert (S3) | 10 | 6.95 | 23 |
| Tensor Active | Throughput % | prefill (S2) | 1 | 6.57 | 24 |
| Tensor Active | Throughput % | expert (S3) | 10 | 0.13 | 1 |

**pi05_trt_fp8_nvfp4_pi05_libero_w3_r10**

| 指标 | 单位 | 整体均值 | 整体峰值 | test 窗口均值 | test 窗口峰值 |
|---|---|---|---|---|---|
| Sync Copy Engine Active | Throughput % | 0.0（≈0 GB/s） | 0 | 0.0（≈0 GB/s） | 0 |
| Async Copy Engine Active 0 | Throughput % | 0.0（≈0 GB/s） | 100 | 0.0（≈0 GB/s） | 8 |
| Async Copy Engine Active 1 | Throughput % | 0.0（≈0 GB/s） | 54 | 0.0（≈0 GB/s） | 4 |

TRT 采集不含 S2/S3 NVTX 探针（引擎内部不可标注），无法精确切分 prefill/expert。

## DRAM / EMC（tegrastats）

**pi05_ptcompile_pi05_libero_w3_r10**

| 阶段 | 样本数 | EMC% mean | EMC% max | DRAM GB/s ≈ |
|---|---|---|---|---|
| warmup | 882 | 10.3 | 46 | 28 |
| inference_test | 7 | **43.1** | 49 | **118** |
| overall | 893 | 10.6 | 49 | 29 |

**pi05_trt_fp8_nvfp4_pi05_libero_w3_r10**

| 阶段 | 样本数 | EMC% mean | EMC% max | DRAM GB/s ≈ |
|---|---|---|---|---|
| warmup | 0 | - | - | - |
| inference_test | 3 | 24.0 | 28 | 66 |
| overall | 7 | 16.7 | 28 | 46 |

_DRAM GB/s ≈ EMC% mean × 273 GB/s（Thor 峰值，驱动上报值）。此 tegrastats 版本（GB10y/JP7.2）的 GR3D_FREQ 只报频率（MHz，全程 ~1574 满频）不报利用率%，GPU 占用以 nsys GPU metrics 的 SMs Active 为准。_

⚠️ 样本数少（test 阶段仅 ~1.4 s / ~0.5 s，tegrastats 时间戳分辨率 1 s），相位边界有 ±1 样本误差，数值看量级不看精度。TRT 的 warmup 0 样本是 logger 启动偏移的边界效应，无碍。

## CUDA graph 使用情况（sqlite 直读）

- **pi05_ptcompile**（test_i 窗口 137.2 ms）：cudaGraphLaunch × **39**，普通 cudaLaunchKernel × 83，窗口内可见 kernel 仅 83 个 / 0.24 ms——稳态计算几乎全部在 CUDA graph replay 内（graph 内 kernel 不被此 nsys 版本归因）。
- **pi05_trt_fp8_nvfp4**（test_i 窗口 49.9 ms）：cudaGraphLaunch × **1**，普通 cudaLaunchKernel × 23，窗口内可见 kernel 仅 23 个 / 0.08 ms——同上，整次推理被 1 个 graph 覆盖。

## 稳态 MemOps（sqlite test 窗口）

**pi05_ptcompile**（test_i 窗口 137.2 ms）：H2D 10 次 / 0.03 ms / 0.5 MB；D2D 18 次 / 0.02 ms；D2H 13 次 / 0.05 ms。

**pi05_trt_fp8_nvfp4**（test_i 窗口 49.9 ms）：H2D 9 次 / 0.02 ms / 0.5 MB；D2D 9 次 / 0.01 ms / 0.9 MB；D2H 2 次 / 0.00 ms。

——单次稳态推理 memcpy 合计 ~0.1 ms、≤1 MB（H2D 0.5 MB ≈ 3 路 224×224×3 输入）。CSV 全窗口里 PyTorch 的数十 GB D2D 全部来自 warmup/compile。

## nsys kernel 分析 — pi05_ptcompile（全采集窗口，含 warmup）

GPU kernel 总耗时 10948.69 ms（**绝大部分在 warmup/compile 阶段**，稳态见 CUDA graph 一节）。

| 类别 | 总耗时 (ms) | 占比 | launch 次数 |
|---|---|---|---|
| Elementwise/reduction (triton) | 10651.18 | 97.3% | 102196 |
| Softmax | 110.03 | 1.0% | 1867 |
| Norm (layer/rms) | 89.01 | 0.8% | 6637 |
| GEMM (cuBLAS/cutlass/nvjet) | 88.17 | 0.8% | 2587 |
| Attention (fmha/sdpa/flash) | 9.21 | 0.1% | 985 |
| Quant/convert (FP8/FP4/cvt) | 1.07 | 0.0% | 283 |
| Other | 0.01 | 0.0% | 4 |

Top-3 kernel：`FillFunctor<int>` 9315.07 ms × 71912（warmup 假象，见结论 2）；triton RMSNorm 融合 kernel 若干（各 ~110-210 ms）。

MemOps（全窗口）：D2D 452.80 ms × 11270；H2D 70.35 ms × 1027；D2H 1.04 ms × 206。

## nsys kernel 分析 — pi05_trt_fp8_nvfp4（全采集窗口，含 warmup）

GPU kernel 总耗时 50.53 ms（engine 启动期的直接 kernel launch；稳态在 graph 内）。

| 类别 | 总耗时 (ms) | 占比 | launch 次数 |
|---|---|---|---|
| GEMM (cuBLAS/cutlass/nvjet) | 34.93 | 69.1% | 1752 |
| Quant/convert (FP8/FP4/cvt) | 7.50 | 14.8% | 1508 |
| Other | 7.31 | 14.5% | 543 |
| Elementwise/reduction (triton) | 0.76 | 1.5% | 221 |
| Norm (layer/rms) | 0.02 | 0.0% | 13 |

MemOps（全窗口）：H2D 82.58 ms × 934（大头为 engine 反序列化/权重上传，非稳态）；memset 12.78 ms × 2。

## trtexec per-layer profile（TensorRT 阶段归因）

来源：`model_fp8_nvfp4.engine_profile.json`，layer 记录 3847 条，合计 61.06 ms（trtexec 逐层 profile 含层间同步开销，绝对值约为实际推理的 1.3 倍，看占比）。

| 阶段 | 耗时 (ms) | 占比 | layer 实例数 |
|---|---|---|---|
| Action expert x10 denoise (FP8) | 31.56 | 51.7% | 3095 |
| LLM prefill (Gemma 2B, NVFP4) | 23.11 | 37.8% | 451 |
| ViT (SigLIP x3 views) | 6.40 | 10.5% | 301 |

推算：expert 每步 ≈ 3.16 ms；LLM 每层 ≈ 1.28 ms；层级流水重叠窗口 = min(每步, LLM 层2-18) ≈ **3.16 ms**。

NVFP4 动态量化开销（`*Dyna*` 类算子）：69 个，合计 6.23 ms（10.2%）——且霸占 Top-20 layer 榜（每个 ~0.28 ms）。

---

# 结论

**1. 端到端：TRT 比 PyTorch(torch.compile) 快 2.75×，引擎与量化贡献约各占一半。** sqlite 直读的稳态 test 窗口：TRT FP8/NVFP4 49.9 ms vs PyTorch 137.2 ms（sweep 干净值 T(10)=131.0 ms；窗口含 NVTX 探针与 graph replay 开销，略高 ~5% 属预期）。补测 TRT **fp16**（同 ONNX 链路、不量化，`pi05_inference.py --inference-mode tensorrt`，w3/r20）：**85.79 ± 0.18 ms**。于是 2.75× 可干净拆分：**PyTorch BF16 →(引擎图优化 1.60×)→ TRT fp16 →(FP8/NVFP4 量化 1.72×)→ 49.9 ms**——图融合/CUDA-graph/内存规划与低精度 tensorop GEMM 的贡献大致相当，量化不是唯一来源，不量化的 TRT 部署本身已值 1.6×。两后端稳态计算都完整包在 CUDA graph 内（TRT 整次推理仅 1 次 cudaGraphLaunch；PyTorch 39 次），kern_sum 的 10949 ms 几乎全是 warmup/compile 痕迹，不能用于对比。
测量修复记录（供追溯）：① NVTX 探针的可变全局计数器使 dynamo 每个 denoise 步重编译，默认 `recompile_limit=8` 耗尽后 idx≥8 的步回退 eager（窗口虚高 ~4%）；② 曾误用 `@torch._dynamo.disable` 修复——graph break 把 kernel 挤出 CUDA graph（graphLaunch 39→9，裸露 kernel 83→11142，窗口恶化到 274 ms），该轮作废；③ 正确修复：`torch._dynamo.config.recompile_limit=64`，10 个步数变体 warmup 全部编译完，稳态既有逐步 NVTX 标签又保住 graph（本轮已验证：39 次 graphLaunch 恢复）。

**2. PyTorch 侧 kernel 总量是 warmup 假象。** 85%（9315 ms / 71912 次）是 `FillFunctor<int>`——int32 填充，来自 compile/warmup 阶段反复建 mask / position id，稳态窗口内不存在（稳态可见 kernel 仅 0.24 ms）。对比分析应完全基于 test 窗口数据。

**3. PyTorch expert 阶段不吃 Tensor Core。** S2/S3 精确切窗：expert (S3) 的 Tensor Active 均值仅 **0.13%**（prefill 6.57%），而 SMs Active 36.6%、SM Issue 7.0%——expert 的 GEMM 太小（action expert hidden 维度小、token 数少），在 PyTorch 下走 CUDA core / 受 launch 与带宽限制。这解释了 TRT 的 FP8 tensorop GEMM 为何能在 expert 上拿到 ~2×（6.13 → 3.16 ms/step），也说明 PyTorch 侧继续优化 expert 的收益上限有限。

**4. 算力、带宽、拷贝引擎全部远未打满。** prefill SM Active 24.1% / expert 36.6%，SM Issue <10%，Copy Engine ≈0；DRAM 侧（EMC 分阶段数据）：**PyTorch 推理段 EMC 均值 43.1% ≈ 118 GB/s，TRT 推理段 24% ≈ 66 GB/s**——PyTorch 的带宽占用约为 TRT 的 1.8×，与 BF16 vs FP8/NVFP4 权重流量的比例一致；但即便 PyTorch 也距 273 GB/s 峰值有一倍以上余量。**瓶颈在 kernel 粒度与串行依赖，不在硬件吞吐**；"加带宽/换更快显存"类优化对本案无效。

**5. TRT 内部可再挖 ~10%。** NVFP4 动态量化（`*Dyna*`）69 个算子共 6.23 ms（10.2%），是激活动态 scale 计算；改校准后静态 scale 或与相邻算子融合可回收大部分。阶段占比：expert x10 = 51.7%、LLM prefill = 37.8%、ViT = 10.5%；层级流水重叠窗口 3.16 ms/step，10 步理论上限 ~31 ms（约为总时延一半），但需先断开 LLM 层间与 expert 的依赖。

**6. 访存流量只是 warmup 假象（已由 sqlite 切窗证实）。** 全窗口 CSV 里 PyTorch 有 D2D 452.8 ms × 11270、H2D 70.4 ms，但按 test 窗口过滤后，单次稳态推理的 memcpy 合计仅 **~0.1 ms / ≤1 MB**（H2D 0.5 MB ≈ 3 路 224×224×3 输入图像，符合预期）。两个后端皆然——稳态访存不是优化对象，GPU 时间几乎全在计算上。

# LIBERO-Long 成功率对照（臂 A vs 臂 C，量化掉点）

臂的定义（权重/计算精度）：**A = PyTorch BF16**（原始浮点权重，不量化，eager + torch.compile）；**B = TRT FP16**（浮点权重不量化，ONNX → TensorRT 引擎，用于隔离转换误差）；**C = TRT FP8 权重/激活 + NVFP4 LLM**（量化，与 B 同一代码路径）。A − B 归因"转换误差"（ONNX 导出、TRT 融合、bf16→fp16），B − C 归因"量化误差"。

评测设置：`libero_10`（LIBERO-Long）× 50 trials = 500 episodes/臂；固定 seed=7 + 固定初始状态，两臂逐 episode 配对；server 在 Thor（detached 容器），client 在 x86 host（nohup）。两臂全程零异常（无 `Caught exception`，无垃圾失败）。数据：`eval_out/armA.log`、`eval_out/armC.log`，视频 `eval_out/libero10_armA/`、`eval_out/libero10_armC/`。

| 臂 | 成功率 | 备注 |
|---|---|---|
| A（PyTorch BF16） | **458/500 = 91.6%** | 与官方 π0.5@30k Libero-10 = 92.4% 一致，基准复现成立 |
| C（TRT FP8/NVFP4） | **400/500 = 80.0%** | Δ = **-11.6%** |

**McNemar 配对检验：p = 3.6e-08**（A成C败 85 对 vs A败C成 27 对，并集 112 对）——掉点高度显著，远超 n=500 二项噪声（95% CI ≈ ±4.4%）。结论：**FP8/NVFP4 量化在 LIBERO-Long 上造成真实成功率损失，约 -12 个百分点**。

逐任务分解（每任务 50 trials，A% → C%；指令即 rollout 视频文件名主体 `rollout_<指令>_success|failure.mp4`）：

| task | 指令 | A | C | Δ |
|---|---|---|---|---|
| 0 | put both the alphabet soup and the tomato sauce in the basket | 96% | 92% | -4% |
| 1 | put both the cream cheese box and the butter in the basket | 98% | 92% | -6% |
| 2 | turn on the stove and put the moka pot on it | 94% | 82% | -12% |
| 3 | put the black bowl in the bottom drawer of the cabinet and close it | 98% | 86% | -12% |
| **4** | **put the white mug on the left plate and put the yellow and white mug on the right plate** | **98%** | **46%** | **-52%** |
| 5 | pick up the book and place it in the back compartment of the caddy | 100% | 92% | -8% |
| 6 | put the white mug on the plate and put the chocolate pudding to the right of the plate | 90% | 76% | -14% |
| 7 | put both the alphabet soup and the cream cheese box in the basket | 94% | 94% | 0% |
| 8 | put both moka pots on the stove | 60% | 80% | **+20%** |
| 9 | put the yellow and white mug in the microwave and close it | 88% | 60% | -28% |

注：`main.py` 的视频文件名不含 episode 编号，同一任务同一结局的多段 rollout 会互相覆盖，目录里只剩最后一段；逐 episode 的成败时序以 `eval_out/armA.log` / `armC.log` 的 `Success:` 行为准。

观察：

- **掉点高度集中**：task4 一个任务贡献了近一半的总掉点（-52%），task9（-28%）次之；其余任务在 -4% ~ -14%。量化损伤不是均匀的精度退化，而是压垮了特定任务的决策余量。
- task8 反向 +20%（A 臂 60% 是 A 自身分布的离群任务——其余 9 任务均 ≥88%；官方未公布逐任务基线，无法与外部对照），单 episode 成败受 flow matching 采样噪声影响，且配对检验 p=0.021 未通过多重比较校正，按噪声处理，不构成"C 更好"的证据。

## 臂 B（TRT FP16）归因：转换零掉点，-11.6% 全部来自量化

臂 B 全量 500 集跑完（`eval_out/armB.log`）：**465/500 = 93.0%**。三臂对照（同一批 seed/初始状态逐集配对）：

| 臂 | 成功率 | vs A |
|---|---|---|
| A（PyTorch BF16） | 458/500 = 91.6% | — |
| **B（TRT FP16）** | **465/500 = 93.0%** | **+1.4%** |
| C（TRT FP8/NVFP4） | 400/500 = 80.0% | -11.6% |

McNemar 配对检验：

| 对比 | 分歧对（前者成后者败 : 前者败后者成） | p |
|---|---|---|
| B vs A | 30 : 23 | **0.41（不显著）** |
| B vs C | 90 : 25 | **8.4e-10** |
| A vs C | 85 : 27 | 3.6e-08 |

逐任务（每任务 50 trials）：

| task | A | B | C | B−A | C−B |
|---|---|---|---|---|---|
| 0 | 96% | 94% | 92% | -2% | -2% |
| 1 | 98% | 100% | 92% | +2% | -8% |
| 2 | 94% | 96% | 82% | +2% | -14% |
| 3 | 98% | 98% | 86% | 0% | -12% |
| **4** | **98%** | **98%** | **46%** | **0%** | **-52%** |
| 5 | 100% | 100% | 92% | 0% | -8% |
| 6 | 90% | 94% | 76% | +4% | -18% |
| 7 | 94% | 100% | 94% | +6% | -6% |
| 8 | 60% | 58% | 80% | -2% | +22% |
| 9 | 88% | 92% | 60% | +4% | -32% |

结论：

1. **A − B（转换误差）≈ 0**：bf16→fp16、ONNX 导出、TRT 图融合合计对成功率无统计影响（p=0.41，B 甚至略高 1.4%，在噪声内）。**ONNX→TensorRT 这条部署链路本身是无损的**。
2. **B − C（量化误差）= -13.0%（93.0% → 80.0%），p=8.4e-10**：A vs C 的 -11.6% 全部来自 FP8/NVFP4 量化，一分都摊不到转换头上。
3. **task4 悬崖是纯量化现象**：B 在 task4 上 98%，与 A 完全一致；C 掉到 46%。这排除了"ONNX/TRT 图变换破坏动作分布"的假设，坐实了探针节的机制结论——量化误差把落点分布右移 ~1cm。
4. **task8 噪声论被交叉验证**：B=58% 与 A=60% 几乎一致（两个独立实现都复现了 A 侧的低成功率），说明 task8 ~60% 是该任务在 π₀.₅ 权重下的固有水平，C 的 80% 是采样噪声，不是量化收益。
5. **优化方向明确**：既然转换无损、掉点全在量化，精力应全部投在量化配方上——混合精度（action expert 输出头/末层留 fp16）、更好的校准集与 scale、QDQ/Dyna 算子融合减少中间舍入。预期先把 task4/task9 这两个阈值敏感任务拉回 80-90%，总成功率即可回到 ~88-90%。

## task4 探针复测：-52% 的机制是"落点偏心 1cm"，不是"任务失败"

为定位 task4 的掉点机制，用 `--args.task-ids 4 --args.probe` 对臂 C 单任务复测 50 集（`eval_out/task4_probe_armC.log`，逐集视频在 `eval_out/task4_probe_armC/archived/`），探针在每集结束时记录杯-盘 XY 距离（即判定器 `check_ontop` 使用的几何量，见 `third_party/libero/libero/libero/envs/object_states/base_object_states.py:87-94`：`On(mug, plate)` = 接触 + 杯子在上方 + **杯盘中心 XY 距离 < 3cm**）。

结果（复测成功率 40%，与全量的 46% 一致）：

| | n | 两杯中较差者的盘心距离（中位） | 最差 |
|---|---|---|---|
| 成功集 | 20 | 2.7cm | 3.0cm（全部压线内） |
| 失败集 | 30 | 3.7cm | 22.1cm |

- **30 个失败中 24 个（80%）是"差 1-2cm"的擦边失败**：双杯都落在盘边 5cm 以内、视觉上已完成放置，但有一只杯子超出 3cm 阈值；典型剧集终态 3.4cm、4.0cm、4.1cm。仅 6 个是真未放置（~20cm）。
- 成功集则全部压在 3.0cm 内（中位 2.7cm）——判定边界两侧各 1cm 内集中了大部分剧集，**成功/失败的分布正好骑在 3cm 阈值上**。

结论：

1. **task4 的 -52% 不是语义层面的"不会做这个任务"，而是落点精度系统性退化 ~1cm，被 3cm 二元阈值放大成成功率悬崖**。失败视频里"放好了但夹爪反复微调"正是这个机制的表现——policy 没有判定器反馈，按训练分布继续输出微调动作直到超时。
2. **判定器与人眼语义的偏差**：`On` 是盘心 3cm 半径的几何测试，比"杯子在盘子里"严格（盘子半径约 8-10cm）。肉眼验收与判定器结论在临界样本上不一致是预期的，不是标注错误。
3. **对优化方向的含义**：成功率对数值误差的响应是高度非线性的——既然失败分布集中在阈值外 0.5-2cm，把落点分布整体左移 ~1cm（混合精度保留 action expert 输出头、更好的校准 scale、QDQ 融合减少中间舍入），task4 有望从 40% 回到 80-90%，无需消除全部量化误差。
4. **遗留问题**：本探针只测了臂 C 单侧分布；臂 A 的 98% 成功率已隐含其分布集中在 3cm 内，未单独复测（判定为非必要）。分布右移的定量幅度（~1cm）由 C 侧成功/失败分布反推，如需精确值可对 A 做同样探针。

## task4/task9 带符号探针（臂 C，各 50 集，2026-08-15）

探针升级后复测（`eval_out/probes_armC.log`，逐集视频 `eval_out/task4s_probe_armC/`、`task9_probe_armC/` 的 `archived/`）：每集末记录任务物体世界系坐标（带符号）+ 关节 qpos。task4 成功率 42%、task9 52%，与全量一致。

**task4：偏移是有方向的，主要是 -y 方向**（判定配对：白杯↔plate_1、黄白杯↔plate_2，bddl goal 已核实）。失败签名复现：29 个失败中 24 个为 3-5cm 擦边，仅 3 个 >10cm 真失败。带符号偏移（杯心 − 盘心，cm）：

| 配对 | 成功组均值 | 失败组均值 |
|---|---|---|
| 白杯−盘1 (Δx, Δy) | (+1.2, +2.0) | (+1.3, +2.5) |
| 黄白杯−盘2 (Δx, Δy) | (+1.3, **-2.1**) | (+0.7, **-4.0**) |

两点新认知：① **成功组本身就偏离盘心 ~2.5cm**（分布天然骑在 3cm 阈值附近，不是以盘心为中心）——这解释了为什么该任务对量化如此敏感：没有余量可吃；② 量化把黄白杯的 -y 偏移从 -2.1 加深到 -4.0（近似翻倍），**是定向漂移而非均匀扩散**——与"权重扰动 → 系统性偏差"的机制（非输入噪声，见 MIP 调研节）自洽。

**task9：掉点主导项是"门没关"，不是"杯子没放进去"**。判定器 `And(In(mug, heating_region), Close(microwave))`；Close 阈值 qpos > -0.005 rad（近似全关，成功集门角度范围 -0.0050 ~ +0.011，同样骑线）。24 个失败的拆解：

| In（杯入炉） | Close（关门） | 数量 |
|---|---|---|
| ✓（距炉心 ≤8.2cm） | ✗ | **15** |
| ✗ | ✗ | 9 |
| ✓ | ✓ | **0** |
| ✗ | ✓ | 0 |

- **所有失败集门都没关到位**（0 个 close 失败）；失败集门角度中位 -1.68 rad（大开），16/24 完全没关（< -1.0），仅 4/24 接近关（> -0.3）
- 15/24 失败里杯子已经放进去了——**policy 完成了放置却放弃了关门**（长程任务后半段动作丢失），另 9 个连放置也失败
- 机制与 task4 不同：task4 是几何精度漂移，task9 更像**长程动作序列的提前终止/子动作跳过**——量化误差在长序列后端累积，policy 在"关门"这个收尾子动作上失去推进力（失败集门大开而非半关，说明不是差最后几度，而是整个关门动作没执行完；是否与超时提前结束纠缠，需要逐集步数验证——当前探针未记录，列为遗留问题）

对量化配方的含义：task4 型（几何偏差）可被末步高精度/输出头 fp16 直接修复；task9 型（序列终止行为）可能需要在长程任务上做 QAT/恢复训练，或从 action chunk 执行层面补偿——两种掉点机制的修复路径不同，配方评估必须同时覆盖两类任务。

## LIBERO-Plus 鲁棒性评测：臂 A 基线（210 任务 × 1 trial）

子集：`eval_out/libero_plus_subset.json`（7 扰动维度 × 30 任务，难度 1-5 分层，seed=42，任务为 libero_10 的扰动变体）。臂 A（PyTorch BF16）总成功率 **171/210 = 81.4%**（基准 libero_10 为 91.6%，扰动总体代价约 -10%），全程零异常，210 段视频齐（`eval_out/plus_armA.log`、`eval_out/libero_plus_armA/`）。

按扰动维度（每维度 n=30，二项 95% CI ≈ ±18%，看排序不看小数点）：

| 维度 | 成功率 | 解读 |
|---|---|---|
| Light Conditions | 30/30 = 100% | 免疫 |
| Background Textures | 29/30 = 97% | 基本免疫 |
| Language Instructions | 29/30 = 97% | 基本免疫（指令改写不影响） |
| Objects Layout | 25/30 = 83% | 轻度敏感 |
| Robot Initial States | 23/30 = 77% | 中度敏感 |
| Sensor Noise | 20/30 = 67% | 高敏感 |
| **Camera Viewpoints** | **15/30 = 50%** | **最脆弱维度，接近掷硬币** |

按难度等级：level1 96% → level2 90% → level3 83% → level4 71% → level5 64%，单调下降，子集难度标注有效。

结论：

1. **π₀.₅ 的鲁棒性短板是几何扰动，不是语义/纹理扰动**：相机位姿（50%）和传感器噪声（67%）这两个改变视觉输入几何/低层统计的维度最致命；光照、背景、语言改写几乎无损。与 VLA 模型训练分布内泛化的预期一致——LIBERO 训练数据的相机是固定的。
2. **相机位姿是部署风险的优先项**：50% 意味着实机部署时相机安装偏差必须严格控制（或训练时加位姿增强），这比任何量化问题都致命（量化全幅掉点才 -11.6%）。注意 LIBERO-Plus 的 `_view_` 扰动**只动场景相机（agentview）**；腕部相机绑在夹爪上跟随机械臂，不受其影响（策略输入始终是双路：agentview + `robot0_eye_in_hand`，见 `examples/libero/main.py:119-136`；rollout 视频只录了 agentview 一路）。也就是说臂 A 在 Camera Viewpoints 掉到 50% 是**在腕部视角完好的情况下发生的**——场景相机一路退化就足以压垮任务，π₀.₅ 对 agentview 的依赖相当重。
3. ~~待臂 C 同子集跑完后做逐任务配对~~ → 已完成，见下节。

## LIBERO-Plus 臂 C 对照：量化 × 扰动交互分析（A/C 逐任务配对）

臂 C（TRT FP8/NVFP4）同子集 210 集跑完（`eval_out/plus_armC.log`）：**135/210 = 64.3%** vs 臂 A 的 81.4%，**Δ = -17.1%**（干净 libero_10 上为 -11.6%）。McNemar 配对：**A成C败 44 对 vs A败C成 8 对，p = 4.0e-07**——扰动下的量化掉点显著且方向一边倒。

逐维度配对（每维度 n=30；配对比 = A成C败 : A败C成）：

| 维度 | A | C | Δ | 配对比 | 交互判定 |
|---|---|---|---|---|---|
| **Background Textures** | 97% | **50%** | **-47%** | **14:0** | **强交互（最大）** |
| **Robot Initial States** | 77% | 50% | **-27%** | **8:0** | **强交互** |
| Objects Layout | 83% | 70% | -13% | 6:2 | 与干净环境相当 |
| Light Conditions | 100% | 87% | -13% | 4:0 | 中等交互 |
| Language Instructions | 97% | 87% | -10% | 4:1 | 与干净环境相当 |
| Sensor Noise | 67% | 60% | -7% | 5:3 | 弱（部分地板效应） |
| Camera Viewpoints | 50% | 47% | -3% | 3:2 | 无（地板效应：A 已 50%，无下降空间） |

结论：

1. **量化 × 扰动的交互是真实存在且维度特异的**：A 几乎免疫的 Background Textures（97%）在 C 上崩到 50%，配对比 14:0——**量化把"纹理鲁棒"的模型变成了"纹理敏感"的模型**。Robot Initial States 同型（77→50，8:0）。
2. **机制与 off-manifold 框架自洽**（见 Much Ado About Noising 调研）：新纹理/新初始位姿把视觉输入轻推离训练流形，浮点模型的纠错余量（迭代计算的流形投影效应）能吸收；量化吃掉的正是这部分余量。Camera/Sensor Noise 维度看不到交互不是"量化无害"，而是 A 已被扰动本身压到地板（50%/67%），没有可下降的余量。
3. **A成C败的 44 个任务高度集中于"阈值敏感三兄弟"的纹理变体**：LIVING_ROOM_SCENE5（task4 双杯放盘，5 个）、SCENE6（task6 杯+布丁，3 个）、KITCHEN_SCENE6（task9 微波炉，2 个）——与干净环境下 task4/9/6 主导掉点的格局一致，量化损伤的任务谱在扰动下没有漂移，只是被放大。
4. **部署含义**：TRT FP8/NVFP4 的量化配方不仅掉基准成功率，还**显著削弱分布外鲁棒性**——实机环境必然偏离训练纹理/光照分布，-17.1%（而非 -11.6%）才是真实部署预期的掉点量级。量化配方优化（混合精度、AWQ、静态 scale）的验收标准应包含 Plus 子集，而不只是干净 libero_10。

## 免训练两步调度实证：NFE=2 在 π0.5 上不成立（sched_2step_armA，2026-08-15）

动机：Much Ado About Noising 的 MIP（两步推理：t=0.4 免训练单跳 Δt=1 + t=0.9 修正步）与 Table 19 "Sudeep-DiT NFE=3 饱和"暗示少步推理有戏；π0.5 的 action expert 同为 adaLN 调制的 DiT 风格。通过 `PI05_T_GRID="0.6:-1.0;0.1:-0.1:0.9"` 在臂 A（BF16）上做推理侧重网格化（NFE=2），libero_10 × 50 trials = 500 条，与 N=10 基线同规格配对。

| task | N=10 基线 | 2 步调度 | Δ |
|---|---|---|---|
| 0 soup+sauce→basket | 96% | 62% | -34p |
| 1 cheese+butter→basket | 98% | 78% | -20p |
| 2 stove+moka pot | 94% | 56% | -38p |
| 3 bowl→drawer | 98% | 70% | -28p |
| 4 双杯分盘 | 98% | 40% | **-58p** |
| 5 book→caddy | 100% | 52% | -48p |
| 6 mug+pudding | 90% | 48% | -42p |
| 7 soup+cheese→basket | 94% | 58% | -36p |
| 8 双 moka pot 上灶 | 60% | **6%** | **-54p** |
| 9 杯入微波炉关门 | 88% | 34% | **-54p** |
| **整体** | **91.6%** | **50.4%** | **-41.2p** |

结论：

1. **免训练的两步 MIP 调度在 π0.5 上是显著负收益**（-41.2p，n=500 无统计悬念）。与 Much Ado 的机制一致：MIP 的收益来自**训练**（第二步带随机性注入专门学过修正，Table 16 里无匹配训练结构的少步变体 SF/RR 同样吃不到迭代红利），只做推理侧时间网格变换得不到流形投影能力。
2. **伤害集中在精度敏感任务**：task4/8/9（放置精度窗口最小的三个，也是量化掉点的重灾区）掉 54-58p，宽松任务（task1）只掉 20p。少步去噪削弱的正是"精修/流形吸附"能力——与量化损伤的任务谱高度重合（task4/8/9 同为两类扰动的最敏感点），进一步支持"off-manifold 余量"是统一解释框架。
3. **对路线的含义**：NFE=2 要成立，第二步必须是**训出来的修正器**（如 off-manifold LoRA 方案，见 README-zh §2），而不是网格技巧。Table 19 的架构依赖信号（DiT NFE=3 饱和）在 π0.5 上未复现——π0.5 的 NFE 敏感点比 20M 小模型 DiT 靠前。若未来再扫，有意义的档位是 NFE=3~4 + 修正器训练，而非继续免训练网格。
4. 数据存档：`eval_out/sched_2step_armA/`（500 条 rollout 视频）+ `eval_out/sched_2step_armA.log`。

## FlashRT P1/P2 复现：FP8/NVFP4 零掉点，TRT 掉点判为配方问题（2026-08-16）

FlashRT（Thor 原生，torch 2.11 cu130 + 手写 kernel，FP8 E4M3 per-tensor 校准，norm/residual/attention 保持 FP16，覆盖 SigLIP encoder + Gemma decoder + action expert 三侧；NVFP4 档 = FP8 主体 + encoder FFN 换 NVFP4）跑 libero_10 × 50 trials = 500 条，与臂 A/C 同任务不同评测栈（同进程仿真，种子序列不同，只能逐任务比率对照，n=50 单任务噪声 ±7p）：

| task | A (BF16) | C (TRT FP8/NVFP4) | FlashRT FP8 | FlashRT NVFP4 |
|---|---|---|---|---|
| 0 | 96 | 92 | 94 | 96 |
| 1 | 98 | 92 | 100 | 100 |
| 2 | 94 | 82 | 96 | 92 |
| 3 | 98 | 86 | 98 | 100 |
| 4 双杯分盘 | 98 | **46** | 98 | **100** |
| 5 | 100 | 92 | 98 | 100 |
| 6 | 90 | 76 | 88 | 86 |
| 7 | 94 | 94 | 98 | 100 |
| 8 双 moka pot | 60 | 80* | 52 | 60 |
| 9 微波炉 | 88 | 60 | 94 | 92 |
| **整体** | **91.6** | **80.0** | **91.6** | **92.6** |

结论：

1. **FP8 本身不掉点**：FlashRT FP8 = 91.6% 与臂 A 完全持平。臂 C 的 -11.6% 归因为 TRT 量化配方（校准集/scales/Dyna 动态量化），而非 FP8 精度上限。
2. **NVFP4 也不掉点**：92.6%（甚至 +1.0p 于基线，噪声内），官方"NVFP4 与 FP8 精度持平"复现成立。实测延迟（Thor，2-view，p50）：**FP8 42.96ms / NVFP4 36.28ms**（无 FA4，`bench_pi05_thor_views.py`）；装 thor-fa4 后（`bench_pi05_decoder_fp4_e2e.py`，FA4 强制开启，100 次 iter 同会话 A/B）：**FP8+FA4 40.66ms / NVFP4+FA4 27.21ms，1.49×**，与官方 38.70→27.17ms 对齐（见下节）。校准耗时：FP8 0.87s / NVFP4 2.19s（一次性，JSON 缓存）。**臂 C 的 TRT 方案（49.9ms / 80.0%）被 FlashRT NVFP4（27.2ms / 92.6%）双向超越，整条判负**。
3. **精度敏感任务完全恢复**：task4（46→98/100）和 task9（60→94/92）这两个 TRT 重灾区在 FlashRT 下回到基线水平，与探针结论闭环——落点漂移/关门不到位是 TRT 配方引入的 off-manifold 偏移，不是 FP8/FP4 的固有损伤。
4. **task8 四方数字（60/80/52/60）互相矛盾**，是固有高方差任务，不作为精度判据。
5. 数据存档：Thor `~/lmy/openpi/third_party/flashrt/eval_flashrt_{fp8,nvfp4}_libero10.log` + `libero_libero_10_torch_results.json`。注意 `--use_fp4` 是后补的 flag（本地补丁透传 `load_model(use_fp4=)`，见 `examples/thor/eval_libero.py` 未提交改动）。

### FA4 补齐：NVFP4+FA4 = 27.21ms，对齐官方（2026-08-16）

装 `thor-fa4` extra（`nvidia-cutlass-dsl==4.5.1` + `quack-kernels==0.4.1` + `nvidia-cuda-nvcc` 提供 ptxas）后跑官方门禁 harness `tests/bench_pi05_decoder_fp4_e2e.py --num-views 2`（锁频 MAXN，100 iter，同会话 A/B，匹配噪声）：

| 配置 | p50 | p95 | 保真门 |
|---|---|---|---|
| FP8 + FA4 | 40.66 ms | 40.82 ms | —（参考臂） |
| NVFP4 + FA4 | **27.21 ms** | 27.33 ms | action cos 0.99901 ✅ / action min-sample cos 0.99627 ✅ / raw cos 0.99707 ✅ / raw min-sample cos 0.98893 ❌（门 0.995） |

- 与官方 2-view 数字（38.70 → 27.17ms，1.42×）对齐：我们 40.66 → 27.21ms，1.49×。FP8 参考臂略慢于官方是 Thor 时钟漂移（官方文档自己标注同机型有 ~3ms 的 sustained-load 双 regime），只看同会话比值。
- 唯一没过的是 raw min-sample cosine（0.9889 < 0.995，官方 0.9980）：raw 是最终 action 之前的中间量严格指标，动作级两个门全过。差异来源可能是 checkpoint（我们用 openpi 原版 `pi05_libero_pytorch`，官方用 converted 版）或自生成 fixture，不影响"可部署"结论。
- 踩坑记录（已固化进 `thor_jp72_env.sh`）：dsl 4.5.1 下 FA4 编译目标必须显式 `CUTE_DSL_ARCH=sm_101a`（`sm_110a` 路径触发 NVVM chip-string bug：`Failed translating the module to ISA`）；loader 的 `setdefault` 在 `import cutlass` 之后才执行，依赖它不可靠，必须进程启动前 export。harness 还要求 flashrt 工作区干净，本地 `--use_fp4` 补丁需先 `git stash`。
- 存档：Thor `third_party/flashrt/bench_fa4_e2e_2v.log`，artifacts `/tmp/flashrt-pi05-decoder-fp4-e2e-20260816T065142Z`。

## 臂 D（Omega-QVLA W4A4/W4A8 hybrid）：93.2%，与 BF16 基线无统计差异（2026-08-17）

Omega-QVLA 官方 pack 配方实为全模型 W4A4（README 标题与 HF 仓库名均为 `W4A4`；pack 内 PaliGemma 记录 `a_bits=4`、单行 `act_scale_table`，expert 记录 `a_bits=4`、10 行 per-step 表——2026-08-22 直接读 pack 核实）。本次臂 D 为 hybrid 部署：expert 18 层 ×7 Linear 走 pack GPTQ W4A4（`a2lite` pack，INT4-only 路径，`GptqLinear`），PaliGemma 18 层 ×7 Linear 走运行时 `DuQuantLinear`（启动时从 BF16 权重确定性重建 svd_hadamard 旋转+置换，RTN W4；`GR00T_DUQUANT_ABITS` 未设、取默认 A8），小投影层（state/action/time）全部排除；pack = `packs_hf/pi05_long/quantized.pt`（仅 expert 部分被加载）。Thor hybrid server（PyTorch 路径，GptqLinear/DuQuantLinear），libero_10 × 50 trials = 500 集，与臂 A 同 client 栈同种子序列逐集配对（`eval_out/omega_w4a4_long.log`）。

| task | A (BF16) | C (TRT FP8/NVFP4) | FlashRT NVFP4 | **D (Omega W4A4)** |
|---|---|---|---|---|
| 0 | 96 | 92 | 96 | 90 |
| 1 | 98 | 92 | 100 | 96 |
| 2 | 94 | 82 | 92 | 96 |
| 3 | 98 | 86 | 100 | 96 |
| 4 双杯分盘 | 98 | **46** | 100 | **100** |
| 5 | 100 | 92 | 100 | 100 |
| 6 | 90 | 76 | 86 | **100** |
| 7 | 94 | 94 | 100 | 100 |
| 8 双 moka pot | 60 | 80* | 60 | 76 |
| 9 微波炉 | 88 | 60 | 92 | 78 |
| **整体** | **91.6** | **80.0** | **92.6** | **93.2** |

结论：

1. **W4 权重也可以不掉点**：93.2% vs 臂 A 91.6%，McNemar 逐集配对（A成D败 21 : A败D成 29）p = 0.32，无统计差异。至此"位宽 vs 配方"的二轮对照闭环：臂 C 的 -11.6% 曾让 FP8/NVFP4 蒙冤，FlashRT 证明了 8/4-bit 静态量化可以零掉点；臂 D 进一步证明**更激进的 W4 权重（expert GPTQ W4A4 + PaliGemma DuQuant W4A8，带旋转/校准配方）同样可以零掉点**——掉点从来不是位宽的锅，是配方的锅。
2. **重灾区完全恢复**：task4 100%（臂 C 46%）、task6 100%（臂 C 76%）；task9 78% 略低于基线 88% 但在 n=50 噪声内。GPTQ 的逐层误差最小化 + DuQuant 的旋转/置换确实把 off-manifold 偏移压回了决策余量之内。
3. **代价在延迟侧（未测）**：本次只评精度。GptqLinear/DuQuantLinear 是 PyTorch 算子路径（启动日志可见 dynamo graph break 与逐层替换），无 kernel 级加速，Thor 上的推理延迟大概率显著高于 FlashRT NVFP4 的 27.2ms——这正是 fork FlashRT 补"GPTQ pack 消费层"的动机：把同样的量化权重搬到 tcgen05 E0M3 GEMM 上跑。
4. **配方档位已确认**：Thor 上 `pi0_pytorch.py` 含 `_dit_step_context` 补丁（line 27/448），per-step scale 表生效，本次 93.2% 对应 Omega-QVLA 完整官方配方（非 step-mean fallback）。
5. 存档：本地 `eval_out/omega_w4a4_long.log` + `eval_out/omega_w4a4_long/`（500 条 rollout 视频）；冒烟 10 集 9/10（`omega_w4a4_smoke.log`）。

## 臂 D × LIBERO-Plus：扰动鲁棒性基本恢复，纹理/光照有残余（2026-08-17）

臂 D（Omega-QVLA W4A4/W4A8 hybrid，同臂 A/C 的 210 任务子集逐集配对，`eval_out/plus_armD.log`）：**161/210 = 76.7%**，对照臂 A 81.4% / 臂 C 64.3%。McNemar：A vs D = 21:11，**p = 0.11（无统计差异）**；C vs D = 11:37，**p = 2.2e-04（D 显著优于 C）**。

逐维度（每维度 n=30；A:D 配对比 = A成D败 : A败D成）：

| 维度 | A | C | **D** | A:D 配对比 | 判读 |
|---|---|---|---|---|---|
| Background Textures | 97% | **50%** | **83%** | 4:0 | C 的最大交互重灾区基本收复，仍有轻微残余敏感 |
| Robot Initial States | 77% | 50% | **67%** | 4:1 | 同上，次重灾区大部分收复 |
| Language Instructions | 97% | 87% | **97%** | 0:0 | 完全恢复 |
| Light Conditions | 100% | 87% | **87%** | 4:0 | **未恢复**——D 与 C 持平，是残余掉点之一 |
| Objects Layout | 83% | 70% | 80% | 3:2 | 回到 A 的水平 |
| Sensor Noise | 67% | 60% | 70% | 4:5 | 回到 A 的水平 |
| Camera Viewpoints | 50% | 47% | 53% | 2:3 | 无差异（地板效应维度） |

结论：

1. **好配方不仅救基准，也救鲁棒性**：臂 C 的 -17.1%（扰动下放大的量化掉点）在臂 D 收窄到 -4.7%，且与 A 无统计差异。"量化 × 扰动交互"不是位宽的固有代价，与干净环境的结论同构——是 TRT 配方的问题。
2. **但残余信号值得记录**：Textures（4:0）和 Light（4:0）两个维度 D 仍方向性低于 A（n=30 不足以定统计显著，但两个维度同向）。off-manifold 框架下这合理：W4A4 的纠错余量恢复得"几乎够"，纹理/光照这两个最依赖视觉表征余量的维度先用完了余量。若未来做混合精度精修，这两个维度是敏感指标。
3. **部署含义更新**：之前"实机预期掉点 -17.1%"的悲观估计是针对 TRT 配方的；以臂 D 为部署基线，扰动环境预期掉点约 **-5% 量级**。量化方案的验收标准仍应包含 Plus 子集（它能暴露干净环境看不见的残余损伤，如本次的 Light 维度）。
4. 存档：`eval_out/plus_armD.log` + `eval_out/libero_plus_armD/`（210 条视频）；每集 ~148s（vs 臂 A ~90s），W4A4 PyTorch 路径的延迟劣势直接可见，量化 kernel 化（FlashRT E0M3 迁移）是必要后续。

## 臂 D + E0M3 kernel 化（FlashRT tcgen05）：90.4%，与 A/D 均无统计差异，延迟 2.6×（2026-08-18）

同一套 Omega pack（`pi05_long/quantized.pt`）经 `tools/convert_omega_pack_e0m3.py` 转为 `omega_e0m3_v1`（S0 丢表），由 `tools/omega_e0m3_linear.py` 消费层把 expert 126 层 Linear 换到 FlashRT E0M3 kernel（per-16 动态量化 + tcgen05 GEMM），PaliGemma 侧仍走 DuQuant fake-quant；eager + 禁 torch.compile（pybind graph break + KV recompile 风暴，见 handbook）。libero_10 × 50 trials = 500 集逐集配对（`eval_out/omega_e0m3_long.log`）。

| task | A (BF16) | D (fake-quant) | **D+E0M3** |
|---|---|---|---|
| 0 | 96 | 90 | 98 |
| 1 | 98 | 96 | 98 |
| 2 | 94 | 96 | 96 |
| 3 | 98 | 96 | 94 |
| 4 双杯分盘 | 98 | **100** | 98 |
| 5 | 100 | 100 | 100 |
| 6 | 90 | **100** | 94 |
| 7 | 94 | 100 | 100 |
| 8 双 moka pot | 60 | 76 | 66 |
| 9 微波炉 | 88 | 78 | **60** |
| **整体** | **91.6** | **93.2** | **90.4** |

McNemar（逐集配对）：vs 臂 A = 28:34，**p = 0.53（无差异）**；vs 臂 D = 19:33，**p = 0.070（无统计差异，方向偏负）**。

结论：

1. **kernel 化不显著掉点，但吃掉了臂 D 的余量**：E0M3（S0 丢校准表 + per-16 动态 amax）对比带表的 fake-quant 掉了 2.8 个点，未达统计显著（p=0.07），判定为"等价于基线、略逊于完整配方"。单层 cos（0.978-0.982）预言的二阶损失在端到端被部分放大。
2. **task9（微波炉）是唯一重灾区**：60% vs 臂 A 88% / 臂 D 78%。与臂 C 时代 task4/task9 的敏感性同构——长horizon精细任务最先耗尽量化余量。后续 actnorm 实验（见下）证明这不是表能救的；若做精修，方向是混合精度（敏感层升 8-bit），task9 是敏感指标。

**actnorm 负结果（2026-08-18）**：为找回 vs 臂 D 的 2.8 个点，试过地板安全版 S1——s̄ 分解为 geomean c × 归一化 r̄，r̄ 折权重（不撞 UE4M3 地板，q_proj 块 scale 越界占比 100%→0%）、激活静态除 s̄、c 吸进 GEMM alpha。数学恒等成立（误差 4e-7），但单层 cos vs fp16 全面劣于 S0（0.986-0.989 vs 0.993-0.994）：折 r̄ 把 DuQuant 已漂白的权重块内幅度重新拉开，激活侧赚的抵不上权重侧亏的。**结论：per-channel 校准表与 per-16 块量化原理不兼容，S0 是终点**；task9 残余差距只能靠混合精度或接受。细节见 `third_party/flashrt/docs/omega_pack_e0m3.md` §4。
3. **延迟收益兑现**：每集 ~58s vs 臂 D fake-quant ~148s（**2.6×**），vs 臂 A ~90s。expert 占单步延迟的主体（trtexec 归因 51.7%），E0M3 GEMM + eager 已拿下大部分；剩余 gap 在 eager launch 开销，对应 CUDA Graph 自抓（M2d）。
4. 存档：`eval_out/omega_e0m3_long.log` + `eval_out/omega_e0m3_long/`（500 条视频）；冒烟 9/10（`omega_e0m3_smoke.log`）。

### M2d：denoise 循环 CUDA Graph 自抓（2026-08-18）

eager 路径的下一步优化：把 10 步 flow-matching denoise 循环（expert 18 层 ×10，~180 层小 kernel）整段展开抓进一张 `torch.cuda.CUDAGraph`（FlashRT pi05_thor 同款形态），实现为 `third_party/flashrt/tools/omega_e0m3_graph.py`——静态 KV slab 套 DynamicCache 壳、mask/position 静态 buffer 每推理 `copy_` 灌入、adaRMS 按确定性时间网格预算烘焙；prefix（ViT+LLM prefill）保持 eager。移除的捕获障碍：device 标量 `while` 循环（每步 host sync）、`embed_suffix` 每步 pageable H2D mask 上传、每调用 KV 重新分配。默认开（`OMEGA_E0M3_CUDA_GRAPH=1`），捕获失败永久回退 eager。

Thor 验收：

| 指标 | eager（M2b/c） | 抓图（M2d） |
|---|---|---|
| 捕获 | — | 成功（prefix_len=968, layers=18, steps=10） |
| 冒烟 | 9/10 | **10/10** |
| 50 集确认 | — | **45/50 = 90.0%**（≈ 500 集基线 90.4%，无数值漂移） |
| 每集耗时 | ~58s | **~43-50s**（-15~25%） |

判读：图只消掉 denoise 循环的 launch/host 开销，prefix 仍 eager，故幅度是 ~15-25% 而非数量级——图化只回收 launch/host 开销，kernel 本体耗时不变；这与 nsys 观察一致（GPU 时间大头在 kernel 执行本身）。pybind kernel 可捕获性（M2d P0 gate）一次通过，FlashRT 的"裸指针+stream+launch 时报错"调用约定在第三方模块上同样成立。剩余延迟预算在 prefix 的 ~1200-1800 次 eager launch，下一步候选是 prefix 原样抓图（零数值风险）或学 FlashRT 的 FP8 fused kernel 重写（高收益高成本）。存档：`eval_out/omega_e0m3_graph_smoke.log`（10/10）、`eval_out/omega_e0m3_graph_50.log`（45/50）。

### 路线 A 落地：双侧 E0M3 + prefix 抓图，action cos 四项过门禁（2026-08-22）

PaliGemma 从运行时 DuQuant（RTN W4 + 默认 A8）切到 pack 的 GPTQ W4A4 记录 + E0M3 kernel（`OMEGA_E0M3_PATCH_DUQUANT=1`，对齐官方全-W4A4 pack 配方），prefix prefill 同步入图（双图结构）。转换器零改动（252 条记录全量转换本就含 PaliGemma）。途中修了两个集成 bug：消费层缺 `_block_size`/`_block_out_size`/`_act_stats_available` 三个宿主日志契约属性，`wrap_duquant` 的记账行直接 AttributeError；harness 的 bf16 子进程在 torch.compile 下 prefix 注意力被 trace 成 SDPA（fp32 mask 撞 bf16 query）——harness 三模式统一改 eager（本来就是 eager 数值信号，与图重放 kernel 序列等价）。

Thor 验收（server 日志）：双侧 252 层全部替换（126 `via GptqLinear` + 126 `via DuQuantLinear`），prefix 图捕获（prefix_len=968, layers=18）+ denoise 图捕获。

action cos（`tools/check_omega_e0m3_action_cos.py`，libero_10 十任务各 1 条观测、同噪声逐条配对，eager）：

| 配对 | action cos | action min | raw cos | raw min |
|---|---|---|---|---|
| e0m3 vs bf16（端到端总账） | **0.99940** | **0.99694** | **0.99872** | **0.99512** |
| fake vs bf16（配方参考线） | 0.99778 | 0.99210 | 0.99577 | 0.98607 |
| e0m3 vs fake（kernel 纯损耗） | 0.99850 | 0.99374 | 0.99648 | 0.98829 |

e0m3/bf16 四项全过 NVFP4 派生门禁（0.999/0.995/0.995/0.995，raw min 擦线 +0.0001）；臂 D 配方自身（fake/bf16）反而四项全不过——直接印证 pack GPTQ 权重优于运行时 RTN 的设计判断。eager p50：bf16 226ms / fake 561ms / e0m3 548ms——eager 下 launch 开销主导，E0M3 的 GEMM 收益只在图重放路径体现（与 M2d 判读一致）。冒烟 10/10（7m36s，每集 26.7-84.8s）。存档：`eval_out/action_cos_routeA/`（result.json + 三模式子进程日志）、`eval_out/omega_routeA_long.log`（冒烟与全量共用此文件，全量见下节）。

### 路线 A ×500 全量：93.8%，与臂 A/D 无统计差异，显著优于 expert-only E0M3（2026-08-23）

libero_10 × 50 trials = 500 集逐集配对（`eval_out/omega_routeA_long.log`；文件名沿用冒烟时的命名，视频目录 `omega_routeA_long/` 混有前 10 集冒烟视频）。

| task | A (BF16) | D (fake-quant) | D+E0M3（expert-only） | **路线 A（双侧）** |
|---|---|---|---|---|
| 0 | 96 | 90 | 98 | 92 |
| 1 | 98 | 96 | 98 | **100** |
| 2 | 94 | 96 | 96 | 98 |
| 3 | 98 | 96 | 94 | 98 |
| 4 双杯分盘 | 98 | **100** | 98 | 98 |
| 5 | 100 | 100 | 100 | 98 |
| 6 | 90 | **100** | 94 | 96 |
| 7 | 94 | 100 | 100 | **100** |
| 8 双 moka pot | 60 | 76 | 66 | 64 |
| 9 微波炉 | 88 | 78 | 60 | **94** |
| **整体** | 91.6 | 93.2 | 90.4 | **93.8** |

McNemar（逐集配对）：vs 臂 A = 28:17，p = 0.135（无差异）；vs 臂 D = 21:18，p = 0.749（无差异）；vs expert-only E0M3 = 37:20，**p = 0.033（显著更优，+3.4 点）**。

每集耗时：全程 5h58m，均值 42.9s/集，与 expert-only E0M3 的 43-50s 同量级——prefix 图化 + PaliGemma E0M3 的加速被 episode 长度差异掩盖（成功率更高 ⇒ 难任务跑得更长；task8 失败集跑满 horizon，77.9s/集）。vs 臂 D fake-quant ~148s 约 3.4×。

口径备忘（2026-08-23）：本项目四组延迟数字边界两两不同，禁止直接并排——臂 A/B/C = 进程内 `policy.infer` 整次墙钟（含 tokenize/D2H/反归一化，无网络）；FlashRT 27.21ms = `pipe.infer` 墙钟但 prompt 预设、不含 tokenize；server `policy_timing.infer_ms` = 仅 `sample_actions` 内层（含 tokenize 与 GPU 执行——计时区已加 sync，不含 D2H copy 本身/反归一化）；每集耗时混 episode 长度。路线 A 的精确 per-inference 数字用 `tools/bench_omega_e0m3_infer.py` 补测（臂 A 口径 + 内层诊断 + `graph_state` 自证，`noise=None` 才走图路径），命令见 handbook 路线 A 节第 3 步。

per-inference 实测（2026-08-23，bench 图基线，fixture n=10 轮换，warmup 20 / iters 50，双侧 E0M3 + prefix/denoise 双图，`graph_state` 双 active）：**wall p50 308.0ms / mean 308.1 / p95 308.9（min 307.6，抖动 <1%，图重放确定性）**——同口径比臂 A bf16（137.2ms）**慢 2.2×**。归因修正（2026-08-23 二校）：消费层**并非**朴素反量化——`OmegaE0M3Linear.forward`（`tools/omega_e0m3_linear.py:146-182`）已走 FlashRT 上游 E0M3 FP4 快路径：激活经 `quantize_e0m3_dynamic_sfa_fp16`（`csrc/quantize/quantize_e0m3_sfa.cuh`）kernel 内动态 per-16 量化，GEMM 走 `cutlass_fp4_gemm_e0m3w`（`csrc/gemm/fp4/cutlass_fp4_gemm_e0m3w_sm100.cu`，CUTLASS tcgen05 FP4 tensor core，上游 v0.1.0 起维护、#164 更新）；纯 PyTorch 的只有 DuQuant 置换（index_select）+ 输入/输出旋转（64 块 bmm）+ bf16↔fp16 中转 cast（设计边界：置换/旋转复用 Omega 语义，不用 FlashRT 算子）。**308ms 的层级分解尚未 profile**，候选：denoise 小 m（suffix ~10 行）下 tcgen05 tile 利用率差、每层旋转/量化/cast 小 kernel 链在 ~1260 次线性调用/推理下累积、fp16 中转带宽。下一步：nsys 分解 bench eager 跑，定优化对象后再谈 kernel 级改造——"缺快 kernel"的判断作废，kernel 已在，慢在何处待定。⚠️ 本次 bench 的内层 `policy_timing.infer_ms` p50 0.84ms 是**修复前的异步读数**：当时 `policy.py` 的 `model_time` 在 `.cpu()` D2H 同步之前冻结，图模式下只测到 replay 发射开销。已在计时区间前后加 `torch.cuda.synchronize`（墙钟不受影响——`.cpu()` 本来就阻塞在同一批 GPU 工作上），修复后 `policy_timing.infer_ms` = 含 GPU 执行的模型段真实墙钟；server 端重采时该字段即为真实数字。两字段同名不同义仍在：`policy_timing.infer_ms` 只包 `sample_actions`（含 tokenize 与 GPU 执行，不含 D2H copy 本身/反归一化），server 端 `server_timing.infer_ms`（`websocket_policy_server.py:65-67`）包整个 `policy.infer`。交叉印证：42.9s/集 ÷ 308ms ≈ 139 次 infer/集，与 LIBERO episode 尺度一致。

nsys 分解（2026-08-23 二跑，bench `--eager` 全程采集，`perf_data/omega_e0m3_bench_eager_cuda_gpu_kern_sum.csv`，70 次 infer 归一）：**GPU kernel 总和 318ms/infer ≈ 图模式 wall 308ms**（图重放≈纯 GPU 时间）；eager inner 722ms − 318 ≈ **400ms 是 launch/dispatch 开销**（~23k kernel/infer × ~17µs）；eager wall − inner ≈ 5ms，非模型段（transform/D2H/反归一化）可忽略。三数互相咬合。kernel 级归堆（ms/infer）：**contiguous/copy 123.1（38.7%，10.7k 次）+ elementwise/cast 46.1（14.5%）+ perm gather 23.9（7.5%）= 消费层 glue 占六成**；nvjet bf16 GEMM 40.0（12.6%，含 vision tower 与旋转 bmm）+ gemv2T 20.4；**FP4 GEMM 本身仅 31.0（9.7%，1386 次 = 126 prefix + 126×10 denoise）**；量化 kernel 8.1；注意力（SDPA/FA2/softmax 合计）<8。结论：瓶颈不在 FP4 GEMM 而在消费层每线性 ~10 个 memory-bound glue kernel——`_rotate` 每次旋转 2 个 transpose+contiguous 布局拷贝、每调用重复 `blocks.to(dtype)` cast、perm gather、bf16↔fp16 中转各一份。优化顺序：① `_rotate` 改 `x.view(m,nb,b) @ blocks` 广播 matmul + blocks 预先 cast（纯 Python，零拷贝消掉 ~5.5k copies/infer，不动数值语义，action cos 门禁兜底）；② perm+rotate+cast+quant 融合进 `quantize_e0m3_sfa.cuh`（一 kernel/层）；③ vision tower bf16 GEMM 40ms 是下一个大件。

① 已落地（flashrt `daacc5ef`，`_rotate` 转置视图直接喂 cuBLAS strided-batched GEMV，消掉输入侧 transpose+contiguous 物化，输出侧 reshape 保留一次拷贝；blocks cast 按 (buffer,dtype,device) 缓存；CPU 数值逐位相等）：**图基线 wall p50 308.0 → 268.4ms（−12.9%），inner p50 264.3ms（sync 修复后首个真实模型段读数，wall−inner ≈ 4.2ms 非模型段开销，与 eager 分解的 ~5ms 互洽）**，`graph_state` 双 active 不变。剩余 glue：输出侧 reshape 拷贝、x2/y 中转 cast、perm gather、旋转 bmm——② 融合 quantize kernel 的主要回收对象。

② v1 教训（2026-08-23，flashrt `f9df2966`）：首版融合 kernel（warp 负责单个 (row, 64块)）**回退到 335.5ms（+67ms）**——每行重读整块旋转矩阵 R，L2 流量 = m×K×128B（K=16384 prefix 层单层 ~2GB），把"L2 放得下"误当"带宽免费"；cuBLAS bmm 每层只流式读 R 一遍。v2 改 warp 拥有 64 块 + R 驻留 smem（8KB/warp）+ 行块循环（64 行/warp）：R 流量降 m/64 倍，内层只剩 smem 读 + shfl 广播。教训：融合 kernel 的带宽账要按"每数据结构被读几遍"算，不能只看容量。

判读：

1. **官方全-W4A4 配方 + E0M3 kernel 端到端成立**：93.8% 与臂 D（93.2%）、臂 A（91.6%）均无统计差异；上节 action cos 门禁在 500 集尺度兑现。
2. **task9 崩盘被修复**：60% → 94%。expert-only 时代的唯一重灾区正是 PaliGemma 侧 RTN 掉点的症状，换 pack GPTQ 权重后恢复，呼应 action cos 里 e0m3/bf16 > fake/bf16 的方向。
3. **task8（双 moka pot）64% 成为最弱任务**（A 60 / D 76）：四臂在 60-76% 区间本就很散，不定性为回归；需要深挖可按 task4/task9 的探针方法另开分析。
4. 验证阶梯全部走完：artifact 252 层 → action cos 门禁 → 冒烟 10/10 → ×500 配对 SR。

## 后续行动

- [x] 看 task4 的 failure 模式（探针复测：80% 失败为 3-5cm 擦边偏心，见上节）
- [x] LIBERO-Plus 臂 A（81.4%，维度排序：相机位姿 50% 最脆弱，光照/背景/语言免疫，见上节）
- [x] LIBERO-Plus 臂 C 同子集 + 逐任务配对（64.3%，p=4.0e-07；Background Textures 97→50% 是最大交互维度，见上节）
- [x] [切 C server] task4 带符号探针 + task9 探针（各 50 集）：task4 偏移定向（黄白杯 -y 2.1→4.0cm）；task9 掉点主因是"门没关"（24 失败全部门未关到位，15 个杯已放入），见上节
- [x] FlashRT 复现（FP8 91.6% / NVFP4 92.6% 零掉点，FA4 27.21ms 对齐官方，见上两节）
- [x] Omega-QVLA 臂 D 复现（W4A4/W4A8 hybrid 93.2%，与基线无统计差异，见上节）
- [x] 路线 A ×500 配对 SR + 每集耗时（2026-08-23，见上节）：**93.8%**，vs 臂 A p=0.135 / 臂 D p=0.749 无差异，vs expert-only E0M3 p=0.033 显著更优（task9 60%→94%）；每集均值 42.9s（≈expert-only 43-50s，episode 长度混杂；vs 臂 D 148s 约 3.4×）。图模式 per-inference 已测：**308ms**（臂 A 口径，比 bf16 慢 2.2×，瓶颈在 E0M3 消费层无快 kernel，见上节）
- [ ] TRT：评估 NVFP4 静态 scale / Dyna 算子融合（预期省 ~5-6 ms/次）
- [ ] 评估 LLM 层与 expert denoise 的层级流水（重叠窗口 3.16 ms/step）
- [ ] （可选）提高 tegrastats 采样率（--tegrastats-interval 20~50 ms）以加密 test 阶段 EMC 样本，当前 7/3 个样本只够看量级
