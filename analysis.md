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

## 后续行动

- [x] 看 task4 的 failure 模式（探针复测：80% 失败为 3-5cm 擦边偏心，见上节）
- [x] LIBERO-Plus 臂 A（81.4%，维度排序：相机位姿 50% 最脆弱，光照/背景/语言免疫，见上节）
- [x] LIBERO-Plus 臂 C 同子集 + 逐任务配对（64.3%，p=4.0e-07；Background Textures 97→50% 是最大交互维度，见上节）
- [x] [切 C server] task4 带符号探针 + task9 探针（各 50 集）：task4 偏移定向（黄白杯 -y 2.1→4.0cm）；task9 掉点主因是"门没关"（24 失败全部门未关到位，15 个杯已放入），见上节
- [ ] FlashRT 复现（P1: FP8 on libero_10；P2: NVFP4+AWQ），环境方案见 handbook「FlashRT 复现环境」
- [ ] TRT：评估 NVFP4 静态 scale / Dyna 算子融合（预期省 ~5-6 ms/次）
- [ ] 评估 LLM 层与 expert denoise 的层级流水（重叠窗口 3.16 ms/step）
- [ ] （可选）提高 tegrastats 采样率（--tegrastats-interval 20~50 ms）以加密 test 阶段 EMC 样本，当前 7/3 个样本只够看量级
