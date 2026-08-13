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

**1. 端到端：TRT 比 PyTorch(torch.compile) 快 2.75×。** sqlite 直读的稳态 test 窗口：TRT 49.9 ms vs PyTorch 137.2 ms（sweep 干净值 T(10)=131.0 ms；窗口含 NVTX 探针与 graph replay 开销，略高 ~5% 属预期）。两后端稳态计算都完整包在 CUDA graph 内（TRT 整次推理仅 1 次 cudaGraphLaunch；PyTorch 39 次），kern_sum 的 10949 ms 几乎全是 warmup/compile 痕迹，不能用于对比。
测量修复记录（供追溯）：① NVTX 探针的可变全局计数器使 dynamo 每个 denoise 步重编译，默认 `recompile_limit=8` 耗尽后 idx≥8 的步回退 eager（窗口虚高 ~4%）；② 曾误用 `@torch._dynamo.disable` 修复——graph break 把 kernel 挤出 CUDA graph（graphLaunch 39→9，裸露 kernel 83→11142，窗口恶化到 274 ms），该轮作废；③ 正确修复：`torch._dynamo.config.recompile_limit=64`，10 个步数变体 warmup 全部编译完，稳态既有逐步 NVTX 标签又保住 graph（本轮已验证：39 次 graphLaunch 恢复）。

**2. PyTorch 侧 kernel 总量是 warmup 假象。** 85%（9315 ms / 71912 次）是 `FillFunctor<int>`——int32 填充，来自 compile/warmup 阶段反复建 mask / position id，稳态窗口内不存在（稳态可见 kernel 仅 0.24 ms）。对比分析应完全基于 test 窗口数据。

**3. PyTorch expert 阶段不吃 Tensor Core。** S2/S3 精确切窗：expert (S3) 的 Tensor Active 均值仅 **0.13%**（prefill 6.57%），而 SMs Active 36.6%、SM Issue 7.0%——expert 的 GEMM 太小（action expert hidden 维度小、token 数少），在 PyTorch 下走 CUDA core / 受 launch 与带宽限制。这解释了 TRT 的 FP8 tensorop GEMM 为何能在 expert 上拿到 ~2×（6.13 → 3.16 ms/step），也说明 PyTorch 侧继续优化 expert 的收益上限有限。

**4. 算力、带宽、拷贝引擎全部远未打满。** prefill SM Active 24.1% / expert 36.6%，SM Issue <10%，Copy Engine ≈0；DRAM 侧（EMC 分阶段数据）：**PyTorch 推理段 EMC 均值 43.1% ≈ 118 GB/s，TRT 推理段 24% ≈ 66 GB/s**——PyTorch 的带宽占用约为 TRT 的 1.8×，与 BF16 vs FP8/NVFP4 权重流量的比例一致；但即便 PyTorch 也距 273 GB/s 峰值有一倍以上余量。**瓶颈在 kernel 粒度与串行依赖，不在硬件吞吐**；"加带宽/换更快显存"类优化对本案无效。

**5. TRT 内部可再挖 ~10%。** NVFP4 动态量化（`*Dyna*`）69 个算子共 6.23 ms（10.2%），是激活动态 scale 计算；改校准后静态 scale 或与相邻算子融合可回收大部分。阶段占比：expert x10 = 51.7%、LLM prefill = 37.8%、ViT = 10.5%；层级流水重叠窗口 3.16 ms/step，10 步理论上限 ~31 ms（约为总时延一半），但需先断开 LLM 层间与 expert 的依赖。

**6. 访存流量只是 warmup 假象（已由 sqlite 切窗证实）。** 全窗口 CSV 里 PyTorch 有 D2D 452.8 ms × 11270、H2D 70.4 ms，但按 test 窗口过滤后，单次稳态推理的 memcpy 合计仅 **~0.1 ms / ≤1 MB**（H2D 0.5 MB ≈ 3 路 224×224×3 输入图像，符合预期）。两个后端皆然——稳态访存不是优化对象，GPU 时间几乎全在计算上。

## 后续行动

- [ ] TRT：评估 NVFP4 静态 scale / Dyna 算子融合（预期省 ~5-6 ms/次）
- [ ] 评估 LLM 层与 expert denoise 的层级流水（重叠窗口 3.16 ms/step）
- [ ] （可选）提高 tegrastats 采样率（--tegrastats-interval 20~50 ms）以加密 test 阶段 EMC 样本，当前 7/3 个样本只够看量级
