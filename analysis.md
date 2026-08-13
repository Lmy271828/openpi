# π₀.5 性能分析报告（Jetson Thor，pi05_libero，w3/r10）

采集：`bash deployment_scripts/collect_perf_data.sh`（nsys `--gpu-metrics-devices=all` + tegrastats + trtexec 逐层 profile）
分析：`python deployment_scripts/analyze_perf.py --perf-dir perf_data --dram-peak-gbps 273`

> 注：本版修复了 analyze_perf.py 的 MemOps「次数」列（nsys CSV 列名是 `Count` 而非 `Instances`），并补拷了两个 `.sqlite`，因此新增「DRAM / GPU metrics」与「CUDA graph 使用情况」两节。

## num_steps 扫描（PyTorch）

| num_steps | mean (ms) | std (ms) | min | max |
|---|---|---|---|---|
| 1 | 74.71 | 0.86 | 74.18 | 77.22 |
| 2 | 81.35 | 0.43 | 80.98 | 82.53 |
| 4 | 94.14 | 0.16 | 93.86 | 94.36 |
| 6 | 106.93 | 0.36 | 106.34 | 107.39 |
| 8 | 119.58 | 0.37 | 119.10 | 120.52 |
| 10 | 131.94 | 0.39 | 131.39 | 132.56 |

线性拟合 `T(N) ≈ T_fixed + N·T_step`：
- **T_fixed（ViT + LLM prefill + host）= 68.58 ms**，占 T(10) 的 51.9%
- **T_step（单次 expert denoise）= 6.36 ms**，10 步合计 63.60 ms，占 T(10) 的 48.1%

## DRAM / 显存带宽（GPU metrics）

**pi05_ptcompile_pi05_libero_w3_r10**

nsys GPU metrics 在本设备（Tegra iGPU）未暴露 DRAM 计数器——DRAM 挂在 SoC 侧 MC/EMC，不归 GPU metrics 采样；以下为 Copy Engine 吞吐作为显存传输代理。真实 DRAM 带宽需用 tegrastats（EMC%）或 NCU `dram__*` 指标。

| 指标 | 单位 | 整体均值 | 整体峰值 | test 窗口均值 | test 窗口峰值 |
|---|---|---|---|---|---|
| Sync Copy Engine Active | Throughput % | 0.0（≈0 GB/s） | 0 | 0.0（≈0 GB/s） | 0 |
| Async Copy Engine Active 0 | Throughput % | 0.0（≈0 GB/s） | 100 | 0.0（≈0 GB/s） | 11 |
| Async Copy Engine Active 1 | Throughput % | 0.0（≈0 GB/s） | 60 | 0.1（≈0 GB/s） | 14 |

分阶段饱和度（S2/S3 NVTX 探针精确切窗，中位 test 窗口）：

| 指标 | 单位 | 阶段 | 窗口数 | 均值 | 峰值 |
|---|---|---|---|---|---|
| Copy Engine ×3 | Throughput % | prefill (S2) | 1 | 0.00 | 0 |
| Copy Engine ×3 | Throughput % | expert (S3) | 10 | 0.00 | 0 |
| SMs Active | Throughput % | prefill (S2) | 1 | 21.83 | 98 |
| SMs Active | Throughput % | expert (S3) | 10 | 33.51 | 99 |
| SM Issue | Throughput % | prefill (S2) | 1 | 6.33 | 28 |
| SM Issue | Throughput % | expert (S3) | 10 | 9.67 | 29 |
| Tensor Active | Throughput % | prefill (S2) | 1 | 6.83 | 38 |
| Tensor Active | Throughput % | expert (S3) | 10 | 0.14 | 1 |

**pi05_trt_fp8_nvfp4_pi05_libero_w3_r10**

| 指标 | 单位 | 整体均值 | 整体峰值 | test 窗口均值 | test 窗口峰值 |
|---|---|---|---|---|---|
| Sync Copy Engine Active | Throughput % | 0.0（≈0 GB/s） | 21 | 0.0（≈0 GB/s） | 0 |
| Async Copy Engine Active 0 | Throughput % | 0.0（≈0 GB/s） | 100 | 0.0（≈0 GB/s） | 9 |
| Async Copy Engine Active 1 | Throughput % | 0.0（≈0 GB/s） | 54 | 0.0（≈0 GB/s） | 5 |

TRT 采集不含 S2/S3 NVTX 探针（引擎内部不可标注），无法精确切分 prefill/expert。

## CUDA graph 使用情况（sqlite 直读）

- **pi05_ptcompile**（test_i 窗口 137.0 ms）：cudaGraphLaunch × **39**，普通 cudaLaunchKernel × 83，窗口内可见 kernel 仅 83 个 / 0.22 ms——稳态计算几乎全部在 CUDA graph replay 内（graph 内 kernel 不被此 nsys 版本归因）。
- **pi05_trt_fp8_nvfp4**（test_i 窗口 49.9 ms）：cudaGraphLaunch × **1**，普通 cudaLaunchKernel × 23，窗口内可见 kernel 仅 23 个 / 0.08 ms——同上，整次推理被 1 个 graph 覆盖。

## DRAM / EMC（tegrastats）

两份 `*_emc.log` 均零条 EMC/GR3D 字段（容器未挂 `/sys`，tegrastats 静默省略字段，见结论 4 与排查记录）；analyze_perf.py 该小节已就绪，重采后自动输出分阶段 EMC%/GR3D% 表。

## 稳态 MemOps（sqlite test 窗口）

**pi05_ptcompile**（test_i 窗口 137.0 ms）：H2D 10 次 / 0.02 ms / 0.5 MB；D2D 18 次 / 0.03 ms；D2H 13 次 / 0.04 ms。

**pi05_trt_fp8_nvfp4**（test_i 窗口 49.9 ms）：H2D 9 次 / 0.02 ms / 0.5 MB；D2D 9 次 / 0.01 ms / 0.9 MB；D2H 2 次 / 0.00 ms。

——单次稳态推理 memcpy 合计 ~0.1 ms、≤1 MB（H2D 0.5 MB ≈ 3 路 224×224×3 输入）。CSV 全窗口里 PyTorch 的 36.4 GB D2D 全部来自 warmup/compile。

## nsys kernel 分析 — pi05_ptcompile（全采集窗口，含 warmup）

GPU kernel 总耗时 11283.40 ms（**绝大部分在 warmup/compile 阶段**，稳态见 CUDA graph 一节）。

| 类别 | 总耗时 (ms) | 占比 | launch 次数 |
|---|---|---|---|
| Elementwise/reduction (triton) | 11001.85 | 97.5% | 105831 |
| Softmax | 92.35 | 0.8% | 1764 |
| Norm (layer/rms) | 90.95 | 0.8% | 6425 |
| GEMM (cuBLAS/cutlass/nvjet) | 87.73 | 0.8% | 2587 |
| Attention (fmha/sdpa/flash) | 9.06 | 0.1% | 985 |
| Quant/convert (FP8/FP4/cvt) | 1.45 | 0.0% | 364 |
| Other | 0.01 | 0.0% | 4 |

Top-5 kernel：

| 总耗时 (ms) | 次数 | kernel |
|---|---|---|
| 9624.58 | 74072 | vectorized_elementwise_kernel<4, FillFunctor<int>, ...> |
| 253.40 | 1790 | triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_8 |
| 172.63 | 1493 | triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_7 |
| 128.35 | 1238 | triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_11 |
| 123.20 | 1138 | triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_10 |

MemOps（全窗口）：D2D 402.70 ms × 11153（合计 36.4 GB）；H2D 72.09 ms × 1027（8.9 GB）；D2H 1.55 ms × 206。

## nsys kernel 分析 — pi05_trt_fp8_nvfp4（全采集窗口，含 warmup）

GPU kernel 总耗时 51.16 ms（engine 启动期的直接 kernel launch；稳态在 graph 内）。

| 类别 | 总耗时 (ms) | 占比 | launch 次数 |
|---|---|---|---|
| GEMM (cuBLAS/cutlass/nvjet) | 35.16 | 68.7% | 1752 |
| Quant/convert (FP8/FP4/cvt) | 7.81 | 15.3% | 1508 |
| Other | 7.40 | 14.5% | 543 |
| Elementwise/reduction (triton) | 0.76 | 1.5% | 221 |
| Norm (layer/rms) | 0.02 | 0.0% | 13 |

MemOps（全窗口）：H2D 82.84 ms × 934（大头为 engine 反序列化/权重上传，非稳态）；memset 12.94 ms × 2。

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

**1. 端到端：TRT 比 PyTorch(torch.compile) 快 2.74×。** sqlite 直读的稳态 test 窗口：TRT 49.9 ms vs PyTorch 137.0 ms。两后端稳态计算都完整包在 CUDA graph 内（TRT 整次推理仅 1 次 cudaGraphLaunch；PyTorch 39 次），此前 kern_sum 的 11283 ms 几乎全是 warmup/compile 痕迹，不能用于对比。
注意：本次 ptcompile 窗口被 NVTX 探针轻微污染——`_make_denoise_step` 的可变全局计数器导致 dynamo 每步重编译、触发 recompile_limit(8) 后 idx≥8 的步回退 eager（console.log 有记录）；137.0 ms 比干净的 sweep T(10)=131.9 ms 高 ~4% 即源于此。已在 `pi05_inference_nvtx.py` 给探针加 `@torch._dynamo.disable` 修复，重采后该窗口应与 sweep 对齐。

**2. PyTorch 侧 kernel 总量是 warmup 假象。** 85%（9624 ms / 74072 次）是 `FillFunctor<int>`——int32 填充，来自 compile/warmup 阶段反复建 mask / position id，稳态窗口内不存在（稳态可见 kernel 仅 0.22 ms）。对比分析应完全基于 test 窗口数据。

**3. PyTorch expert 阶段不吃 Tensor Core。** S2/S3 精确切窗显示：expert (S3) 的 Tensor Active 均值仅 **0.14%**（prefill 6.83%），而 SMs Active 33.5%、SM Issue 9.7%——expert 的 GEMM 太小（action expert hidden 维度小、token 数少），在 PyTorch 下走的是 CUDA core / 受 launch 与带宽限制。这解释了 TRT 的 FP8 tensorop GEMM 为何能在 expert 上拿到 ~2×（6.36 → 3.16 ms/step），也说明 PyTorch 侧继续优化 expert 的收益上限有限。

**4. 两阶段都远未打满。** prefill SM Active 21.8% / expert 33.5%，SM Issue <10%，Copy Engine ≈0——无论计算还是拷贝引擎都有余量，瓶颈在 kernel 粒度与串行依赖，不在硬件吞吐。真正的 DRAM 带宽占用 nsys 测不到（Tegra iGPU 无 DRAM 计数器），需解析 `*_emc.log`（tegrastats EMC%）补齐——目前 analyze_perf.py 尚未消费该文件。

**5. TRT 内部可再挖 ~10%。** NVFP4 动态量化（`*Dyna*`）69 个算子共 6.23 ms（10.2%），是激活动态 scale 计算；改校准后静态 scale 或与相邻算子融合可回收大部分。阶段占比：expert x10 = 51.7%、LLM prefill = 37.8%、ViT = 10.5%；层级流水重叠窗口 3.16 ms/step，10 步理论上限 ~31 ms（约为总时延一半），但需先断开 LLM 层间与 expert 的依赖。

**6. 访存流量只是 warmup 假象（已由 sqlite 切窗证实）。** 全窗口 CSV 里 PyTorch 有 D2D 36.4 GB / 402.7 ms、H2D 8.9 GB，但按 test 窗口过滤后，单次稳态推理的 memcpy 合计仅 **~0.1 ms / ≤1 MB**（H2D 0.5 MB ≈ 3 路 224×224×3 输入图像，符合预期；D2D/D2H 为微量 glue）。两个后端皆然——稳态访存不是优化对象，GPU 时间几乎全在计算上。

## 后续行动

- [ ] **重采 tegrastats**（根因已查明：容器只挂了 tegrastats 二进制、没挂 `/sys`，tegrastats 读不到 sysfs 节点时静默省略 EMC/GR3D 字段）。`docker run` 加 `-v /sys:/sys:ro`（handbook Step 3 已更新）后重跑 `collect_perf_data.sh`；analyze_perf.py 的「DRAM / EMC（tegrastats）」小节已就绪，会自动按 phase 对齐输出
- [ ] TRT：评估 NVFP4 静态 scale / Dyna 算子融合（预期省 ~5-6 ms/次）
- [ ] 评估 LLM 层与 expert denoise 的层级流水（重叠窗口 3.16 ms/step）
