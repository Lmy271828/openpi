## num_steps 扫描（PyTorch）

| num_steps | mean (ms) | std (ms) | min | max |
|---|---|---|---|---|
| 1 | 74.82 | 0.87 | 74.21 | 77.29 |
| 2 | 80.42 | 0.27 | 80.05 | 81.09 |
| 4 | 92.99 | 0.31 | 92.57 | 93.64 |
| 6 | 106.04 | 0.27 | 105.63 | 106.41 |
| 8 | 118.79 | 0.52 | 118.10 | 120.04 |
| 10 | 130.27 | 0.50 | 129.36 | 131.14 |

线性拟合 `T(N) ≈ T_fixed + N·T_step`：
- **T_fixed（ViT + LLM prefill + host）= 68.31 ms**，占 T(10) 的 52.3%
- **T_step（单次 expert denoise）= 6.24 ms**，10 步合计 62.41 ms，占 T(10) 的 47.7%

## CUDA graph 使用情况（sqlite 直读）

**PyTorch torch.compile**（test_i 窗口 135.0 ms）：cudaGraphLaunch × **39**，普通 cudaLaunchKernel × 83，窗口内可见 kernel 仅 83 个 / 0.24 ms——稳态计算几乎全部在 CUDA graph replay 内（graph 内 kernel 不被此 nsys 版本归因，per-kernel 分析需用 Eager 对照组或 TRT 逐层 profile）

**TensorRT**（test_i 窗口 49.6 ms）：cudaGraphLaunch × **1**，普通 cudaLaunchKernel × 23，窗口内可见 kernel 仅 23 个 / 0.08 ms——稳态计算几乎全部在 CUDA graph replay 内（graph 内 kernel 不被此 nsys 版本归因，per-kernel 分析需用 Eager 对照组或 TRT 逐层 profile）

## nsys kernel 分析 — PyTorch Eager（对照组）

**PyTorch Eager** — GPU kernel 总耗时 2270.27 ms（含 warmup，归一化见报告正文）

| 类别 | 总耗时 (ms) | 占比 | launch 次数 |
|---|---|---|---|
| GEMM (cuBLAS/cutlass/nvjet) | 1137.17 | 50.1% | 31018 |
| Elementwise/reduction (triton) | 492.43 | 21.7% | 86437 |
| Quant/convert (FP8/FP4/cvt) | 288.89 | 12.7% | 46722 |
| Other | 258.12 | 11.4% | 5642 |
| Softmax | 71.25 | 3.1% | 2574 |
| Norm (layer/rms) | 22.42 | 1.0% | 2158 |

Top-15 kernel：

| 总耗时 (ms) | 次数 | kernel |
|---|---|---|
| 257.24 | 5070 | void gemv2T_kernel_val<int, int, float, float, float, float, (int)128, (int)16, (int)4, (int)4, (bool)0, (bool |
| 213.38 | 702 | nvjet_sm110_tst_256x200_64x3_2x1_v_bz_TNT |
| 200.22 | 468 | nvjet_sm110_tst_256x248_64x3_1x2_h_bz_TNT |
| 178.56 | 4680 | nvjet_sm110_tst_256x16_64x6_4x1_2cta_v_bz_TNT |
| 93.56 | 7722 | void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, |
| 81.76 | 2340 | nvjet_sm110_tst_192x16_64x8_2x1_2cta_v_bz_splitK_TNT |
| 80.56 | 8450 | void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[l |
| 72.56 | 1053 | nvjet_sm110_tst_256x128_64x4_1x2_h_bz_splitK_bias_TNT |
| 71.25 | 2574 | void <unnamed>::softmax_warp_forward<float, float, float, (int)10, (bool)0, (bool)0, (int)32>(T2 *, const T1 * |
| 67.59 | 3627 | void at::native::vectorized_elementwise_kernel<(int)4, at::native::GeluCUDAKernelImpl(at::TensorIteratorBase & |
| 67.05 | 4212 | nvjet_sm110_tst_128x128_64x6_1x2_h_bz_bias_TNT |
| 60.53 | 2340 | void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_bf16_64x64_32x6_tn_align2>(T1::Params) |
| 59.46 | 13273 | void at::native::vectorized_elementwise_kernel<(int)4, at::native::bfloat16_copy_kernel_cuda(at::TensorIterato |
| 57.15 | 1053 | nvjet_sm110_tst_224x128_64x7_1x2_2cta_h_bz_bias_TNN |
| 56.68 | 5148 | void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::CUDA |

MemOps：

| 操作 | 总耗时 (ms) | 次数 |
|---|---|---|
| [CUDA memcpy Host-to-Device] | 59.80 | None |
| [CUDA memcpy Device-to-Host] | 0.33 | None |
| [CUDA memcpy Device-to-Device] | 0.31 | None |
| [CUDA memset] | 0.00 | None |

## trtexec per-layer profile（TensorRT 阶段归因）

来源：`model_fp8_nvfp4.engine_profile.json`，layer 记录 3847 条，合计 61.74 ms（trtexec 逐层 profile 含层间同步开销，绝对值约为实际推理的 1.3 倍，看占比）

| 阶段 | 耗时 (ms) | 占比 | layer 实例数 |
|---|---|---|---|
| Action expert x10 denoise (FP8) | 32.18 | 52.1% | 3095 |
| LLM prefill (Gemma 2B, NVFP4) | 22.95 | 37.2% | 451 |
| ViT (SigLIP x3 views) | 6.61 | 10.7% | 301 |

推算：expert 每步 ≈ 3.22 ms；LLM 每层 ≈ 1.27 ms；层级流水重叠窗口（§7）= min(每步, LLM 层2-18) ≈ **3.22 ms**。

NVFP4 动态量化开销（`*Dyna*` 类算子）：69 个，合计 6.11 ms（9.9%）。

Top-20 layer（按耗时）：

| 耗时 (ms) | 占比 | layer |
|---|---|---|
| 0.284 | 0.5% | __myl_MulReshDyna_myl0_485 |
| 0.283 | 0.5% | __myl_MulReshDyna_myl0_535 |
| 0.283 | 0.5% | __myl_MulReshDyna_myl0_585 |
| 0.282 | 0.5% | __myl_MulReshDyna_myl0_460 |
| 0.282 | 0.5% | __myl_MulReshDyna_myl0_335 |
| 0.282 | 0.5% | __myl_MulReshDyna_myl0_660 |
| 0.281 | 0.5% | __myl_MulReshDyna_myl0_560 |
| 0.281 | 0.5% | __myl_MulReshDyna_myl0_410 |
| 0.281 | 0.5% | __myl_MulReshDyna_myl0_635 |
| 0.280 | 0.5% | __myl_MulReshDyna_myl0_610 |
| 0.280 | 0.5% | __myl_MulReshDyna_myl0_435 |
| 0.280 | 0.5% | __myl_MulReshDyna_myl0_385 |
| 0.280 | 0.5% | __myl_MulReshDyna_myl0_360 |
| 0.279 | 0.5% | __myl_MulReshDyna_myl0_510 |
| 0.279 | 0.5% | __myl_MulReshDyna_myl0_735 |
| 0.279 | 0.5% | __myl_MulReshDyna_myl0_710 |
| 0.278 | 0.5% | __myl_MulReshDyna_myl0_685 |
| 0.258 | 0.4% | /layers.9/mlp/gate_proj/MatMul_myl0_559 |
| 0.254 | 0.4% | /layers.8/mlp/up_proj/MatMul_myl0_532 |
| 0.244 | 0.4% | /layers.5/mlp/gate_proj/MatMul_myl0_459 |
