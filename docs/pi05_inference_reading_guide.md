# pi0.5 推理优化部署：openpi 项目阅读指南

> 面向目标：对 pi0.5 进行软硬件协同推理优化部署（latency / throughput 优化、量化、kernel 优化、部署链路改造）。

## 0. 项目定位

在本仓库（openpi）中，**pi0.5 不是独立文件**，而是 `pi0.py` 中的一个开关：

- `src/openpi/models/pi0_config.py:31` — `pi05: bool = False`
- pi05 相对 pi0 的关键差异：
  - action expert 使用 adaRMS 归一化（`use_adarms=[False, True]`）
  - 状态输入采用离散化（`discrete_state_input`）
  - 最大 token 长度从 48 提升到 200

做推理优化部署，阅读主线是：**数据怎么进来 → 模型怎么算 → 结果怎么送出去**，训练代码基本可跳过。

## 1. 第一层：推理部署骨架（先读，建立全局观）

| 顺序 | 文件 | 关注点 |
|---|---|---|
| 1 | `scripts/serve_policy.py` | 部署入口。看 `create_policy()` 如何组装模型、数据变换和 checkpoint，`main()` 如何起 websocket server |
| 2 | `src/openpi/serving/websocket_policy_server.py` | 网络层（很轻）；客户端/服务端协议在 `packages/openpi-client` |
| 3 | `src/openpi/policies/policy.py` | 核心封装类 `Policy`，`infer()` 是**每次推理的完整热路径**：输入变换 → 模型采样 → 输出变换，优化主战场 |
| 4 | `examples/simple_client/`、`examples/libero/` | 客户端侧如何发观测、收 action chunk |

## 2. 第二层：模型本体（pi0.5 的计算图）

| 顺序 | 文件 | 关注点 |
|---|---|---|
| 5 | `src/openpi/models/pi0.py` | 主模型类。重点：`sample_actions()`（flow matching 的多步 ODE 积分循环，推理延迟大头）、`embed_prefix()`、KV cache。`pi05` 分支见 `pi0.py:93,151,162` |
| 6 | `src/openpi/models/gemma.py`、`src/openpi/models/siglip.py` | 骨干网络：SigLIP ViT 做图像编码，Gemma 是 LLM + action expert 双 expert 结构 |
| 7 | `src/openpi/models/pi0_config.py` | 全部超参：action horizon、denoising steps、token 长度——latency/吞吐权衡直接对应这些旋钮 |

## 3. 第三层：两条后端的取舍（最关键）

- **JAX/Flax 版**（`src/openpi/models/`）：官方主线，训练也用它，依赖 `flax.nnx` + `jax`。
- **PyTorch 版**（`src/openpi/models_pytorch/`）：`pi0_pytorch.py`、`gemma_pytorch.py`，基于 HuggingFace transformers 重写，对部署更友好（CUDA、TensorRT、量化生态）。注意其中的 `transformers_replace/` 补丁目录，读之前先看它替换了什么。
- 权重转换：`examples/convert_jax_model_to_pytorch.py`。

**部署建议**：如果目标是 GPU 服务器 + 低延迟优化（KV cache、量化、kernel 融合），PyTorch 版通常是更好的起点；JAX 版适合对照验证数值正确性。

## 4. 第四层：外围（按需查阅，不必通读）

- `src/openpi/transforms.py` — 归一化/反归一化、图像 resize 等前后处理；部署时确认这些在 CPU 还是 GPU 上执行
- `docs/remote_inference.md` — 官方远程推理部署文档，**必读**
- `scripts/compute_norm_stats.py` — 推理反归一化依赖的 norm stats
- `src/openpi/training/config.py` — 只需看 pi05 相关的 config 定义（如 `pi05_libero`），确认待部署模型的参数配置

## 5. 高效实操顺序

1. 先跑通官方远程推理 demo（`serve_policy.py` + libero client），确认环境可用；
2. 在 `Policy.infer()` 里打 profiling 点，拆出各阶段耗时占比：
   - ViT 图像编码
   - Gemma prefix 前向
   - action expert 每步 denoise 迭代
   - 数据前后处理
   —— 这决定优化收益空间（通常 action expert 迭代 loop 与图像编码是大头）；
3. 带着 profiling 结果回头精读 `pi0.py` 的 `sample_actions()` 及对应后端实现。
