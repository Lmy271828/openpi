# openpi π0.5 Jetson Thor 部署与量化修正研究（中文说明）

> 本文件是研究侧说明，不替代英文原版 [README.md](README.md)。
> 代码基线：Physical Intelligence openpi（PyTorch 分支），部署目标为 NVIDIA Jetson Thor（JetPack 7.2，sm_110）。

## 1. 研究背景

π0.5 是 flow-matching VLA 模型，标准推理需要 10 步去噪。在 Jetson Thor 上我们构建了三条部署臂做归因实验：

| 臂 | 方案 | libero_10 (500 trials) | 端到端延迟 |
|---|---|---|---|
| A | PyTorch BF16 + torch.compile | 91.6% | 137 ms |
| B | TensorRT FP16 | 93.0% | 85.8 ms |
| C | TensorRT FP8 + NVFP4 | 80.0% | 49.9 ms |

McNemar 配对检验显示 B≈A（p=0.41），C 相对 A 的 −11.6% 完全由量化造成。LIBERO-Plus 210 任务上 C 相对 A 掉 17.1%（纹理维度 97%→50% 强交互）。探针分析表明量化的失败模式是**动作落点的定向漂移**（off-manifold 偏移），而非随机抖动。

## 2. 核心思路：量化主干 + 高精度外挂 LoRA 修正器（MIP 式两步推理）

### 2.1 动机

《Much Ado About Noising》的核心结论：扩散/流模型的多步迭代推理（MIP, Manifold-constrained Iterative Process）之所以提升成功率，不是 L2/cosine 意义上动作更"准"，而是第二步把第一步的输出**吸附回动作流形**（manifold adherence）。传统 L2 代理指标与成功率不相关，off-manifold 残差才是有效代理。

由此得到一个工程假设：**量化误差本质上是一种 off-manifold 的结构化扰动**。既然 MIP 第二步能修正首步的几何偏移，那么可以用一个小的高精度修正器专门承担"拉回流形"的职责，而主干保持 FP8/NVFP4 全量化以保住延迟收益。

### 2.2 方案

- **第一步（量化主干）**：FP8/NVFP4 引擎在 t=0.4 做免训练单跳（Δt=1），快速给出动作草案 â₀。
- **第二步（高精度修正器）**：按 MIP 推理形式构造输入 `x = t* · â₀`（t*=0.9，z=0），送入**冻结量化主干 + bf16 外挂 LoRA**（仅第二步生效），输出修正后的动作。
- LoRA 秩很小（如 r=8~16），只挂在 attention/MLP 投影上，第二步引入的额外开销控制在 1~2 ms。

### 2.3 Loss 设计（借鉴 Much Ado 的 off-manifold L2）

不直接对动作做 L2，而是惩罚**流形外分量**：

```
L = || (I − U Uᵀ) · (a_pred − a_gt) ||²   +   λ · || U Uᵀ · (a_pred − a_gt) ||²
```

- `U ∈ R^{d×k}` 是动作流形切空间的正交基，**由校准数据离线预计算**：收集专家动作 chunk（如 LIBERO 校准集的 10×32 动作序列），展平后对协方差矩阵做 SVD/PCA，取前 k 个主奇异向量（k 由能量占比 99% 截断，典型 k ≪ d=320）。
- `(I − U Uᵀ) a` 即动作的 off-manifold 分量，是 Much Ado 论文验证过与成功率相关的代理方向；λ ≪ 1 让优化压力集中在流形外残差上。
- 训练时冻结全部主干权重（含量化 scale），只更新 LoRA；第二步输入按 `t*·â₀ + 噪声扰动` 做数据增强，覆盖量化主干实际输出的偏移分布。

### 2.4 免训练前奏：自定义采样时间网格

在训练 LoRA 之前，先用纯调度实验验证"两步 MIP 结构"本身在 π0.5 上的收益（Much Ado Table 19 显示 NFE 敏感性是架构依赖的，必须实测）。已实现通过环境变量注入任意时间网格：

```bash
# 格式: t_eval:dt[:input_scale];...   （openpi 约定 t=1 噪声 → t=0 动作）
export PI05_T_GRID="0.6:-1.0;0.1:-0.1:0.9"   # 本文调度：t=0.4 单跳 + t=0.9 修正步
export PI05_T_GRID="1.0:-1.0"                # 对照 N=1
export PI05_T_GRID="1.0:-0.5;0.5:-0.5"       # 对照 N=2 均匀
# 不设置 = 默认 N=10 基线
```

实现位于 `src/openpi/models_pytorch/pi0_pytorch.py` 的 `sample_actions`，与 torch.compile 兼容（换网格需重启 server 触发 guard 重编译）。

## 3. Thor 侧换臂操作（以 armA + 自定义调度为例）

在 Thor（10.191.163.226）`~/lmy/openpi` 下：

```bash
# 1. 拉取新代码（含 PI05_T_GRID 支持）
cd ~/lmy/openpi
git pull --recurse-submodules
# 若 GitHub 直连不稳，从本地 rsync 单文件：
# rsync -avz ~/pynoob/openpi/src/openpi/models_pytorch/pi0_pytorch.py \
#   hcclab@10.191.163.226:~/lmy/openpi/src/openpi/models_pytorch/pi0_pytorch.py

# 2. 停掉当前 server
sudo docker stop pi05_server && sudo docker rm pi05_server

# 3. 起 armA（PyTorch BF16，带自定义调度）
sudo docker run -d --name pi05_server --runtime nvidia \
  --cap-add SYS_ADMIN \
  --network host \
  -v "$PWD":/workspace \
  -v "$HOME/.cache/openpi":/root/.cache/openpi \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -v /usr/bin/tegrastats:/usr/bin/tegrastats:ro \
  -v /sys:/sys:ro \
  -w /workspace \
  openpi-pi0.5:l4t-jp7.2 \
  bash -c "TF_DIR=/usr/local/lib/python3.12/dist-packages/transformers && \
           cp -r src/openpi/models_pytorch/transformers_replace/* \$TF_DIR/ && \
           export PYTHONPATH=packages/openpi-client/src:src:. && \
           export PI05_T_GRID='0.6:-1.0;0.1:-0.1:0.9' && \
           python scripts/serve_policy.py --port 8000 policy:checkpoint \
             --policy.config=pi05_libero \
             --policy.dir=/root/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"

# 4. 确认就绪（看到 server listening on 0.0.0.0:8000）
sudo docker logs -f pi05_server
```

换臂只需替换 `--policy.dir` 指向对应 checkpoint；跑对照档只需改 `PI05_T_GRID` 字符串并重启容器。

本地 client（LIBERO 评测）命令见 [handbook.md](handbook.md)，输出目录建议按调度档位命名（如 `eval_out/sched_2step_armA`）。

## 4. FlashRT 环境构建（Thor 原生，SM110）

开发模式为本地 `git push` → Thor `git pull --recurse-submodules`，但以下内容**不走 git**，需在 Thor 侧一次性就位：

- `third_party/flashrt/third_party/cutlass`（被 flashrt 自身 `.gitignore` 排除）
- `third_party/flashrt/.venv`、`build/`（必须 Thor 本地生成）

环境配置脚本 `deployment_scripts/thor_jp72_env.sh` 在主仓库内，随 git 走。

```bash
# Thor 侧：CUTLASS 钉版依赖（二选一）
cd ~/lmy/openpi/third_party/flashrt
git clone --depth 1 --branch v4.4.2 https://github.com/NVIDIA/cutlass.git third_party/cutlass
# 或本地 clone 后 rsync（在本地 third_party/flashrt 下执行）：
# rsync -avz third_party/cutlass hcclab@10.191.163.226:~/lmy/openpi/third_party/flashrt/third_party/

# Thor 侧：恢复环境（激活 venv + LD_LIBRARY_PATH + LIBERO 评测变量，含 nvpl/cu13 缺失库修复记录）
source ~/lmy/openpi/deployment_scripts/thor_jp72_env.sh

# 构建
pip install pybind11 cmake "numpy>=1.24" safetensors "transformers<4.56" pandas pillow pyarrow
pip install -e ".[torch]"
cmake -B build -S . -DGPU_ARCH=110 && cmake --build build -j$(nproc)
bash scripts/download_paligemma_tokenizer.sh
```

实测环境：torch 2.11.0（cu130 sbsa wheel，`pip install torch --index-url https://pypi.jetson-ai-lab.io/sbsa/cu130`），capability `(11, 0)`。wheel 为 `--no-deps` 风格，缺库修复（nvpl-lapack / libcudss 路径）详见 `deployment_scripts/thor_jp72_env.sh` 注释。

## 5. LIBERO 评测

**臂 A/B/C（openpi，server/client 分离）**：见 [handbook.md](handbook.md)。

**FlashRT 复现（Thor 同进程：模型 + 仿真同一 venv）**，完整流程：

```bash
# ===== 本地 x86 侧（一次性）：同步 libero 仓库到 Thor =====
rsync -avz --exclude=.git ~/pynoob/openpi/third_party/libero hcclab@10.191.163.226:~/lmy/openpi/third_party/

# ===== Thor 侧 =====
cd ~/lmy/openpi
git pull --recurse-submodules          # 拿到 deployment_scripts/thor_jp72_env.sh
source deployment_scripts/thor_jp72_env.sh   # 激活 venv + LD_LIBRARY_PATH + PYTHONPATH(libero) + TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD
cd third_party/flashrt

# 验证环境
python -c "import flash_rt, torch; print(flash_rt.__version__, torch.__version__, torch.cuda.is_available())"

# tokenizer（已存在则跳过）
ls ~/.cache/openpi/big_vision/paligemma_tokenizer.model 2>/dev/null || bash scripts/download_paligemma_tokenizer.sh

# 仿真最小依赖（版本与 x86 client 侧对齐；libero 本体走 PYTHONPATH，不 pip 安装）
pip install "robosuite==1.4.1" "mujoco==3.2.3" bddl easydict gym matplotlib opencv-python-headless PyOpenGL
pip install ml_dtypes tqdm   # 上游 [torch] extra 漏装：pipeline_rtx.py 无条件 import ml_dtypes
python -c "from libero.libero import benchmark; print('libero ok')"

# LIBERO 路径配置（注意：init 目录名是 init_files，层级为 libero/libero/libero）
cat > ~/.libero/config.yaml <<'EOF'
assets: /home/hcclab/lmy/openpi/third_party/libero/libero/libero/./assets
bddl_files: /home/hcclab/lmy/openpi/third_party/libero/libero/libero/./bddl_files
benchmark_root: /home/hcclab/lmy/openpi/third_party/libero/libero/libero
datasets: /home/hcclab/lmy/openpi/third_party/libero/libero/libero/../datasets
init_states: /home/hcclab/lmy/openpi/third_party/libero/libero/libero/./init_files
EOF

# 冒烟（3 任务 x 3 episodes）
python examples/thor/eval_libero.py \
  --checkpoint ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch \
  --task_suite libero_10 --quick

# 全量 FP8（10 任务 x 50 episodes；对照臂 C = 80.0%，官方参考 92.6-93%）
setsid nohup python examples/thor/eval_libero.py \
  --checkpoint ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch \
  --task_suite libero_10 \
  > eval_flashrt_fp8_libero10.log 2>&1 < /dev/null &

# 查进度
tail -5 eval_flashrt_fp8_libero10.log

# FP8 跑完后再跑 NVFP4 变体（两个全量不要同时跑，避免抢 GPU）
setsid nohup python examples/thor/eval_libero.py \
  --checkpoint ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch \
  --task_suite libero_10 --use_fp4 \
  > eval_flashrt_nvfp4_libero10.log 2>&1 < /dev/null &
```

结果写入 `libero_libero_10_torch_results.json`（逐任务成功率 + P50 延迟）。max steps 口径与 openpi client 一致（libero_10 = 520 步/episode），成功率可直接对比。

## 5.5 FlashRT 代码阅读路径（pi0.5 部分）

面向有 C++ 基础、没写过 CUDA C 的开发者。原则：**从 Python 调用侧往 kernel 侧下沉，每一层只读"它和上下层的接口"，GEMM 内部当黑盒**。四个阶段，每个阶段读完能回答对应的问题就算过：

**阶段 1：Python 前端（纯 Python，无 CUDA）**

- `flash_rt/models/pi05/pipeline_thor.py`（929 行）——Thor 侧 π0.5 推理管线：模型怎么拼装、校准 scale 从哪加载、每个手写 kernel 在 forward 的哪个位置被调用。这是全项目的"地图"。
- `examples/thor/eval_libero.py` + `tests/bench_pi05_thor_views.py`——管线的两个消费方，看它们怎么传参（`use_fp4` 怎么走通）就知道 API 边界在哪。
- 读完能回答：一次 denoise step 依次调用了哪些 kernel？FP8/NVFP4 的分支在哪切换？

**阶段 2：第一个 kernel（小、独立、无模板）**

- `csrc/fused_fp4/siglip_ln_vec.cu`（269 行）——SigLIP 的向量化 LayerNorm，项目里最短的手写 kernel 之一。对照 `csrc/fp4_bindings.cpp` 里它的 pybind 注册读，搞清三件事：`__global__`/`__device__` 的分工、grid/block 怎么算、host 侧怎么把 torch tensor 变成 kernel 参数。
- 读完能回答：一个 CUDA kernel 从 Python 调用到上 GPU 执行的完整链路长什么样？

**阶段 3：融合算子（本项目核心资产）**

- `csrc/fused_fp4/pi05_e0m3_act.cu`（480 行）——激活侧融合 kernel：量化 + RHT（随机 Hadamard 变换）+ UE4M3 block scale 一次过。这是 NVFP4 精度能零掉点的关键之一，也是后续"GPTQ pack 消费层"要对接的激活格式。
- 读法：先读 `.cuh` 的函数签名和注释搞清楚输入输出契约，再读 `.cu` 主 kernel 的线程分工，RHT 的数学细节可以先当"一个正交变换"跳过。
- 读完能回答：E0M3 激活 + UE4M3 scale 的内存布局是什么？为什么融合比分开做快？

**阶段 4：GEMM 只读壳（不深入 CUTLASS）**

- `csrc/gemm/fp4/cutlass_fp4_gemm_e0m3w_sm100.cu`（151 行）——E0M3 权重 tcgen05 GEMM 的 host 侧封装。只读它怎么选 CUTLASS 模板、怎么填 descriptor、怎么 launch；`.cuh` 里的 CUTLASS 模板展开和 tcgen05 指令**不要追**，那是另一个数量级的复杂度（等真要写 GEMM 时再回来）。
- 读完能回答：CUTLASS kernel 的"配置 → 实例化 → launch"三段式长什么样？per-16 block scale 是以什么形式传进 GEMM 的？

辅助材料：仓库根 `README.md` / `USAGE.md` / `docs/`（作者的设计说明）；`perf_data/` 里的 nsys 统计（kernel 名字和 csrc 文件名基本一一对应，可以拿 profile 反查"这个 kernel 实际占多少时间"）。读 kernel 时手边放一份 CUDA C Programming Guide 的 memory model 章节即可，不需要先系统学完 CUDA。

**阶段 5：CUDA Graph runtime（终态形态，M2 后续路线）**

FlashRT 不走 torch.compile，而是自己抓 CUDA Graph：buffer 全部静态持有（地址 capture 后不动）、输入形状分档、每档 warmup 后录制一次、之后整图回放，新数据只就地拷进静态 buffer。与 torch.compile 的区别：录制在驱动层进行、不需要理解 kernel 语义，所以 pybind 黑盒/裸指针随便用；代价是形状和 buffer 地址必须静态。按顺序读：

| 顺序 | 文件 | 看什么 |
|---|---|---|
| 1 | `USAGE.md` §Pi0.5 State Prompts（~L130-165） | fixed 模式语义：一图打天下、无 warmup、不重录 |
| 2 | `docs/architecture.md` | 八大组件全景，graph capture 在 frontend 层的位置 |
| 3 | `flash_rt/core/cuda_graph.py`（81 行，全读） | 裸机制：ctypes 直调 cudaStreamBeginCapture，框架无关 |
| 4 | `flash_rt/frontends/torch/pi05_thor_fp4.py`：`_alloc_fp4_scratch_for_Se`(L1103)、`_capture_siglip_graph`(L838)、`_capture_enc_ae_graph`(L1277)、`_fp4_scratch_dict`(L1151) | 真实模型里的 buffer 所有制 + capture 完整姿势 |
| 5 | `flash_rt/structures/impls/decode_loop/whole_step.py` 的 `_StaticHybridCache`(L31) | KV cache 这种变长结构怎么静态化（预分配最大长度 + 就地更新） |
| 6 | `tests/test_pi05_state_prompt_fixed_graph.py` | 行为合同测试：什么情况允许重录、什么必须命中旧图 |
| 7 | `docs/adding_new_model.md` §0 + `docs/exec_contract.md` | 仓库硬规则（PR② 须遵守）+ Buffer/Graph/Plan C ABI |

读完能回答：为什么 CUDA Graph 能绕过 graph break 问题？我们 eager 消费层（`tools/omega_e0m3_linear.py`）要图化缺哪三件事（按 M 分档的持久 buffer、整段 expert forward 一次录制、输入就地拷贝）？

## 6. 当前进展

- [x] 三臂归因 + 探针失败模式分析（analysis.md）
- [x] LIBERO-Plus 鲁棒性配对（A 81.4% / C 64.3%）
- [x] 自定义时间网格注入（`PI05_T_GRID`）
- [x] 两步调度免训练评测（libero_10 ×500：50.4% vs 基线 91.6%，证否，见 analysis.md）
- [ ] off-manifold LoRA 修正器训练（基 U 预计算 + 校准集构建）
- [x] FlashRT 复现（FP8 91.6% / NVFP4 92.6% 零掉点，NVFP4+FA4 27.21ms；third_party/flashrt，见 analysis.md）
- [x] Omega-QVLA 臂 D 复现（W4A4/W4A8 hybrid 93.2%，per-step scale 完整配方，见 analysis.md）
