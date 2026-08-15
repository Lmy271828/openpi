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
# ===== 本地 x86 侧：同步 libero 仓库到 Thor（一次性）=====
rsync -avz --exclude=.git ~/pynoob/openpi/third_party/libero hcclab@10.191.163.226:~/lmy/openpi/third_party/

# ===== Thor 侧 =====
cd ~/lmy/openpi/third_party/flashrt
source ~/lmy/openpi/deployment_scripts/thor_jp72_env.sh

# 验证环境
python -c "import flash_rt, numpy, torch; print(flash_rt.__version__, numpy.__version__, torch.__version__, torch.cuda.is_available())"

# tokenizer（已存在则跳过）
ls ~/.cache/openpi/big_vision/paligemma_tokenizer.model 2>/dev/null || bash scripts/download_paligemma_tokenizer.sh

# 仿真最小依赖（PYTHONPATH 已由 thor_jp72_env.sh 设置，子进程会继承）
pip install "robosuite==1.4.1" "mujoco==3.2.3" bddl easydict gym opencv-python-headless PyOpenGL
python -c "from libero.libero import benchmark; print('libero ok')"

# LIBERO 路径配置（Thor 侧直接写入；注意 init 目录名是 init_files，层级为 libero/libero/libero）
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

# 全量（10 任务 x 50 episodes，FP8；对照臂 C = 80.0%，官方参考 92.6-93%）
setsid nohup python examples/thor/eval_libero.py \
  --checkpoint ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch \
  --task_suite libero_10 \
  > eval_flashrt_fp8_libero10.log 2>&1 < /dev/null &

# NVFP4 变体（加 --use_fp4）
setsid nohup python examples/thor/eval_libero.py \
  --checkpoint ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch \
  --task_suite libero_10 --use_fp4 \
  > eval_flashrt_nvfp4_libero10.log 2>&1 < /dev/null &

# 查进度
tail -5 eval_flashrt_fp8_libero10.log
```

结果写入 `libero_libero_10_torch_results.json`（逐任务成功率 + P50 延迟）。max steps 口径与 openpi client 一致（libero_10 = 520 步/episode），成功率可直接对比。

## 6. 当前进展

- [x] 三臂归因 + 探针失败模式分析（analysis.md）
- [x] LIBERO-Plus 鲁棒性配对（A 81.4% / C 64.3%）
- [x] 自定义时间网格注入（`PI05_T_GRID`）
- [ ] 两步调度免训练冒烟（libero_10 × 10 trials/档）
- [ ] off-manifold LoRA 修正器训练（基 U 预计算 + 校准集构建）
- [ ] FlashRT FP8 W8A8 静态 scale / NVFP4+AWQ 复现（third_party/flashrt）
