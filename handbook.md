# π₀.5 on Jetson AGX Thor 部署与性能复现手册

> 目标：任何人按本手册操作，可以在 AGX Thor 上复现 π₀.₅（`pi05_libero`）的
> TensorRT FP8/NVFP4 部署、推理性能数据（含 DRAM 带宽观测）和 LIBERO-Long 成功率。
>
> 本手册与 [OpenPi π₀.₅ on Jetson Thor | Jetson AI Lab](https://www.jetson-ai-lab.com/tutorials/openpi_on_thor/)
> 的关系：本分支（`nvidia-trt`）**已内置**该教程的全部部署脚本与源码补丁，
> 因此教程的 Step 2（打补丁/下载脚本）**不需要执行**；环境准备见本手册 Step 1–3，
> **从容器内环境配置（教程 Step 5）到 TensorRT 推理（教程 Step 12）按官方教程执行**，
> 之后的性能采集与成功率评测用本手册 Step 4–6（本分支新增工具）。

---

## 角色与环境

| 角色 | 配置 | 用途 |
|---|---|---|
| **Thor** | Jetson AGX Thor DevKit，JetPack 7.2（L4T R39.x），MAXN 功耗模式 | 部署、推理、性能采集、policy server |
| **host** | x86_64 Linux（分析机） | 解析 profiling 数据、跑 LIBERO 仿真 client |

软件基线：Docker 28+、nvidia-container-toolkit 1.18+、容器镜像 `openpi-pi0.5:l4t-jp7.2`
（base `nvcr.io/nvidia/pytorch:26.05-py3`，内置 nsys 2026.2.1 / TensorRT 10.16 / ModelOpt 0.43）。

验证环境：

```bash
cat /etc/nv_tegra_release     # 应显示 R39
nvidia-smi                    # Thor GPU，CUDA 13.x
docker --version
```

---

## Step 0：Fork 并克隆本分支

在 GitHub 上 Fork `https://github.com/Lmy271828/openpi`（或直接克隆），然后：

```bash
git clone -b nvidia-trt --recurse-submodules \
  https://github.com/<你的用户名>/openpi.git
cd openpi
git log --oneline -5
```

应看到（自上而下）：

```
<latest>  Remove round-1 perf report and data; re-measure on libero_10
34c6d0f   Add DRAM bandwidth observability to perf tooling
d484582   Add Thor deployment scripts and TensorRT FP8/NVFP4 perf analysis
8f1d732   Add TensorRT engine backend to serve_policy and relax checkpoint loading
15a9616   update output objects to support batching (#975)   ← 上游分叉点（教程验证的 commit）
```

分支内容 = 上游 `15a9616`（教程验证基线）+ 四层增量：

- `8f1d732`：`serve_policy.py` 支持 `--use-tensorrt` / `--tensorrt-engine`；checkpoint 加载 `strict=False`
- `d484582`：`deployment_scripts/` 全套（ONNX 导出、引擎构建、nsys 采集、num_steps 扫描）
- `34c6d0f`：tegrastats 阶段对齐（`--tegrastats-log`）+ GPU metrics 按 S2/S3 NVTX 探针精确切窗
- 最新：产物命名规范化（`pi05_<backend>[_<engine>]_<config>_w<W>_r<R>`）与大文件 gitignore

> **与官方教程的差异**：教程 Step 2.2 的 `download.sh`（下载 deployment_scripts + 打 4 个补丁）
> 在本分支上**全部跳过**——脚本和补丁（含 `transformers_replace`、`serve_policy` TRT 支持）
> 已在仓库内。

---

## Step 1：Thor 设为最高性能（对应教程 Step 1）

```bash
sudo nvpmodel -m 0        # MAXN
sudo jetson_clocks        # 锁定最高频率
sudo jetson_clocks --show # 验证
```

> 性能数据对功耗模式敏感：非 MAXN 下数字不可比。

## Step 2：构建 Docker 镜像（对应教程 Step 3）

```bash
sudo docker build -t openpi-pi0.5:l4t-jp7.2 -f deployment_scripts/thor.Dockerfile .
```

首次构建 15–20 分钟。base 镜像在 `nvcr.io`，拉取失败先 `docker login nvcr.io`。

## Step 3：启动容器（对应教程 Step 4，**多两个参数**）

```bash
sudo docker run --rm -it --runtime nvidia \
  --cap-add SYS_ADMIN \
  --network host \
  -v "$PWD":/workspace \
  -v "$HOME/.cache/openpi":/root/.cache/openpi \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -v /usr/bin/tegrastats:/usr/bin/tegrastats:ro \
  -v /sys:/sys:ro \
  -w /workspace \
  openpi-pi0.5:l4t-jp7.2
```

与教程的差异及原因：

- `--cap-add SYS_ADMIN`：GPU metrics 采样（Step 4 的 DRAM/SM 饱和度观测）需要，不加会被
  nsys 拒绝（`Illegal --gpu-metrics-devices usage ... Insufficient privilege`）
- `--network host`：替代 `-p 8000:8000`，同时方便 policy server 与 tegrastats 时间对齐
- `-v /sys:/sys:ro`：tegrastats 的 EMC_FREQ/GR3D_FREQ 读自 sysfs 节点，只挂二进制不够——
  读不到时 tegrastats 会静默省略这两个字段（`*_emc.log` 有采样行但没有 EMC/GR3D 数据）。
  必须把整个 `/sys` 挂进容器（NVIDIA 官方确认，forums.developer.nvidia.com/t/311539）
- `.cache` 两个挂载务必保留：checkpoint（~6 GB）、HF 数据集、ONNX/引擎全部落在里面，
  容器是 `--rm` 的，不挂载则每次重来

> **注意**：容器内写出的文件属 root，宿主机上清理/传输前先收权：
> `sudo chown -R hcclab:hcclab perf_data`

---

## Step 3.5 → 官方教程 Step 5–12（容器内，按教程执行）

从这里开始**完全按官方教程**操作，逐步对应关系与验收值：

| 教程步骤 | 内容 | 验收标志 |
|---|---|---|
| Step 5 | `export PYTHONPATH` + 选 `CONFIG_NAME=pi05_libero` + 打 transformers 补丁 | 补丁 cp 无报错 |
| Step 6 | 下载 JAX checkpoint（GCS 自动下载） | `~/.cache/openpi/openpi-assets/checkpoints/pi05_libero/` |
| Step 7 | JAX → PyTorch 转换（5–10 min） | `pi05_libero_pytorch/` 含 `model.safetensors` + `assets/` |
| Step 8 | PyTorch 推理自检 | **~130 ms**（torch.compile BF16） |
| Step 9 | ONNX 导出（FP8 + NVFP4，ModelOpt 校准） | `onnx/model_fp8_nvfp4.onnx` + `.data` |
| Step 10 | trtexec 构建引擎（10–30 min） | `engine/model_fp8_nvfp4.engine` |
| Step 11 | TRT 推理 | **~49 ms**，约 2.7× 加速 |
| Step 12 | compare 模式数值对照 | cosine similarity ≈ 0.99 |

教程链接：[OpenPi π₀.₅ on Jetson Thor](https://www.jetson-ai-lab.com/tutorials/openpi_on_thor/)

三个补充说明：

1. **Step 9 的校准数据**会自动从 HuggingFace 下载 `physical-intelligence/libero`
   （LeRobot 格式，缓存于 `~/.cache/huggingface`）。网络受限时：
   `export HF_ENDPOINT=https://hf-mirror.com`；需要 token 时 `export HF_TOKEN=<token>`。
2. **不要用 `--precision fp16`**：π₀.₅ 原生 BF16，FP16 动态范围不足会在 Gemma
   attention 层溢出（教程 Step 9 明确说明）。
3. 引擎构建产物（`*_profile.json` / `*_layers.json` / `.log`）在引擎同目录，
   Step 4 的采集脚本会自动拷贝。

---

## Step4：性能数据采集（Thor 容器内，本分支工具）

一键采集（sweep + PyTorch/TRT 两路 nsys + GPU metrics + 内嵌 tegrastats EMC + trtexec 产物）：

```bash
# 可选环境变量：CONFIG_NAME(=pi05_libero) NUM_WARMUP(=3) NUM_TEST_RUNS(=10)
#               CKPT_DIR / ENGINE_PATH
bash deployment_scripts/collect_perf_data.sh
```

产物命名**编码了采集超参**，不同配置的采集互不覆盖：

```
numsteps_sweep_<config>_w<W>_r<R>.csv                      # num_steps 扫描
pi05_ptcompile_<config>_w<W>_r<R>{.nsys-rep,.sqlite,_*.csv}  # PyTorch torch.compile
pi05_trt_<engine-tag>_<config>_w<W>_r<R>{.nsys-rep,...}      # TensorRT，engine-tag 取自
                                                             # 引擎文件名（如 fp8_nvfp4）
<prefix>_emc.log                                           # tegrastats 原始采样（EMC%/GR3D%）
<prefix>_console.log                                       # 控制台全量输出（含分阶段 EMC 表）
```

**DRAM 带宽（EMC）采集**：Thor 是 Tegra 统一内存架构，nsys GPU metrics **不含 DRAM 计数器**
（DRAM 挂在 SoC 侧 EMC），用 tegrastats 补齐——已嵌入推理脚本并经 `--tegrastats-log`
并入上面的一键采集（与 nsys 同一进程、时间轴天然对齐），无需单独跑。手工单独采集时：

```bash
python deployment_scripts/pi05_inference_nvtx.py \
  --config-name ${CONFIG_NAME} \
  --checkpoint-dir ~/.cache/openpi/openpi-assets/checkpoints/${CONFIG_NAME}_pytorch \
  --inference-mode tensorrt \
  --engine-path ~/.cache/openpi/openpi-assets/checkpoints/${CONFIG_NAME}_pytorch/engine/model_fp8_nvfp4.engine \
  --tegrastats-log perf_data/pi05_trt_fp8_nvfp4_${CONFIG_NAME}_w3_r10_emc.log
# 输出 warmup / inference_test 两阶段的 EMC% 均值/峰值与换算 GB/s（峰值 273 GB/s），
# 并写 <同前缀>.windows.json 供离线重新对齐；
# 文件名遵循一键脚本规范 <prefix>_emc.log（w/r 与实际 --num-warmup/--num-test-runs 一致）
```

> 判读参考：EMC% 稳态 >80% 才是带宽瓶颈；本模型两阶段 SM Issue 仅 ~9–20%，
> 属内存**延迟**受限而非带宽饱和（优化方向是 graph/融合/流水，不是继续压精度）。

拷回 host：

```bash
# host 上执行
scp hcclab@<THOR_IP>:~/lmy/openpi/perf_data/* perf_data/
```

---

## Step 5：host 侧分析

```bash
python deployment_scripts/analyze_perf.py --perf-dir perf_data --dram-peak-gbps 273
```

输出各段（运行前缀自动发现，多轮采集可同目录对比）：

| 段 | 数据源 | 内容 |
|---|---|---|
| num_steps 扫描 | `numsteps_sweep_*.csv` | 线性分解 `T(N) ≈ T_fixed + N·T_step`（ViT+LLM prefill vs expert 去噪） |
| DRAM / 显存带宽 | `*_gpumetrics.csv` 或含 GPU_METRICS 的 `.sqlite` | DRAM 计数器（dGPU）或 Copy Engine 代理（Tegra）；含 S2/S3 NVTX 探针的采集会额外输出 prefill/expert 分阶段饱和度表 |
| CUDA graph 使用 | `<prefix>.sqlite` 直读 | 稳态 graph replay 统计 |
| nsys kernel 分析 | `<prefix>_cuda_gpu_kern_sum.csv` | kernel 分类汇总（GEMM/attention/quant/elementwise…） |
| trtexec 逐层 profile | `*_profile.json` | ViT / LLM(NVFP4) / expert(FP8) 阶段耗时归因 |

> 注意：torch.compile 与 TRT 路径稳态计算在 CUDA graph 内，kernel 级 CSV 为空属
> **预期**（此 nsys 版本不归因 graph 内 kernel）；逐 kernel 分析用 Eager 对照组
> （`pi05_inference_nvtx.py --profile-eager` 采集）或 trtexec 逐层 profile。

---

## Step 6：LIBERO-Long 成功率评测（量化掉点对照）

LIBERO-Long 即 `libero_10` 任务套件。三臂对照以分离"TRT 转换误差"与"量化误差"：

| 臂 | 权重/计算精度 | 推理路径 | 回答的问题 |
|---|---|---|---|
| A | **BF16**（原始浮点权重，不量化） | PyTorch eager + torch.compile | 基准成功率 |
| B | **FP16**（浮点权重，不量化） | ONNX → TensorRT 引擎 | 转换误差（A − B）：ONNX 导出、TRT kernel 融合、bf16→fp16 数值差异 |
| C | **FP8 权重/激活 + NVFP4 LLM**（量化） | ONNX（QDQ）→ TensorRT 引擎 | 量化误差（B − C）：低精度带来的额外掉点 |

B 与 C 的代码路径完全相同，唯一差别是引擎是否量化；A − C 的总掉点由此拆成
"转换"和"量化"两段归因。

```bash
export CKPT=~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch
```

### server 的稳健启动方式（detached 容器 + tmux 看日志）

交互式 `docker run -it` 里的 server 会随终端/SSH 断开被杀（容器还是 `--rm` 的，直接消失）；
client 不会自动重连，之后每个 episode 都在第一次 infer 失败、被记为 False——
实测曾因此产生 115 个垃圾失败（一帧的 failure 视频即为此类异常的特征，
真实任务失败的视频是完整长度的）。评测务必让 server 脱离终端：

```bash
# Thor 宿主机上：server 作为容器主进程后台运行（注意是 -d，不是 -it；不要用 --rm）
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
           python scripts/serve_policy.py --port 8000 policy:checkpoint \
             --policy.config=pi05_libero \
             --policy.dir=/root/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"

# 观察日志（tmux 包住，SSH 断开也能回来看）
tmux new -s serve
docker logs -f pi05_server
# Ctrl+b d 脱离；tmux attach -t serve 重新d进入
```

- 启动日志出现 `[trt hooks] attention mask dtype fix installed` 才说明
  dtype 补丁（SDPA bias 与 query 同 dtype）已生效；缺了它第一个 client 请求必炸
  （`invalid dtype for bias`，client 侧表现为 1011 internal error）
- 换臂：见下节「换臂操作：server 换引擎 / client 换输出目录」
- 想要崩溃自动拉起：加 `--restart unless-stopped`（与 `--rm` 冲突，二选一）

### 换臂操作：server 换引擎 / client 换输出目录

臂 A 全量跑完（client 日志出现 `Total success rate`）后：

```bash
# ① Thor 宿主机：停掉当前 server
sudo docker stop pi05_server && sudo docker rm pi05_server
```

**臂 B 专属：先构建 fp16 引擎**（臂 C 跳过——fp8/nvfp4 引擎已建好）。
在 Thor 宿主机上另起一个临时构建容器（`--rm` 退出即删，产物落在挂载卷 `$CKPT/engine/` 不丢）：

```bash
sudo docker run --rm -it --runtime nvidia \
  --cap-add SYS_ADMIN \
  --network host \
  -v "$PWD":/workspace \
  -v "$HOME/.cache/openpi":/root/.cache/openpi \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -w /workspace \
  openpi-pi0.5:l4t-jp7.2 
  bash

# ---- 以下为容器内命令 ----
TF_DIR=/usr/local/lib/python3.12/dist-packages/transformers && \
  cp -r src/openpi/models_pytorch/transformers_replace/* $TF_DIR/
export PYTHONPATH=packages/openpi-client/src:src:.
export CKPT=/root/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch

# ONNX 导出（--precision fp16 是默认值，不量化、不需要校准，比 fp8 快）
python deployment_scripts/pytorch_to_onnx.py \
  --checkpoint_dir $CKPT --output_path $CKPT \
  --config_name pi05_libero --precision fp16

# 引擎构建（产出 $CKPT/engine/model_fp16.engine）
ACTION_HORIZON=10 bash deployment_scripts/build_engine.sh \
  $CKPT/onnx/model_fp16.onnx \
  $CKPT/engine/model_fp16.engine

# 5 分钟数值预检（cosine ≥ 0.999 再往下走，否则回去修导出，别浪费 5 小时跑 LIBERO）
python deployment_scripts/pi05_inference.py \
  --inference-mode compare \
  --config-name pi05_libero \
  --checkpoint-dir $CKPT \
  --engine-path $CKPT/engine/model_fp16.engine
```


**② Thor 宿主机：起 B/C 的 server**（两者命令相同，只换 `--tensorrt-engine` 路径；
注意 TRT 参数是顶层参数，必须在 `policy:checkpoint` 之前）：

```bash
# armB
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
           python scripts/serve_policy.py --port 8000 \
             --use-tensorrt \
             --tensorrt-engine /root/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch/engine/model_fp16.engine \
             policy:checkpoint \
             --policy.config=pi05_libero \
             --policy.dir=/root/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"

# armC
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
           python scripts/serve_policy.py --port 8000 \
             --use-tensorrt \
             --tensorrt-engine /root/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch/engine/model_fp8_nvfp4.engine \
             policy:checkpoint \
             --policy.config=pi05_libero \
             --policy.dir=/root/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
             
# 确认就绪：看到 server listening on 0.0.0.0:8000
# （TRT 首次加载引擎可能花几分钟）
docker logs -f pi05_server
```

**③ x86 host：起 client**（与臂 A 命令相同，只改输出目录/日志名）：

```bash
cd ~/pynoob/openpi
setsid nohup bash -c '
  source examples/libero/.venv/bin/activate &&
  export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero &&
  python examples/libero/main.py \
    --args.task-suite-name libero_10 \
    --args.num-trials-per-task 50 \
    --args.host <THOR_IP> --args.port 8000 \
    --args.video-out-path eval_out/libero10_arm<B|C>
' > eval_out/arm<B|C>.log 2>&1 < /dev/null &
pgrep -f "python examples/libero/main.py" | tail -1 > eval_out/arm<B|C>.pid
```

- server 异常退出排查：`docker logs pi05_server`（容器已删则宿主机
  `dmesg | grep -i oom` 查 OOM）

| 臂 | 权重类型 | server 启动命令（Thor 容器内） | 作用 |
|---|---|---|---|
| A | BF16 浮点（不量化） | `python scripts/serve_policy.py --port 8000 policy:checkpoint --policy.config=${CONFIG_NAME} --policy.dir=$CKPT` | PyTorch BF16 基准 |
| B | FP16 浮点（不量化） | `python scripts/serve_policy.py --port 8000 --use-tensorrt --tensorrt-engine $CKPT/engine/model_fp16.engine policy:checkpoint --policy.config=${CONFIG_NAME} --policy.dir=$CKPT`（需先按上节构建 fp16 引擎） | 隔离转换误差 |
| C | FP8 + NVFP4（量化） | `python scripts/serve_policy.py --port 8000 --use-tensorrt --tensorrt-engine $CKPT/engine/model_fp8_nvfp4.engine policy:checkpoint --policy.config=${CONFIG_NAME} --policy.dir=$CKPT` | 量化总掉点 |

注意 tyro 的参数归属规则：`--port`、`--use-tensorrt`、`--tensorrt-engine` 是顶层参数，
必须放在子命令 `policy:checkpoint` **之前**；放之后会报 Unrecognized options。

client（x86 仿真机，需先按 `examples/libero/README.md` 装好 LIBERO 环境）：

```bash
python examples/libero/main.py \
  --args.task-suite-name libero_10 \
  --args.num-trials-per-task 50 \
  --args.host <THOR_IP> --args.port 8000 \
  --args.video-out-path eval_out/libero10_<臂标记>
```

### client 的稳健启动方式（x86 host，nohup 脱离会话）

500 episodes 约 5 小时，client 同样不能挂在交互终端/AI 会话上。注意两侧的分工：
**server 在 Thor**（detached 容器，见上节），**client 在 x86 host**（nohup）：

```bash
# x86 host 上，openpi 仓库根目录
cd ~/pynoob/openpi
setsid nohup bash -c '
  source examples/libero/.venv/bin/activate &&
  export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero &&
  python examples/libero/main.py \
    --args.task-suite-name libero_10 \
    --args.num-trials-per-task 50 \
    --args.host <THOR_IP> --args.port 8000 \
    --args.video-out-path eval_out/libero10_<臂标记>
' > eval_out/<臂标记>.log 2>&1 < /dev/null &

# 记录真实 python PID（$! 是外层 bash 的，杀了不顶用）
pgrep -f "python examples/libero/main.py" | tail -1 > eval_out/<臂标记>.pid
```

- 观察：`tail -f eval_out/<臂标记>.log`；进度 `grep -c "Success:" eval_out/<臂标记>.log`
- 中止：`kill $(cat eval_out/<臂标记>.pid)`
- `setsid` + `nohup` + `< /dev/null` 三者都要：脱离控制终端、免疫 SIGHUP、
  断开 stdin（LIBERO 首次 import 有交互式 `input()` 提问，配置已生成则不会再问，
  但 stdin 悬空的进程在终端关闭时仍可能收到信号）
- 起跑 3–5 分钟没有 episode 完成是正常的：server 端首次推理触发 torch.compile；
  若日志出现批量 `Caught exception`，先查 server（`docker logs pi05_server`）

### LIBERO-Plus 鲁棒性评测（7 扰动维度 × 30 任务子集）

环境独立于原版 LIBERO：专用 venv `examples/libero/.venv-plus`（libero 指向
`third_party/libero_plus`，robosuite 冲突已用放宽版 requirements 解决），
config 切换用备份双份：

```bash
cp ~/.libero/config.yaml.libero_plus ~/.libero/config.yaml   # 切到 plus
cp ~/.libero/config.yaml.libero ~/.libero/config.yaml        # 切回原版
```

评测子集：`eval_out/libero_plus_subset.json`（seed=42，7 维度 × 30 任务 = 210，
按难度 1-5 分层、每层 6 个；Objects Layout 的 level5 只有 3 个，缺口补到了 level1）。
两臂用同一子集、每任务 1 trial，逐任务配对。

```bash
# x86 host，openpi 仓库根目录（server 侧与 Step 6 完全相同，不用动）
cd ~/pynoob/openpi
setsid nohup bash -c '
  source examples/libero/.venv-plus/bin/activate &&
  export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero_plus &&
  TASK_IDS=$(python3 -c "import json; print(\" \".join(map(str, json.load(open(\"eval_out/libero_plus_subset.json\"))[\"task_ids_0based\"])))") &&
  python examples/libero/main.py \
    --args.task-suite-name libero_10 \
    --args.task-ids $TASK_IDS \
    --args.num-trials-per-task 1 \
    --args.host <THOR_IP> --args.port 8000 \
    --args.video-out-path eval_out/libero_plus_<臂标记>
' > eval_out/plus_<臂标记>.log 2>&1 < /dev/null &
```

- 先跑 2-3 个任务冒烟（`--args.task-ids` 只给前几个 id），确认新 assets 加载正常再全量
- 判读：按维度分组比较 A vs C 成功率（每维度 n=30，CI≈±18%，看方向和大差距）；
  量化在扰动下的掉点显著大于基准掉点的维度 = 部署风险维度
- 子集再生成：`task_classification.json` 里 id 是 1-based，suite 索引 = id − 1（已验证）

判读规则：

- 公平性由 `main.py` 内置机制保证：固定 seed（默认 7）+ 固定初始状态，两臂 episode
  逐对配对；`replan_steps=5`、`resize_size=224` 保持一致
- 每套件 n = 10 任务 × 50 trials = 500 episodes，二项 95% CI ≈ ±4.4%；
  **ΔSR < ±5% 只能下"无显著差异"结论**
- 配对样本用 McNemar 检验（只看一成一败的对子）更灵敏
- flow matching 的 noise 每次推理随机采样，单 episode 成败不可比，必须靠样本量
- 迭代期可用 `--args.num-trials-per-task 10` 冒烟，最终结论必须 50

---

## 复现检查清单

按本手册操作后应得到：

- [ ] PyTorch BF16 ~130 ms / TRT FP8+NVFP4 ~49 ms（MAXN，action horizon 10）
- [ ] compare 模式 cosine similarity ≈ 0.99
- [ ] `numsteps_sweep_*.csv` 线性拟合：`T_fixed`（ViT+LLM prefill）与 `T_step`（单步 expert）分解
- [ ] tegrastats EMC% 分阶段表（warmup / inference_test）
- [ ] nsys GPU metrics 分阶段饱和度（SMs Active / SM Issue / Tensor Active × prefill / expert）
- [ ] libero_10 三臂成功率表（每臂 500 episodes）

---

## 排查表

| 症状 | 原因与修复 |
|---|---|
| `ModuleNotFoundError: No module named 'openpi'` | 容器内未 `export PYTHONPATH=packages/openpi-client/src:src:.:$PYTHONPATH`（`collect_perf_data.sh` 已内置自愈，手工跑脚本时需自己 export） |
| `Illegal --gpu-metrics-devices usage ... Insufficient privilege` | 容器缺权限，`docker run` 加 `--cap-add SYS_ADMIN` 重开 |
| 没有生成 `*_emc.log` / 日志提示 `tegrastats not found` | 容器内没有 tegrastats 二进制（logger 会静默禁用）。`docker run` 加 `-v /usr/bin/tegrastats:/usr/bin/tegrastats:ro` 重开后重跑 |
| `*_emc.log` 有采样行但无 EMC_FREQ/GR3D_FREQ 字段 | 容器只挂了 tegrastats 二进制没挂 `/sys`，tegrastats 读不到 sysfs 节点时静默省略字段。`docker run` 加 `-v /sys:/sys:ro` 重开后重跑 |
| EMC 分阶段表 inference_test 0 样本 | GB10y tegrastats 实际采样 ~8 Hz（名义 100 ms），index×间隔对齐漂移 ~20 s/110 s。analyze_perf.py 与 logger 已改为按行时间戳对齐（相对首样本，免时区），旧日志重跑 analyze 即可 |
| client 首请求即 1011 / `invalid dtype for bias` | PyTorch 路径的加性 attention mask 是 fp32、query 是 bf16，编译后 SDPA 要求同 dtype。`serve_policy.py` 已自动安装 `install_attention_mask_dtype_fix`（启动日志应有 `[trt hooks] attention mask dtype fix installed`）；没有这行说明脚本不是最新 |
| 评测中途批量 `no close frame received or sent`、failure 视频只有一帧 | server 进程死了（终端断开或崩溃），client 不重连、每集首步 infer 即异常退出，垃圾失败会计入成功率——该轮数据作废。用 Step 6 的 detached 容器方式起 server；`docker logs pi05_server` 或 `dmesg \| grep -i oom` 查死因 |
| nsys GPU metrics 里没有 DRAM 指标 | Tegra iGPU 的 DRAM 在 SoC 侧 EMC，不归 GPU metrics 采样——用 `--tegrastats-log`（本分支工具）或 NCU `dram__*` |
| host 的 nsys 打不开 Thor 的 `.nsys-rep` | 版本前向不兼容；改读 `nsys stats` 同时导出的 `.sqlite`（标准 SQLite，跨版本可查），`analyze_perf.py` 已这么做 |
| ptc/trt 的 `*_sum.csv` 是 0 字节 | 稳态计算在 CUDA graph replay 内，此 nsys 版本不归因 graph 内 kernel——预期行为，不是采集失败 |
| 宿主机删不动 `perf_data/` 文件 | 容器内 root 写出，`sudo chown -R hcclab:hcclab perf_data` |
| 校准/数据集下载失败 | `export HF_ENDPOINT=https://hf-mirror.com` 或配置 `HF_TOKEN` |
| compare 模式 cosine 偶发偏低 | 每次随机 noise 所致，多跑几次或用 `--golden-noise-path` 固定 |
| ONNX 导出报 FP4 reshape 错 | transformers 补丁没打（教程 Step 5.3 / collect 脚本自愈段） |

---

## 参考

- [OpenPi π₀.₅ on Jetson Thor | Jetson AI Lab](https://www.jetson-ai-lab.com/tutorials/openpi_on_thor/)（部署主流程，本手册 Step 3.5 起沿用）
- [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)（上游仓库，分叉点 `15a9616`）
- `docs/pi05_inference_reading_guide.md`（推理链路代码导读）
