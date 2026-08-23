# π₀.5 on Jetson AGX Thor 部署与性能复现手册

> 目标：复现 π₀.₅（`pi05_libero`）在 Thor 上的部署、推理性能与 LIBERO 评测。
> 本手册只收可复现的操作命令；数字口径、判读规则、实测结果见 `analysis.md`。
>
> 与 [官方教程](https://www.jetson-ai-lab.com/tutorials/openpi_on_thor/) 的关系：
> 本分支（`nvidia-trt`）已内置教程的全部脚本与补丁，教程 Step 2 跳过；
> 环境准备见本手册 Step 1–3；容器内配置到 TRT 推理按官方教程 Step 5–12；
> 之后的性能采集与成功率评测用本手册 Step 4–6。

---

## 角色与环境

| 角色 | 配置 | 用途 |
|---|---|---|
| **Thor** | Jetson AGX Thor DevKit，JetPack 7.2（L4T R39.x），MAXN 功耗模式 | 部署、推理、性能采集、policy server |
| **host** | x86_64 Linux | 解析 profiling 数据、跑 LIBERO 仿真 client |

软件基线：Docker 28+、nvidia-container-toolkit 1.18+、镜像 `openpi-pi0.5:l4t-jp7.2`
（base `nvcr.io/nvidia/pytorch:26.05-py3`，内置 nsys 2026.2.1 / TensorRT 10.16 / ModelOpt 0.43）。

```bash
cat /etc/nv_tegra_release
nvidia-smi
docker --version
```

---

## Step 0：克隆本分支

```bash
git clone -b nvidia-trt --recurse-submodules \
  https://github.com/<你的用户名>/openpi.git
cd openpi
```

## Step 1：Thor 设为最高性能（教程 Step 1）

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
sudo jetson_clocks --show
```

性能数据对功耗模式敏感：非 MAXN 下数字不可比。

## Step 2：构建 Docker 镜像（教程 Step 3）

```bash
sudo docker build -t openpi-pi0.5:l4t-jp7.2 -f deployment_scripts/thor.Dockerfile .
```

## Step 3：启动容器（教程 Step 4，多两个参数）

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

与教程的差异：`--cap-add SYS_ADMIN`（GPU metrics 采样）、`--network host`（替代
`-p 8000:8000`）、`-v /sys:/sys:ro`（tegrastats 的 EMC/GR3D 读自 sysfs，只挂二进制会静默省略）。

容器内写出的文件属 root，宿主机清理前先 `sudo chown -R $USER perf_data`。

## Step 3.5：官方教程 Step 5–12（容器内，按教程执行）

| 教程步骤 | 内容 | 验收标志 |
|---|---|---|
| Step 5 | `export PYTHONPATH` + `CONFIG_NAME=pi05_libero` + transformers 补丁 | 补丁 cp 无报错 |
| Step 6 | 下载 JAX checkpoint | `~/.cache/openpi/openpi-assets/checkpoints/pi05_libero/` |
| Step 7 | JAX → PyTorch 转换（5–10 min） | `pi05_libero_pytorch/` 含 `model.safetensors` + `assets/` |
| Step 8 | PyTorch 推理自检 | ~137 ms（torch.compile BF16） |
| Step 9 | ONNX 导出（FP8 + NVFP4，ModelOpt 校准） | `onnx/model_fp8_nvfp4.onnx` + `.data` |
| Step 10 | trtexec 构建引擎（10–30 min） | `engine/model_fp8_nvfp4.engine` |
| Step 11 | TRT 推理 | ~49.9 ms |
| Step 12 | compare 模式数值对照 | cosine ≥ 0.99 |

补充：Step 9 校准数据从 HuggingFace 下载 `physical-intelligence/libero`，网络受限时
`export HF_ENDPOINT=https://hf-mirror.com`；引擎构建产物（`*_profile.json` / `*_layers.json`
/ `.log`）在引擎同目录，Step 4 的采集脚本会自动拷贝。

---

## Step 4：性能数据采集（Thor 容器内）

```bash
bash deployment_scripts/collect_perf_data.sh
```

可选环境变量：`CONFIG_NAME`(=pi05_libero) `NUM_WARMUP`(=3) `NUM_TEST_RUNS`(=10)
`CKPT_DIR` / `ENGINE_PATH`。产物命名编码采集超参，多轮互不覆盖：

```
numsteps_sweep_<config>_w<W>_r<R>.csv
pi05_ptcompile_<config>_w<W>_r<R>{.nsys-rep,.sqlite,_*.csv}
pi05_trt_<engine-tag>_<config>_w<W>_r<R>{.nsys-rep,...}
<prefix>_emc.log          # tegrastats（EMC%/GR3D），时间轴与 nsys 对齐
<prefix>_console.log
```

拷回 host：

```bash
scp hcclab@10.191.163.226:~/lmy/openpi/perf_data/* perf_data/
```

---

## Step 5：host 侧分析

```bash
python deployment_scripts/analyze_perf.py --perf-dir perf_data --dram-peak-gbps 273
```

数据源对照：`numsteps_sweep_*.csv`（num_steps 线性分解）、`*_gpumetrics.csv` / `.sqlite`
（分阶段饱和度）、`*_cuda_gpu_kern_sum.csv`（kernel 汇总）、`*_profile.json`
（trtexec 逐层 profile）。torch.compile 与 TRT 稳态计算在 CUDA graph 内，kernel 级
CSV 为空属预期（此 nsys 版本不归因 graph 内 kernel）；逐 kernel 分析用 eager 对照组
或 trtexec profile。

---

## Step 6：LIBERO-Long 成功率评测

LIBERO-Long 即 `libero_10` 任务套件。三臂：

| 臂 | 精度 | 推理路径 |
|---|---|---|
| A | BF16（不量化） | PyTorch eager + torch.compile |
| B | FP16（不量化） | ONNX → TRT 引擎 |
| C | FP8 + NVFP4（量化） | ONNX（QDQ）→ TRT 引擎 |

### server（Thor 宿主机，detached 容器）

server 必须脱离终端运行（交互式容器里的 server 随 SSH 断开被杀）：

```bash
sudo docker stop pi05_server 2>/dev/null; sudo docker rm pi05_server 2>/dev/null

# TRT_FLAGS 按臂选择（注意：TRT 参数是顶层参数，必须在 policy:checkpoint 之前）：
#   armA: TRT_FLAGS=""
#   armB: TRT_FLAGS="--use-tensorrt --tensorrt-engine /root/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch/engine/model_fp16.engine"
#   armC: TRT_FLAGS="--use-tensorrt --tensorrt-engine /root/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch/engine/model_fp8_nvfp4.engine"
TRT_FLAGS=""

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
           python scripts/serve_policy.py --port 8000 $TRT_FLAGS \
             policy:checkpoint \
             --policy.config=pi05_libero \
             --policy.dir=/root/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"

sudo docker logs -f pi05_server
```

就绪标志：`server listening on 0.0.0.0:8000`（TRT 首次加载引擎可能要几分钟）。
启动日志应含 `[trt hooks] attention mask dtype fix installed`，缺了它 client 首请求必炸。

换臂 = 换 `TRT_FLAGS` 重跑上面这段；确认当前臂：

```bash
sudo docker logs pi05_server 2>&1 | grep -iE "tensorrt|engine" | head -5
```

### 臂 B 引擎：构建 + 数值预检 + 延迟测试

臂 C 引擎已在教程 Step 9–10 建好；臂 B 需单独构建（Thor 容器内）：

```bash
TF_DIR=/usr/local/lib/python3.12/dist-packages/transformers && \
  cp -r src/openpi/models_pytorch/transformers_replace/* $TF_DIR/
export PYTHONPATH=packages/openpi-client/src:src:.
export CKPT=/root/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch

python deployment_scripts/pytorch_to_onnx.py \
  --checkpoint_dir $CKPT --output_path $CKPT \
  --config_name pi05_libero --precision fp16

ACTION_HORIZON=10 bash deployment_scripts/build_engine.sh \
  $CKPT/onnx/model_fp16.onnx \
  $CKPT/engine/model_fp16.engine

python deployment_scripts/pi05_inference.py \
  --inference-mode compare \
  --config-name pi05_libero \
  --checkpoint-dir $CKPT \
  --engine-path $CKPT/engine/model_fp16.engine
```

预检通过标准：cosine ≥ 0.999。不达标就回去修导出，不要带着坏引擎跑评测。

B/C 引擎端到端延迟（server 空闲时可与 pi05_server 共存，`--engine-path` 换引擎即换臂）：

```bash
sudo docker exec pi05_server bash -c '
export PYTHONPATH=packages/openpi-client/src:src:. && \
python deployment_scripts/pi05_inference.py \
  --inference-mode tensorrt \
  --config-name pi05_libero \
  --checkpoint-dir /root/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch \
  --engine-path /root/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch/engine/model_fp16.engine \
  --num-warmup 3 --num-test-runs 20'
```

看输出末尾 `Model inference time: <mean> ± <std> ms`。

### client（x86 host，nohup 脱离会话）

500 episodes 约 5 小时，client 同样不能挂在交互终端上：

```bash
cd ~/pynoob/openpi

setsid nohup bash -c '
  source examples/libero/.venv/bin/activate &&
  export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero &&
  python examples/libero/main.py \
    --args.task-suite-name libero_10 \
    --args.num-trials-per-task 50 \
    --args.host 10.191.163.226 --args.port 8000 \
    --args.video-out-path eval_out/libero10_armA
' > eval_out/armA.log 2>&1 < /dev/null &
```

换臂只改 `video-out-path` 与日志名。监控与中止：

```bash
tail -f eval_out/armA.log
grep -c "Success:" eval_out/armA.log
ps -eo pid,args | grep "[m]ain.py"
kill <PID>
```

- `setsid` + `nohup` + `< /dev/null` 三者都要（脱离终端、免疫 SIGHUP、断开 stdin）
- 起跑 3–5 分钟没有 episode 完成是正常的（server 端首次推理触发 torch.compile / 引擎加载）
- 日志出现批量 `Caught exception` 先查 server：`sudo docker logs pi05_server`
- 迭代期可用 `--args.num-trials-per-task 10` 冒烟，最终结论必须 50

### task4 / task9 探针分析

探针（`--args.probe`）在每集结束时记录任务物体的世界系坐标（cm，带符号）与关节 qpos。
前置：server 为目标臂；config 必须是**原版 libero**（`get_libero_path` 每个任务重读
`~/.libero/config.yaml`，Plus 评测运行中切 config 会把正在跑的 client 搞崩）：

```bash
cp ~/.libero/config.yaml.libero ~/.libero/config.yaml

cd ~/pynoob/openpi
setsid nohup bash -c '
  source examples/libero/.venv/bin/activate &&
  export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero &&
  python examples/libero/main.py --args.task-suite-name libero_10 --args.task-ids 4 \
    --args.num-trials-per-task 50 --args.probe \
    --args.host 10.191.163.226 --args.port 8000 \
    --args.video-out-path eval_out/task4s_probe_armC &&
  python examples/libero/main.py --args.task-suite-name libero_10 --args.task-ids 9 \
    --args.num-trials-per-task 50 --args.probe \
    --args.host 10.191.163.226 --args.port 8000 \
    --args.video-out-path eval_out/task9_probe_armC
' > eval_out/probes_armC.log 2>&1 < /dev/null &

setsid nohup bash examples/libero/video_archiver.sh eval_out/task4s_probe_armC \
  > eval_out/archiver_task4s.log 2>&1 < /dev/null &
setsid nohup bash examples/libero/video_archiver.sh eval_out/task9_probe_armC \
  > eval_out/archiver_task9.log 2>&1 < /dev/null &
```

探针输出（每集一行，在 client 日志里）：

```
[probe] final_check_success=True done=True | plate_1=(x,y,z)cm ... white_yellow_mug_1_joint0=[7 维] ...
```

逐集视频在 `archived/` 子目录（归档器按完成时间戳改名，避免同结局互相覆盖）。
判读方法见 `analysis.md` 探针节。

### LIBERO-Plus 鲁棒性评测（7 扰动维度 × 30 任务子集）

环境独立于原版：专用 venv `examples/libero/.venv-plus`（libero 指向
`third_party/libero_plus`）。`~/.libero/config.yaml` 的双份备份首次使用前创建：

```bash
cp ~/.libero/config.yaml ~/.libero/config.yaml.libero            # 原版备份（只需一次）
cp ~/.libero/config.yaml.libero_plus ~/.libero/config.yaml       # 切到 plus
cp ~/.libero/config.yaml.libero ~/.libero/config.yaml            # 切回原版
```

评测子集 `eval_out/libero_plus_subset.json`：seed=42，7 维度 × 30 任务 = 210。
两臂用同一子集、每任务 1 trial：

```bash
cd ~/pynoob/openpi

setsid nohup bash -c '
  source examples/libero/.venv-plus/bin/activate &&
  export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero_plus &&
  TASK_IDS=$(python3 -c "import json; print(\" \".join(map(str, json.load(open(\"eval_out/libero_plus_subset.json\"))[\"task_ids_0based\"])))") &&
  python examples/libero/main.py \
    --args.task-suite-name libero_10 \
    --args.task-ids $TASK_IDS \
    --args.num-trials-per-task 1 \
    --args.host 10.191.163.226 --args.port 8000 \
    --args.video-out-path eval_out/libero_plus_armA
' > eval_out/plus_armA.log 2>&1 < /dev/null &
```

先冒烟（`--args.task-ids` 只给前 3 个 id），确认 assets 加载正常再全量。

---

## FlashRT 复现环境（third_party/flashrt submodule）

```bash
git submodule add https://github.com/flashrt-project/FlashRT.git third_party/flashrt

# 离线时用本地克隆做源再改回规范 URL：
git -c protocol.file.allow=always submodule add ~/pynoob/FlashRT third_party/flashrt
git config -f .gitmodules submodule.third_party/flashrt.url \
  https://github.com/flashrt-project/FlashRT.git
git -c protocol.file.allow=always submodule sync third_party/flashrt
```

运行环境独立于 openpi 容器（Thor 宿主机专用 venv；它要编译 SM110 kernel），
装完先跑最简 smoke（加载 checkpoint + 单条推理）。适配 server 为
`deployment_scripts/flashrt_serve.py`（FlashRT 对内、openpi-client websocket 对外）。
跑时停掉 pi05_server 避免 GPU 抢占。FlashRT 自身的构建/bench 命令见其 repo 文档
（`docs/pi05_thor_decoder_fp4_e2e.md` 等）。

---

## Omega-QVLA 复现（臂 D：W4A4 GPTQ + DuQuant）

Omega-QVLA 是 openpi 的**兄弟目录独立克隆，不进 submodule**（不改它的源码；
`packs_hf/` 里的 GPTQ pack 来自 HF 下载，不入 git）。本地 `~/pynoob/Omega-QVLA`，
Thor `~/lmy/Omega-QVLA`，用 rsync 同步：

```bash
rsync -avz --exclude=.git ~/pynoob/Omega-QVLA hcclab@10.191.163.226:~/lmy/
```

集成方式：整个目录挂进 openpi 容器 `/opt/omega`，靠 `PYTHONPATH` + `GR00T_*`
环境变量生效。openpi 侧唯一改动：`src/openpi/models_pytorch/pi0_pytorch.py` 的
`sample_actions` 循环进入 `gr00t.quantization.dit_step_context.set_dit_quant_step(t)`
（guarded import，无 Omega-QVLA 时为 no-op；不打则 per-step scale 表不生效）。

**include 正则必须显式给**（默认值是 GR00T N1.x 的模块名，在 pi0.5 上匹配 0 层，
且不报错——`enable_gptq_if_configured` 匹配 0 层也返回 True）：

- expert（GPTQ）：`.*paligemma_with_expert\.gemma_expert\.model\.layers\.[0-9]+\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj).*`
- PaliGemma（DuQuant）：`.*paligemma_with_expert\.paligemma\.model\.language_model\.layers\.[0-9]+\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj).*`
- `state_proj`/`action_in_proj`/`action_out_proj`/`time_mlp_*` 小投影层在 A4 下会崩，
  官方明确排除在 include 之外（上面的正则天然不含它们）

server（Thor 宿主机，先停掉占 GPU 的容器/进程）：

```bash
sudo docker stop pi05_server 2>/dev/null; sudo docker rm pi05_server 2>/dev/null

sudo docker run -d --name pi05_server --runtime nvidia \
  --cap-add SYS_ADMIN \
  --network host \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$HOME/lmy/openpi":/workspace \
  -v "$HOME/lmy/Omega-QVLA":/opt/omega \
  -v "$HOME/.cache/openpi":/root/.cache/openpi \
  -w /workspace \
  openpi-pi0.5:l4t-jp7.2 \
  bash -c "TF_DIR=/usr/local/lib/python3.12/dist-packages/transformers && \
           cp -r src/openpi/models_pytorch/transformers_replace/* \$TF_DIR/ && \
           export PYTHONPATH=packages/openpi-client/src:src:.:/opt/omega && \
           export GR00T_GPTQ=1 \
                  GR00T_GPTQ_PATH=/opt/omega/packs_hf/pi05_long/quantized.pt \
                  GR00T_GPTQ_INCLUDE='.*paligemma_with_expert\.gemma_expert\.model\.layers\.[0-9]+\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj).*' \
                  GR00T_GPTQ_WBITS_DEFAULT=4 \
                  GR00T_GPTQ_ABITS=4 \
                  GR00T_GPTQ_MISSING=fallback \
                  GR00T_DUQUANT_INCLUDE='.*paligemma_with_expert\.paligemma\.model\.language_model\.layers\.[0-9]+\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj).*' \
                  GR00T_DUQUANT_ROT_MODE=svd_hadamard \
                  GR00T_DUQUANT_PERM_SCORE=weight \
                  GR00T_DUQUANT_BLOCK=64 \
                  GR00T_DUQUANT_BLOCK_OUT=64 \
                  GR00T_DUQUANT_PERMUTE=1 \
                  GR00T_DUQUANT_ROW_ROT=restore \
                  GR00T_DUQUANT_ACT_PCT=99.9 \
                  GR00T_DUQUANT_CALIB_STEPS=32 \
                  GR00T_DUQUANT_LS=0.15 && \
           python -u /opt/omega/scripts/openpi_inference_service.py \
             --model_path /root/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch \
             --data_config pi05_libero \
             --port 8000"

sudo docker logs -f pi05_server
```

启动后必查三项（`python -u` 保证 print 不被 stdout 缓冲吞掉）：

1. `[GR00T-GPTQ] Matched Linear layers: 126` 且 `[GR00T-DUQUANT] Matched Linear layers: 126`；
   是 0 就是正则没匹配上，量化没生效
2. denoise 步数必须 10（pack 的 per-step scale 表按 10 步建）
3. 前 ~32 次推理在收 DuQuant 校准 scale，确认校准完成再发评测流量

client（x86 host，与臂 A 同一入口，只改输出目录）：

```bash
cd ~/pynoob/openpi
setsid nohup bash -c '
  source examples/libero/.venv/bin/activate &&
  export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero &&
  python examples/libero/main.py \
    --args.task-suite-name libero_10 \
    --args.num-trials-per-task 1 \
    --args.host 10.191.163.226 --args.port 8000 \
    --args.video-out-path eval_out/omega_w4a4_smoke
' > eval_out/omega_w4a4_smoke.log 2>&1 < /dev/null &
```

先 10 trials 冒烟，确认成功率不离谱再 `--args.num-trials-per-task 50` 全量
（输出 `eval_out/omega_w4a4_long`）。GptqLinear 是 fake-quant 仿真（bf16 稠密 GEMM），
此臂只验证精度。

```bash
setsid nohup bash -c '
  source examples/libero/.venv/bin/activate &&
  export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero &&
  python examples/libero/main.py \
    --args.task-suite-name libero_10 \
    --args.num-trials-per-task 50 \
    --args.host 10.191.163.226 --args.port 8000 \
    --args.video-out-path eval_out/omega_w4a4_long 
' > eval_out/omega_w4a4_long.log 2>&1 < /dev/null &
```

---

## 路线 A：双侧 E0M3 + prefix 抓图（部署与验证）

把两侧 Linear 全换为 E0M3 消费层（格式与设计见
`third_party/flashrt/docs/omega_pack_e0m3.md`），prefix prefill + 10 步 denoise 双图
（`tools/omega_e0m3_graph.py`）。服务端开关在 `tools/start_e0m3_server.sh` 默认开：
`OMEGA_E0M3_PATCH_DUQUANT=1`、`OMEGA_E0M3_PREFIX_GRAPH=1`、`OMEGA_E0M3_CUDA_GRAPH=1`。

### 0. 同步与 artifact 检查（Thor）

```bash
cd ~/lmy/openpi && git pull --ff-only && git submodule update third_party/flashrt

# artifact 必须含 252 层（expert 126 + PaliGemma 126）；是 126 就重跑转换器
python3 -c "import torch; a=torch.load('$HOME/lmy/openpi/third_party/flashrt/pi05_long_e0m3.pt', map_location='cpu', weights_only=True, mmap=True); print(len(a['weights']))"
```

### 1. action cos 数值门禁

harness：`tools/check_omega_e0m3_action_cos.py`，bf16 / fake-quant / E0M3 三模式
各一个子进程，同 fixture 同噪声逐条配对。四类指标定义对齐
`tests/bench_pi05_decoder_fp4_e2e.py`。

```bash
# x86 host：录 policy 路径 fixture（libero_10 十任务各 1 条；只需录一次）。
# 注意 FlashRT 那份 libero_obs_2v_n8.npz 的 state 是 joint_pos+gripper 布局，
# 不能直接喂 policy（policy 要 eef_pos+axisangle+gripper 的 8 维）
cd ~/pynoob/openpi
PYTHONPATH=third_party/libero:packages/openpi-client/src MUJOCO_GL=egl \
  examples/libero/.venv/bin/python \
  third_party/flashrt/tools/check_omega_e0m3_action_cos.py \
  --record-fixture /tmp/pi05_libero10_obs_n10.npz
rsync -avP /tmp/pi05_libero10_obs_n10.npz hcclab@10.191.163.226:~/lmy/openpi/

# Thor（三子进程共占一张卡串行跑；fake 模式无缓存、PaliGemma 126 层现场算
# SVD pack，建议 tmux 里跑）。fixture 落在 ~/lmy/openpi/ 即容器内 /workspace：
sudo docker run --rm -it --runtime nvidia --network host \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$HOME/lmy/openpi":/workspace -v "$HOME/lmy/Omega-QVLA":/opt/omega \
  -v "$HOME/.cache/openpi":/root/.cache/openpi \
  -w /workspace openpi-pi0.5:l4t-jp7.2 \
  bash -c "TF_DIR=/usr/local/lib/python3.12/dist-packages/transformers && \
           cp -r src/openpi/models_pytorch/transformers_replace/* \$TF_DIR/ && \
           export PYTHONPATH=packages/openpi-client/src:src:.:/opt/omega:/workspace/third_party/flashrt && \
           python -u third_party/flashrt/tools/check_omega_e0m3_action_cos.py \
             --checkpoint /root/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch \
             --pack /opt/omega/packs_hf/pi05_long/quantized.pt \
             --artifact /workspace/third_party/flashrt/pi05_long_e0m3.pt \
             --fixture /workspace/pi05_libero10_obs_n10.npz \
             --output-dir /workspace/eval_out/action_cos_routeA"
```

子进程日志 `eval_out/action_cos_routeA/{bf16,fake,e0m3}.log`，汇总 `result.json`。
门槛沿用 NVFP4 bench（action cos 0.999 / min-sample 0.995 / raw 0.995/0.995），
需要硬门禁时加 `--gate`。

### 2. server + 成功率评测

```bash
bash ~/lmy/openpi/third_party/flashrt/tools/start_e0m3_server.sh
sudo docker logs -f pi05_server
```

启动后必查：

1. `[OMEGA-E0M3] installed: artifact=... (252 layers)`——monkeypatch 生效
2. `[OMEGA-E0M3] cuda graph: installed` → 首次推理后 `prefix graph: captured` +
   `cuda graph: captured (prefix_len=968, layers=18, steps=10)`；
   `DISABLED` 字样 = 对应图回 eager，把括号里的异常贴出来排查

client（x86 host，同臂 A 入口，只改输出目录）：

```bash
cd ~/pynoob/openpi
setsid nohup bash -c '
  source examples/libero/.venv/bin/activate &&
  export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero &&
  python examples/libero/main.py \
    --args.task-suite-name libero_10 \
    --args.num-trials-per-task 1 \
    --args.host 10.191.163.226 --args.port 8000 \
    --args.video-out-path eval_out/omega_routeA_smoke
' > eval_out/omega_routeA_smoke.log 2>&1 < /dev/null &
# 冒烟 10 集成功率不离谱后 --args.num-trials-per-task 50
# （输出 eval_out/omega_routeA_long，日志 eval_out/omega_routeA_long.log）
```

### 3. per-inference 延迟 bench

`tools/bench_omega_e0m3_infer.py`：墙钟包住整个 `policy.infer`，内层 `infer_ms`
（模型段）一并打印；输出带 `graph_state` 自证图模式。每次调用 `noise=None`
（传噪声会静默回退 eager）。先停 server 再跑：

```bash
sudo docker rm -f pi05_server
sudo docker run --rm -it --runtime nvidia --network host \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$HOME/lmy/openpi":/workspace -v "$HOME/lmy/Omega-QVLA":/opt/omega \
  -v "$HOME/.cache/openpi":/root/.cache/openpi \
  -w /workspace openpi-pi0.5:l4t-jp7.2 \
  bash -c "TF_DIR=/usr/local/lib/python3.12/dist-packages/transformers && \
           cp -r src/openpi/models_pytorch/transformers_replace/* \$TF_DIR/ && \
           export PYTHONPATH=packages/openpi-client/src:src:.:/opt/omega:/workspace/third_party/flashrt && \
           python -u third_party/flashrt/tools/bench_omega_e0m3_infer.py \
             --checkpoint /root/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch \
             --pack /opt/omega/packs_hf/pi05_long/quantized.pt \
             --artifact /workspace/third_party/flashrt/pi05_long_e0m3.pt \
             --fixture /workspace/pi05_libero10_obs_n10.npz"
# 参考臂：追加 --eager（不装图）
```

`--fa4` 臂：Gemma 注意力走 FlashRT vendored FA4 kernel（`tools/omega_fa4_attention.py`）。
`--fa4` 强制 `--eager`（mask→索引是 host 侧操作，破 CUDA graph 捕获）。
`CUTE_DSL_ARCH=sm_101a` 必须在进程启动前 export（dsl 4.5.1 的 sm_110a 默认路径
触发 NVVM bug；原因见 `analysis.md` FA4 节）。完整三跑 + 数值门禁：

```bash
# 每次 docker run 的 bash -c 内、python 之前追加（容器是临时的）：
pip install -q nvidia-cutlass-dsl==4.5.1 quack-kernels==0.4.1
export CUTE_DSL_ARCH=sm_101a

# 跑 1 图基线：上文原命令不动
# 跑 2 eager 参考臂：追加
#   --eager --save-actions /workspace/eval_out/bench_eager_actions.npy
# 跑 3 FA4 臂：追加（自动带上 --eager）
#   --fa4 --save-actions /workspace/eval_out/bench_fa4_actions.npy

# 数值门禁（跑 3 vs 跑 2，逐 iter 余弦；容器退后在 Thor 宿主跑，
# 只依赖 numpy，无需激活项目 venv）：
python - <<'EOF'
import os
import numpy as np
a = np.load(os.path.expanduser("~/lmy/openpi/eval_out/bench_eager_actions.npy"))
b = np.load(os.path.expanduser("~/lmy/openpi/eval_out/bench_fa4_actions.npy"))
cos = (a*b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
print(f"FA4 vs eager action cos: min={cos.min():.5f} mean={cos.mean():.5f}")
EOF
```

### 4. nsys kernel 级分解

eager 一跑归因最干净；CUPTI 有采集开销，只看 kernel 占比、不看绝对墙钟。
产物落宿主 `perf_data/`：

```bash
sudo docker rm -f pi05_server  # 先停 server
sudo docker run --rm -it --runtime nvidia --network host \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$HOME/lmy/openpi":/workspace -v "$HOME/lmy/Omega-QVLA":/opt/omega \
  -v "$HOME/.cache/openpi":/root/.cache/openpi \
  -w /workspace openpi-pi0.5:l4t-jp7.2 \
  bash -c "TF_DIR=/usr/local/lib/python3.12/dist-packages/transformers && \
           cp -r src/openpi/models_pytorch/transformers_replace/* \$TF_DIR/ && \
           export PYTHONPATH=packages/openpi-client/src:src:.:/opt/omega:/workspace/third_party/flashrt && \
           mkdir -p perf_data && \
           nsys profile -o perf_data/omega_e0m3_bench_eager --force-overwrite=true \
             -t cuda \
             python -u third_party/flashrt/tools/bench_omega_e0m3_infer.py \
               --checkpoint /root/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch \
               --pack /opt/omega/packs_hf/pi05_long/quantized.pt \
               --artifact /workspace/third_party/flashrt/pi05_long_e0m3.pt \
               --fixture /workspace/pi05_libero10_obs_n10.npz --eager && \
           nsys stats perf_data/omega_e0m3_bench_eager.nsys-rep \
             --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum \
             --report cuda_api_sum --format csv --force-export=true \
             -o perf_data/omega_e0m3_bench_eager"
```

读 `*_cuda_gpu_kern_sum.csv` 按 kernel 名归堆。去掉 `--eager` 同法可采图模式对照
（graph 重放内 kernel 在 nsys 里仍逐条可见）。DRAM/EMC 侧要补 tegrastats 的话沿用
`deployment_scripts/collect_perf_data.sh` 的嵌套采法。

### 5. 融合 glue kernel 重编（消费层 .so 更新后）

消费层的 perm+旋转+cast+量化 / 旋转+cast+bias 融合 kernel 在
`csrc/quantize/quantize_e0m3_duquant.cu`。`.so` 在挂载树
`third_party/flashrt/flash_rt/` 里，源码更新后在容器内重编：

```bash
# 先确认 cutlass 是挂载树里的实体目录（v4.4.2；不是指向 /opt/cutlass 的死链——
# 那是 flashrt 镜像里的路径）：
ls ~/lmy/openpi/third_party/flashrt/third_party/cutlass/include/cutlass/cutlass.h

# 现存 build/ 树若是在宿主机上配置的（cache 记宿主路径 + 失效的 venv cmake），
# 容器内会报 "CMakeCache.txt directory ... is different"——抹掉重配即可；
# 只编 fp4 模块（消费层只需要它），别全量：
sudo docker run --rm -it --runtime nvidia \
  -v "$HOME/lmy/openpi":/workspace -w /workspace/third_party/flashrt \
  openpi-pi0.5:l4t-jp7.2 \
  bash -c "rm -rf build && cmake -B build -S . -DGPU_ARCH=110 && \
           cmake --build build --target flash_rt_fp4 -j\$(nproc)"

# 重编后自检（应打印 True）：
sudo docker run --rm --runtime nvidia -v "$HOME/lmy/openpi":/workspace \
  -w /workspace openpi-pi0.5:l4t-jp7.2 \
  bash -c 'export PYTHONPATH=/workspace/third_party/flashrt && \
           python -c "import flash_rt.flash_rt_fp4 as f; print(hasattr(f, \"quantize_e0m3_duquant_sfa_bf16\"))"'
```

重编前代码自动走原 PyTorch glue 路径（`hasattr` 探测，行为不变）。验证顺序：
`tools/check_omega_e0m3_layer.py` 单层对照（`--pack <pack> --mode kernel --tokens 256`）
→ §1 action cos 门禁 → §3 bench。

### 6. 消费层/服务脚本单文件同步（迭代期）

```bash
rsync -avz ~/pynoob/openpi/third_party/flashrt/tools/{omega_e0m3_linear,serve_omega_e0m3}.py \
  hcclab@10.191.163.226:~/lmy/openpi/third_party/flashrt/tools/
```

稳定后走 git（§0）。

---

## MIP 2-step × W4A4 QAT（Much-ado-about-noising 移植）

实现与约定映射见 `src/openpi/models_pytorch/mip_qat.py` 头部 docstring（测试：
`mip_qat_test.py`）。t* = 0.9；推理 = `PI05_T_GRID="1.0:-1.0;0.1:-0.1"` + 零噪声起步
（serve_omega_e0m3.py 在 T_GRID 设置时自动开 `OMEGA_E0M3_ZERO_NOISE` 并跳过 10 步抓图）。

**1. 训练**（笔记本 `.venv-train`，torch cu128 / sm_120；主 .venv 的
cu126 在 RTX 5060 上无 kernel 可用）。checkpoint 从 Thor rsync（GCS 只有 JAX 版）：

```bash
rsync -avzP hcclab@10.191.163.226:~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch \
  ~/.cache/openpi/openpi-assets/checkpoints/

cd ~/pynoob/openpi
HF_ENDPOINT=https://hf-mirror.com .venv-train/bin/python scripts/train_mip_qat.py \
  --exp_name mip_qat_v1 --batch_size 8 --num_train_steps 5000 --qat
```

冻结 paligemma（ViT+LLM），只训 action expert + 投影（~300M，PagedAdamW8bit）。
OOM 回退：batch 8→4→2。checkpoints 在 `checkpoints/<exp_name>/<step>/model.safetensors`。

**2. 导出 E0M3 artifact**（Thor，kernel 依赖；checkpoint 先 rsync 过去）：

```bash
rsync -avzP ~/pynoob/openpi/checkpoints/mip_qat_v1/5000/model.safetensors \
  hcclab@10.191.163.226:~/lmy/openpi/checkpoints/mip_qat_v1/5000/
# Thor 上
cd ~/lmy/openpi/third_party/flashrt
python tools/convert_qat_ckpt_e0m3.py \
  --ckpt ~/lmy/openpi/checkpoints/mip_qat_v1/5000/model.safetensors \
  --out pi05_mip2step_e0m3.pt --keep-fp16
```

**3. Thor 起 2 步服务**：QAT checkpoint 需组装成完整服务目录（模型权重换掉、
assets/norm stats 沿用 base）：

```bash
# Thor 上
rsync -a ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch/ \
  ~/ckpts/pi05_mip_qat/
cp ~/lmy/openpi/checkpoints/mip_qat_v1/5000/model.safetensors ~/ckpts/pi05_mip_qat/

OMEGA_E0M3_PACK=/workspace/third_party/flashrt/pi05_mip2step_e0m3.pt \
PI05_T_GRID="1.0:-1.0;0.1:-0.1" \
E0M3_MODEL_PATH=/root/ckpts/pi05_mip_qat \
  bash ~/lmy/openpi/third_party/flashrt/tools/start_e0m3_server.sh
```

（`E0M3_MODEL_PATH` 透传若脚本未支持则手改 --model_path；启动日志应见
`zero-noise sampling installed (MIP mode)` + `cuda graph: skipped`。）

**4. 评测**：冒烟 10 集 → LIBERO-10 ×500（命令同路线 A §2 client，改输出目录
`omega_mip2step_*`）。

---

## 参考

- [OpenPi π₀.₅ on Jetson Thor | Jetson AI Lab](https://www.jetson-ai-lab.com/tutorials/openpi_on_thor/)
- [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)（上游，分叉点 `15a9616`）
