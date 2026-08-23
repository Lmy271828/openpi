# π₀.5 on Jetson AGX Thor 部署与性能复现手册

> 目标：复现 π₀.₅（`pi05_libero`）在 Thor 上的 TensorRT FP8/NVFP4 部署、推理性能数据、
> LIBERO-Long 三臂成功率对照、task4/9 探针分析与 LIBERO-Plus 鲁棒性评测。
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

分支内容 = 上游 `15a9616`（教程验证基线）+ 部署脚本、TRT serve 支持、性能采集工具、
LIBERO 评测工具（`--task-ids`、探针、视频归档器）与 LIBERO-Plus/FlashRT submodule。

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

与教程的差异及原因：

- `--cap-add SYS_ADMIN`：GPU metrics 采样需要，否则 nsys 拒绝（Insufficient privilege）
- `--network host`：替代 `-p 8000:8000`，方便 server 与 tegrastats 时间对齐
- `-v /sys:/sys:ro`：tegrastats 的 EMC/GR3D 读自 sysfs；只挂二进制会静默省略这两个字段
- `.cache` 挂载保留 checkpoint（~6 GB）、HF 数据集、ONNX/引擎产物

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

补充：

1. Step 9 的校准数据自动从 HuggingFace 下载 `physical-intelligence/libero`。
   网络受限时 `export HF_ENDPOINT=https://hf-mirror.com`。
2. 教程警告 FP16 会在 Gemma attention 溢出；本分支实测 fp16 引擎（臂 B）
   cosine = 0.99999823、成功率 93.0% 无掉点——该警告不适用于当前导出脚本，
   fp16 导出流程见 Step 6「臂 B 引擎」。
3. 引擎构建产物（`*_profile.json` / `*_layers.json` / `.log`）在引擎同目录，
   Step 4 的采集脚本会自动拷贝。

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
<prefix>_emc.log          # tegrastats（EMC%/GR3D），已内嵌推理脚本，时间轴与 nsys 对齐
<prefix>_console.log
```

DRAM 带宽说明：Thor 是 Tegra 统一内存，nsys GPU metrics 不含 DRAM 计数器
（DRAM 在 SoC 侧 EMC），由 tegrastats 补齐，无需单独采集。判读参考：EMC% 稳态
>80% 才是带宽瓶颈；本模型 SM Issue 仅 ~9–20%，属内存延迟受限。

拷回 host：

```bash
scp hcclab@10.191.163.226:~/lmy/openpi/perf_data/* perf_data/
```

---

## Step 5：host 侧分析

```bash
python deployment_scripts/analyze_perf.py --perf-dir perf_data --dram-peak-gbps 273
```

| 段 | 数据源 | 内容 |
|---|---|---|
| num_steps 扫描 | `numsteps_sweep_*.csv` | `T(N) ≈ T_fixed + N·T_step` 线性分解 |
| DRAM / 显存带宽 | `*_gpumetrics.csv` / `.sqlite` | Copy Engine 代理 + S2/S3 分阶段饱和度 |
| CUDA graph 使用 | `<prefix>.sqlite` 直读 | 稳态 graph replay 统计 |
| nsys kernel 分析 | `*_cuda_gpu_kern_sum.csv` | kernel 分类汇总 |
| trtexec 逐层 profile | `*_profile.json` | ViT / LLM(NVFP4) / expert(FP8) 阶段归因 |

注意：torch.compile 与 TRT 稳态计算在 CUDA graph 内，kernel 级 CSV 为空属预期
（此 nsys 版本不归因 graph 内 kernel）；逐 kernel 分析用 Eager 对照组或 trtexec profile。

---

## Step 6：LIBERO-Long 成功率评测

LIBERO-Long 即 `libero_10` 任务套件。三臂对照分离"转换误差"与"量化误差"：

| 臂 | 精度 | 推理路径 | 作用 |
|---|---|---|---|
| A | BF16（不量化） | PyTorch eager + torch.compile | 基准 |
| B | FP16（不量化） | ONNX → TRT 引擎 | A − B = 转换误差 |
| C | FP8 + NVFP4（量化） | ONNX（QDQ）→ TRT 引擎 | B − C = 量化误差 |

实测结果见 `analysis.md`（A 91.6% / B 93.0% / C 80.0%，延迟 137 / 85.8 / 49.9 ms）。

### server（Thor 宿主机，detached 容器）

server 必须脱离终端运行：交互式容器里的 server 会随 SSH 断开被杀，client 不重连，
之后每个 episode 首步 infer 即失败、记为垃圾失败（一帧的 failure 视频是特征）。

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

预检通过标准：cosine ≥ 0.999（实测 0.99999823）。不达标就回去修导出，
不要带着坏引擎跑 5 小时 LIBERO。

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
参考系（MAXN，num_steps=10）：PyTorch BF16 = 137 ms，TRT fp16 = 85.8 ms，
TRT FP8/NVFP4 = 49.9 ms（引擎图优化 1.60× × 量化 1.72× = 2.75×）。

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

### task4 / task9 探针分析

探针（`--args.probe`）在每集结束时记录任务物体的世界系坐标（cm，带符号）与关节
qpos，用于把"成功率掉点"拆解为几何量偏移。前置：server 为目标臂；config 必须是
**原版 libero**（`get_libero_path` 每个任务重读 `~/.libero/config.yaml`，
Plus 评测运行中切 config 会把正在跑的 client 搞崩）：

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

判读：

- task4（双杯放盘）：判定器 `On(mug, plate)` = 接触 + 杯盘中心 XY 距离 < 3cm。
  从探针坐标离线算杯-盘 XY 偏移，成功/失败两组的分布若骑在 3cm 上，
  即"落点偏心被阈值放大"（实测：成功中位 2.7cm，失败中位 3.7cm，80% 失败为 3-5cm 擦边）
- task9（杯子进微波炉并关门）：判定器 `And(In(mug, heating_region), Close(microwave))`。
  杯坐标验证 `In`，`microwave_1_microjoint`（门铰链角度，开门 ≈ -1.55 rad）验证 `Close`，
  失败可拆为"没放入" / "没关门" / "两者都没"
- 逐集视频在 `archived/` 子目录（归档器按完成时间戳改名，避免同结局互相覆盖）

### LIBERO-Plus 鲁棒性评测（7 扰动维度 × 30 任务子集）

环境独立于原版：专用 venv `examples/libero/.venv-plus`（libero 指向
`third_party/libero_plus`）。`~/.libero/config.yaml` 的双份备份首次使用前创建：

```bash
cp ~/.libero/config.yaml ~/.libero/config.yaml.libero            # 原版备份（只需一次）
cp ~/.libero/config.yaml.libero_plus ~/.libero/config.yaml       # 切到 plus
cp ~/.libero/config.yaml.libero ~/.libero/config.yaml            # 切回原版
```

评测子集 `eval_out/libero_plus_subset.json`：seed=42，7 维度 × 30 任务 = 210，
按难度 1-5 分层。两臂用同一子集、每任务 1 trial，逐任务配对。

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

- 先冒烟（`--args.task-ids` 只给前 3 个 id），确认 assets 加载正常再全量
- 判读：按维度分组比较 A vs C（每维度 n=30，CI≈±18%，看方向和配对比）
- 扰动编码在任务名里：`view_水平_俯仰_缩放x100_旋转_俯仰_initstate_<id>_noise_<seed>`；
  view 段非 `0_0_100_0_0` 才是相机位姿扰动（该编码被 Camera/Noise/InitState 三维共用）

### 判读规则

- 固定 seed（默认 7）+ 固定初始状态，两臂 episode 逐对配对
- 每套件 n = 500 episodes，二项 95% CI ≈ ±4.4%；ΔSR < ±5% 只能下"无显著差异"结论
- 配对用 McNemar 检验（只看一成一败的对子）
- flow matching 的 noise 每次推理随机采样，单 episode 成败不可比，必须靠样本量
- 迭代期可用 `--args.num-trials-per-task 10` 冒烟，最终结论必须 50

---

## FlashRT 复现环境（third_party/flashrt submodule）

FlashRT（自研 CUDA kernel + 静态 CUDA Graph runtime，非 TensorRT）作为推理引擎
对照组，钉在 `2035406`（2026-08-13 main）：

```bash
git submodule add https://github.com/flashrt-project/FlashRT.git third_party/flashrt

git -c protocol.file.allow=always submodule add ~/pynoob/FlashRT third_party/flashrt
git config -f .gitmodules submodule.third_party/flashrt.url \
  https://github.com/flashrt-project/FlashRT.git
git -c protocol.file.allow=always submodule sync third_party/flashrt
```

（第一条是网络正常时；后三条是离线时用本地克隆做源再改回规范 URL。）

组织原则：

- 不改 FlashRT 源码，适配层全部放本仓库；升级用 `git submodule update --remote`
- 运行环境独立于 openpi 容器（Thor 宿主机专用 venv；它要编译 SM110 kernel），
  装完先跑最简 smoke（加载 checkpoint + 单条推理）
- 评测拓扑：写适配 server（`deployment_scripts/flashrt_serve.py`，FlashRT 对内、
  openpi-client websocket 协议对外），不用它自带的单进程 eval_libero.py——
  保证与 A/B/C 同 seed 逐集配对，唯一变量是推理引擎
- 待验证：能否直接吃 `pi05_libero_pytorch` 转换后 checkpoint
- 复现计划：P1 = FP8 on libero_10 10×50（参照值 92.6-93.0%）；
  P2 = NVFP4+AWQ `use_fp4`；跑时停掉 pi05_server 避免 GPU 抢占
- 引用其文档数字前必须自己复现（README 与 examples/thor/README 有不一致）

---

## Omega-QVLA 复现（臂 D：W4A4 GPTQ + DuQuant）

Omega-QVLA 是 openpi 的**兄弟目录独立克隆，不进 submodule**：

- 本地 `~/pynoob/Omega-QVLA`，Thor `~/lmy/Omega-QVLA`，用 rsync 同步：
  `rsync -avz --exclude=.git ~/pynoob/Omega-QVLA hcclab@10.191.163.226:~/lmy/`
- 不进 submodule 的原因：不改它的源码（只调用），且 `packs_hf/` 里的 GPTQ pack
  （`pi05_long/quantized.pt`，数百 MB）来自 HF 下载，不入 git
- 集成方式：整个目录挂进 openpi 容器 `/opt/omega`，靠 `PYTHONPATH` + `GR00T_*`
  环境变量生效。`scripts/openpi_inference_service.py`
  是标准 openpi websocket server，启动时给 `policy._model` 套量化 Linear：
  action expert 用预构建 GPTQ pack（W4A4 + svd_hadamard + per-step scale 表，
  按 10 步 denoise 构建），PaliGemma 主干用 DuQuant 运行时校准（前 ~32 batch 收 scale）
- openpi 侧唯一改动：`src/openpi/models_pytorch/pi0_pytorch.py` 的
  `sample_actions` 循环进入 `gr00t.quantization.dit_step_context.set_dit_quant_step(t)`
  （guarded import，无 Omega-QVLA 时为 no-op）。不打这个补丁则 per-step scale 表
  不生效，GptqLinear 退化为 step-mean scale——Omega-QVLA 官方的 pack 构建和评测
  都依赖他们本地打过补丁的 openpi，仓库里没有携带该补丁

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

1. `[GR00T-GPTQ] Matched Linear layers: 126`（expert 18 层 × 7 投影）且
   `[GR00T-DUQUANT] Matched Linear layers: 126`（PaliGemma 18 层 × 7 投影）；
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
（输出 `eval_out/omega_w4a4_long`）。注意 GptqLinear 是 fake-quant 仿真
（bf16 稠密 GEMM），本轮只验证精度，不做延迟对照。

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

### Omega-QVLA × FlashRT E0M3（M2：fake-quant 换 tcgen05 kernel）

把 GptqLinear 的 fake-quant matmul 换成 FlashRT E0M3 真 kernel（转换产物 +
消费层，格式与设计见 `third_party/flashrt/docs/omega_pack_e0m3.md`）。
前置：artifact `pi05_long_e0m3.pt` 已在 Thor `third_party/flashrt/` 下
（转换器产出，见该文档 Reproducing），消费层/入口脚本已 rsync 到 Thor：

```bash
# 本地执行（迭代期单文件同步；稳定后走 git）
rsync -avz ~/pynoob/openpi/third_party/flashrt/tools/{omega_e0m3_linear,serve_omega_e0m3}.py \
  hcclab@10.191.163.226:~/lmy/openpi/third_party/flashrt/tools/
```

**1. 探路**（容器能否 import 宿主机 venv 构建的 flash_rt_fp4，ABI 验证）：

```bash
sudo docker run --rm --runtime nvidia \
  -v "$HOME/lmy/openpi":/workspace \
  openpi-pi0.5:l4t-jp7.2 \
  bash -c 'export PYTHONPATH=/workspace/third_party/flashrt && \
           python -c "import torch; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available()); import flash_rt.flash_rt_fp4 as m; print(\"flash_rt_fp4 OK\")"'
```

打 `flash_rt_fp4 OK` 才能继续；报 undefined symbol / ABI 错则改走容器内
`pip install /workspace/third_party/flashrt` 构建。

**2. 起服务**（与 hybrid server 命令只差三处：PYTHONPATH 加 flashrt 仓库根
+ venv site-packages、加 `OMEGA_E0M3_PACK`、python 行换入口脚本）：

```bash
sudo docker stop pi05_server 2>/dev/null; sudo docker rm pi05_server 2>/dev/null

sudo docker run -d --name pi05_server --runtime nvidia \
  --cap-add SYS_ADMIN --network host \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$HOME/lmy/openpi":/workspace \
  -v "$HOME/lmy/Omega-QVLA":/opt/omega \
  -v "$HOME/.cache/openpi":/root/.cache/openpi \
  -w /workspace \
  openpi-pi0.5:l4t-jp7.2 \
  bash -c "TF_DIR=/usr/local/lib/python3.12/dist-packages/transformers && \
           cp -r src/openpi/models_pytorch/transformers_replace/* \$TF_DIR/ && \
           export PYTHONPATH=packages/openpi-client/src:src:.:/opt/omega:/workspace/third_party/flashrt && \
           export OMEGA_E0M3_PACK=/workspace/third_party/flashrt/pi05_long_e0m3.pt && \
           export GR00T_GPTQ=1 \
                  GR00T_GPTQ_PATH=/opt/omega/packs_hf/pi05_long/quantized.pt \
                  GR00T_GPTQ_INCLUDE='.*paligemma_with_expert\.gemma_expert\.model\.layers\.[0-9]+\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj).*' \
                  GR00T_GPTQ_WBITS_DEFAULT=4 GR00T_GPTQ_ABITS=4 GR00T_GPTQ_MISSING=fallback \
                  GR00T_DUQUANT_INCLUDE='.*paligemma_with_expert\.paligemma\.model\.language_model\.layers\.[0-9]+\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj).*' \
                  GR00T_DUQUANT_ROT_MODE=svd_hadamard GR00T_DUQUANT_PERM_SCORE=weight \
                  GR00T_DUQUANT_BLOCK=64 GR00T_DUQUANT_BLOCK_OUT=64 \
                  GR00T_DUQUANT_PERMUTE=1 GR00T_DUQUANT_ROW_ROT=restore \
                  GR00T_DUQUANT_ACT_PCT=99.9 GR00T_DUQUANT_CALIB_STEPS=32 GR00T_DUQUANT_LS=0.15 && \
           python -u /workspace/third_party/flashrt/tools/serve_omega_e0m3.py \
             --model_path /root/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch \
             --data_config pi05_libero --port 8000"

sudo docker logs -f pi05_server
```

**2b. 一键脚本 `tools/start_e0m3_server.sh`**（在 fork 仓库内，进 git、
随 rsync 同步——取代早期散落在 Thor home 的 `~/start_e0m3_server.sh`，
后者可删）。路径从脚本自身位置推算，布局不同可用 env 覆盖
（`OPENPI_ROOT` / `OMEGA_ROOT` / `OPENPI_CACHE`）：

```bash
# Thor 上（rsync 或 git pull 之后）
bash ~/lmy/openpi/third_party/flashrt/tools/start_e0m3_server.sh
OMEGA_E0M3_CUDA_GRAPH=1 bash ~/lmy/openpi/third_party/flashrt/tools/start_e0m3_server.sh  # 抓图模式
sudo docker logs -f pi05_server
```

与 §2 裸命令的差异（docker run 本体逐字相同）：

1. 前置 `docker stop/rm || true`——幂等重启，不用手清旧容器
2. `set -euo pipefail`——起失败立即非零退出
3. `OMEGA_E0M3_CUDA_GRAPH=${OMEGA_E0M3_CUDA_GRAPH:-1}` 透传——默认开
   （50 集 45/50 = 90.0% ≈ eager 基线 90.4%，已验收），显式 `=0` 回 eager
4. 不带 `logs -f`——日志单独 `sudo docker logs -f pi05_server`
5. mount 源用 `OPENPI_ROOT`/`OMEGA_ROOT`/`OPENPI_CACHE` 变量，默认
   `$HOME/lmy/...` 布局

启动后必查三项：

1. `[OMEGA-E0M3] installed: artifact=... (252 layers)`——monkeypatch 生效
2. `[GR00T-GPTQ][REPLACED] ...` 照常打印（wrap 流程不变，换的是类）
3. `start_e0m3_server.sh` 默认 `OMEGA_E0M3_PATCH_DUQUANT=1`（路线 A：
   双侧全换，PaliGemma 吃 pack 的 GPTQ W4A4 记录）；显式
   `OMEGA_E0M3_PATCH_DUQUANT=0` 回旧的 expert-only 半场对照

抓图模式（`OMEGA_E0M3_CUDA_GRAPH=1`）再加一项：

4. `sudo docker logs pi05_server 2>&1 | grep "OMEGA-E0M3] cuda graph"`——
   期望 `installed` → 首次推理时 `captured (prefix_len=968, layers=18,
   steps=10)`；`OMEGA_E0M3_PREFIX_GRAPH=1`（默认）时还应有
   `prefix graph: captured (prefix_len=968, layers=18)`。
   `prefix graph: DISABLED` 只丢 prefix 图（prefix 回 eager，denoise 图
   仍在）；`cuda graph: DISABLED` 才是全回退 eager，把括号里的异常
   贴回来排查

**3. 冒烟**（client 同臂 A 入口，只改输出目录；对照基准：
omega_w4a4_smoke 9/10，且每集耗时应显著低于 fake-quant 的 ~148s）：

```bash
cd ~/pynoob/openpi
setsid nohup bash -c '
  source examples/libero/.venv/bin/activate &&
  export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero &&
  python examples/libero/main.py \
    --args.task-suite-name libero_10 \
    --args.num-trials-per-task 1 \
    --args.host 10.191.163.226 --args.port 8000 \
    --args.video-out-path eval_out/omega_e0m3_smoke
' > eval_out/omega_e0m3_smoke.log 2>&1 < /dev/null &
```

成功率不离谱且延迟下降后，`--args.num-trials-per-task 50` 全量
（输出 `eval_out/omega_e0m3_long`），与臂 A（91.6%）/ 臂 D（93.2%）
逐集配对。

### 路线 A：双侧 E0M3 + prefix 抓图（action cos + 端到端延迟）

PaliGemma 从运行时 DuQuant fake-quant（RTN W4 + 默认 A8）切到 pack 的
GPTQ W4A4 记录 + E0M3 kernel——Omega 官方 pack 配方本来就是全模型 W4A4
（pack 内 PaliGemma 记录 `a_bits=4`，2026-08-22 直接读 pack 核实），
所以路线 A 是对齐官方配方，同时把 PaliGemma 权重从 RTN 升级为 GPTQ。
prefix prefill 同步入图（`omega_e0m3_graph.py` 双图）。转换器零改动
（252 条记录全量转换，无侧区分支）；服务端两个开关在
`tools/start_e0m3_server.sh` 已默认开：`OMEGA_E0M3_PATCH_DUQUANT=1`、
`OMEGA_E0M3_PREFIX_GRAPH=1`。

**0. 同步与 artifact 检查**（Thor）：

```bash
rsync -avz ~/pynoob/openpi/third_party/flashrt/tools/ \
  hcclab@10.191.163.226:~/lmy/openpi/third_party/flashrt/tools/
# artifact 必须含 252 层（expert 126 + PaliGemma 126）；是 126 就重跑转换器
python3 -c "import torch; a=torch.load('$HOME/lmy/openpi/third_party/flashrt/pi05_long_e0m3.pt', map_location='cpu', weights_only=True, mmap=True); print(len(a['weights']))"
```

**1. action cos**（四类指标：action cos / action min-sample cos /
raw cos / raw min-sample cos，对齐 `tests/bench_pi05_decoder_fp4_e2e.py`
的定义；raw = unnormalize 前的归一化 action chunk）。harness：
`tools/check_omega_e0m3_action_cos.py`，bf16 / fake-quant（臂 D 配方）/
E0M3（路线 A）三模式各一个子进程，同 fixture 同噪声逐条配对。

```bash
# x86 host：录 policy 路径 fixture（libero_10 十任务各 1 条，已在本机验证）。
# 注意 FlashRT 那份 libero_obs_2v_n8.npz 的 state 是 joint_pos+gripper 布局，
# 不能直接喂 policy（policy 要 eef_pos+axisangle+gripper 的 8 维）
cd ~/pynoob/openpi
PYTHONPATH=third_party/libero:packages/openpi-client/src MUJOCO_GL=egl \
  examples/libero/.venv/bin/python \
  third_party/flashrt/tools/check_omega_e0m3_action_cos.py \
  --record-fixture /tmp/pi05_libero10_obs_n10.npz
rsync -avP /tmp/pi05_libero10_obs_n10.npz hcclab@10.191.163.226:~/lmy/openpi/

# Thor（三子进程共占一张卡串行跑，含 40 次 warmup 让 fake 模式的在线
# DuQuant 校准收敛；fake 模式无 PACKDIR 缓存、PaliGemma 126 层现场算
# SVD pack，建议 tmux 里跑）。fixture 落在 ~/lmy/openpi/ 即容器内
# /workspace，无需额外挂载 /tmp：
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

判读：`e0m3_vs_fake` 是 kernel 化纯损耗（应显著高于另两对，单层 S0
已验证不劣于 fake-quant 自身）；`fake_vs_bf16` 是配方自身保真参考线；
`e0m3_vs_bf16` 是端到端总账。参考门槛沿用 NVFP4 bench（action cos
0.999 / min-sample 0.995 / raw 0.995/0.995），首轮先不设硬门禁，
需要时加 `--gate`。子进程日志 `eval_out/action_cos_routeA/{bf16,fake,e0m3}.log`
（Thor 宿主机路径，即容器内 `/workspace/eval_out/action_cos_routeA/`），
汇总 `result.json`；各模式 eager p50 推理耗时一并打印（含 prefix）。

**2. 端到端延迟 + 成功率**（路线 A 默认配置）：

```bash
bash ~/lmy/openpi/third_party/flashrt/tools/start_e0m3_server.sh
sudo docker logs -f pi05_server
```

冒烟 → 全量（client 同臂 A 入口，只改输出目录）：

```bash
cd ~/pynoob/openpi
setsid nohup bash -c '
  source examples/libero/.venv/bin/activate &&
  export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero &&
  python examples/libero/main.py \
    --args.task-suite-name libero_10 --args.num-trials-per-task 1 \
    --args.host 10.191.163.226 --args.port 8000 \
    --args.video-out-path eval_out/omega_routeA_smoke
' > eval_out/omega_routeA_smoke.log 2>&1 < /dev/null &
# 冒烟 10 集成功率不离谱后 --args.num-trials-per-task 50
# （输出 eval_out/omega_routeA_long，日志 eval_out/omega_routeA_long.log）
```

对照基线：臂 D fake-quant 93.2% / ~148s 每集；expert-only E0M3 90.4% /
43–50s（prefix eager）。实测（2026-08-23）：**93.8%**（vs A p=0.135 /
D p=0.749 无差异 / expert-only p=0.033 显著更优，task9 60%→94%），
每集均值 42.9s——与 expert-only 同量级；episode 长度混杂（成功率更高 ⇒
难任务跑得更长）使每集墙钟不能作延迟结论，精确延迟用下面的 bench。
逐任务表与判读见 `analysis.md` 路线 A 节。

**3. per-inference 延迟（臂 A 口径）**：`tools/bench_omega_e0m3_infer.py`——
墙钟包住整个 `policy.infer`（与臂 A/B/C 的 137.2/85.8/49.9ms 同口径），
内层 `infer_ms`（模型段）作诊断一并打印；输出带 `graph_state` 自证图
模式，没有它不算图模式数字。每次调用 `noise=None`（传噪声会静默回退
eager）。先停 server 再跑：

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

延迟口径速查（四组数字边界两两不同，禁止直接并排）：

| 数字 | 计时区间 | 不含 |
|---|---|---|
| 臂 A/B/C 137.2/85.8/49.9ms | 进程内 `policy.infer` 整次墙钟 | 网络 |
| FlashRT 27.21ms（NVFP4+FA4） | 进程内 `pipe.infer` 墙钟 | 网络、tokenize（prompt 预设） |
| server `policy_timing.infer_ms` | 仅 `sample_actions`（含 tokenize 与 GPU 执行；计时区已加 sync） | 输入 transform、D2H copy 本身、反归一化 |
| 评测每集耗时 | episode 墙钟 | —（混 episode 长度） |

`--fa4` 臂：Gemma 注意力（PaliGemma LM + action expert 一处覆盖）走
FlashRT vendored FA4 kernel（`tools/omega_fa4_attention.py`，从 additive
4D mask 还原有效 key 列后物理压实调用，`causal=False` 双向块语义）。
mask→索引是 host 侧数据相关操作、破 CUDA graph 捕获，故 `--fa4` 强制
`--eager`；图化 FA4 需按 prompt 长度逐长度抓图（见 `analysis.md`）。
容器需先装 thor-fa4 依赖，且 `CUTE_DSL_ARCH=sm_101a` 必须在进程启动前
export——dsl 4.5.1 的 sm_110a 默认路径触发 NVVM chip-string bug，shim
内的 setdefault 若晚于首次 `import cutlass` 则无效（踩坑记录见
`analysis.md` FA4 补齐节），不能依赖它。完整三跑 + 数值门禁：

```bash
# 每次 docker run 的 bash -c 内、python 之前追加（容器是临时的）：
pip install -q nvidia-cutlass-dsl==4.5.1 quack-kernels==0.4.1
export CUTE_DSL_ARCH=sm_101a

# 跑 1 图基线：上文原命令不动
# 跑 2 eager 参考臂：追加
#   --eager --save-actions /workspace/eval_out/bench_eager_actions.npy
# 跑 3 FA4 臂：追加（自动带上 --eager）
#   --fa4 --save-actions /workspace/eval_out/bench_fa4_actions.npy

# 数值门禁（跑 3 vs 跑 2，逐 iter 余弦；容器退后在 Thor 宿主跑）：
python - <<'EOF'
import numpy as np
a = np.load(f"/home/$USER/lmy/openpi/eval_out/bench_eager_actions.npy")
b = np.load(f"/home/$USER/lmy/openpi/eval_out/bench_fa4_actions.npy")
cos = (a*b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
print(f"FA4 vs eager action cos: min={cos.min():.5f} mean={cos.mean():.5f}")
EOF
```

**nsys kernel 级分解**（回答"per-inference 时间花在哪"）：eager 一跑归因
最干净；CUPTI 有采集开销，只看 kernel 占比、不看绝对墙钟。产物落宿主
`perf_data/`：

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

读 `perf_data/omega_e0m3_bench_eager_cuda_gpu_kern_sum.csv`，按 kernel 名
归堆：`cutlass_fp4_gemm_e0m3w`（FP4 GEMM）/ `quantize_e0m3*`（激活量化）/
bmm+index_select（DuQuant 置换+旋转 glue）/ SDPA（注意力）/ cublas bf16
大 GEMM（vision tower 与未量化残余）。去掉 `--eager` 同法可采图模式对照
（graph 重放内 kernel 在 nsys 里仍逐条可见）。DRAM/EMC 侧要补 tegrastats
的话沿用 `deployment_scripts/collect_perf_data.sh` 的嵌套采法。

**融合 glue kernel（quantize_e0m3_duquant）**：消费层的 perm+旋转+cast+
量化 / 旋转+cast+bias 已各融成单 kernel（`csrc/quantize/
quantize_e0m3_duquant.cu`，warp-per-64-block，数值复现 PyTorch 舍入链）。
`.so` 在挂载树 `third_party/flashrt/flash_rt/` 里，拉到该提交后在容器内
增量重编（几分钟）：

```bash
sudo docker run --rm -it --runtime nvidia \
  -v "$HOME/lmy/openpi":/workspace -w /workspace/third_party/flashrt \
  openpi-pi0.5:l4t-jp7.2 \
  bash -c "cmake --build build -j\$(nproc)"
# 若 cmake 报 CUTLASS 路径失效（build 树是在别的镜像里配的），先
# grep CUTLASS_DIR build/CMakeCache.txt 确认路径，必要时重跑
# cmake -B build -S . -DGPU_ARCH=110（需要镜像内有 cutlass 4.x 头）
```

重编前代码自动走原 PyTorch glue 路径（`hasattr` 探测，行为不变）。验证：
`tools/check_omega_e0m3_layer.py` 单层对照 → action cos 门禁 → bench。

### MIP 2-step × W4A4 QAT（Much-ado-about-noising 移植）

从 pi05_libero checkpoint 出发，用 MIP teacher-free 两步损失微调 +
expert W4A4 fake-quant QAT，目标 2 步推理。实现与约定映射见
`src/openpi/models_pytorch/mip_qat.py` 头部 docstring（测试：
`mip_qat_test.py`，5 项）。要点：

- MIP 两步规约为未修改的 `PI0Pytorch.forward(noise=..., time=...)` 两次调用：
  step1 `noise=0, time=1.0`（loss ×1/t*²），step2 `noise=act+η, time=1-t*`（不再除）
- t* = 0.9；推理 = `PI05_T_GRID="1.0:-1.0;0.1:-0.1"` + **零噪声起步**
  （serve_omega_e0m3.py 在 T_GRID 设置时自动开 `OMEGA_E0M3_ZERO_NOISE`
  并跳过 10 步抓图）
- QAT 数值对齐 E0M3 部署：权重 per-16 amax/7、激活 per-token 动态 amax、
  码值 [-7,7]、STE 直通；v1 不带 DuQuant 旋转（S0 语义）

**1. 训练**（笔记本 `.venv-train`，torch cu128 / sm_120；主 .venv 的
cu126 在 RTX 5060 上无 kernel 可用）。checkpoint 从 Thor rsync（GCS 只有
JAX 版）：

```bash
rsync -avzP hcclab@10.191.163.226:~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch \
  ~/.cache/openpi/openpi-assets/checkpoints/

cd ~/pynoob/openpi
HF_ENDPOINT=https://hf-mirror.com .venv-train/bin/python scripts/train_mip_qat.py \
  --exp_name mip_qat_v1 --batch_size 8 --num_train_steps 5000 --qat
```

冻结 paligemma（ViT+LLM），只训 action expert + 投影（~300M，
PagedAdamW8bit）。OOM 回退：batch 8→4→2。checkpoints 在
`checkpoints/<exp_name>/<step>/model.safetensors`。

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

产出 126 层（expert），identity 旋转 + 全 1 表（S0）。验收：artifact
单层 cos（QAT 前向 vs kernel 前向）应 > 0.999——远高于 PTQ 的 0.986，
这是 QAT 的核心收益点。

**3. Thor 起 2 步服务**：QAT checkpoint 需组装成完整服务目录（模型权重
换掉、assets/norm stats 沿用 base）：

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

**4. 评测**：冒烟 10 集 → LIBERO-10 ×500（命令同 §3 冒烟，改输出目录
`omega_mip2step_*`）。对照基线：10 步 E0M3 90.4% / 43-50s 每集；
预期 expert 耗时降为 ~1/5（2 NFE vs 10）。

---

## 复现检查清单

- [ ] 延迟：PyTorch BF16 ~137 ms / TRT fp16 ~85.8 ms / TRT FP8+NVFP4 ~49.9 ms
- [ ] compare 模式 cosine ≥ 0.99
- [ ] `numsteps_sweep_*.csv` 线性拟合 T_fixed / T_step 分解
- [ ] tegrastats EMC% 分阶段表（warmup / inference_test）
- [ ] nsys GPU metrics 分阶段饱和度（SMs Active / SM Issue / Tensor Active × prefill / expert）
- [ ] libero_10 三臂成功率：A ~91.6% / B ~93.0% / C ~80.0%（各 500 episodes）；
  臂 D（Omega W4A4）~93.2%；臂 D+E0M3 kernel ~90.4%（vs A p=0.53 / vs D p=0.07）
- [x] 路线 A（双侧 E0M3 + prefix 图）：action cos 四项全过 NVFP4 门禁
  （e0m3/bf16 = 0.99940/0.99694/0.99872/0.99512；fake/bf16 反而全不过——
  pack GPTQ 优于运行时 RTN 的直接证据）；冒烟 10/10；×500 全量 **93.8%**
  （vs A 91.6% p=0.135 / vs D 93.2% p=0.749 无差异 / vs expert-only E0M3
  90.4% p=0.033 显著更优，task9 60%→94%）；每集均值 42.9s（episode 长度
  混杂，精确 per-inference 延迟用 `bench_omega_e0m3_infer.py` 补测）
  （2026-08-23，见"路线 A"节）
- [ ] LIBERO-Plus 子集：A ~81.4% / C ~64.3% / D ~76.7%（各 210 episodes）

---

## 参考

- [OpenPi π₀.₅ on Jetson Thor | Jetson AI Lab](https://www.jetson-ai-lab.com/tutorials/openpi_on_thor/)
- [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)（上游，分叉点 `15a9616`）
