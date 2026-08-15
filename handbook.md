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

## 复现检查清单

- [ ] 延迟：PyTorch BF16 ~137 ms / TRT fp16 ~85.8 ms / TRT FP8+NVFP4 ~49.9 ms
- [ ] compare 模式 cosine ≥ 0.99
- [ ] `numsteps_sweep_*.csv` 线性拟合 T_fixed / T_step 分解
- [ ] tegrastats EMC% 分阶段表（warmup / inference_test）
- [ ] nsys GPU metrics 分阶段饱和度（SMs Active / SM Issue / Tensor Active × prefill / expert）
- [ ] libero_10 三臂成功率：A ~91.6% / B ~93.0% / C ~80.0%（各 500 episodes）
- [ ] LIBERO-Plus 子集：A ~81.4% / C ~64.3%（各 210 episodes）

---

## 参考

- [OpenPi π₀.₅ on Jetson Thor | Jetson AI Lab](https://www.jetson-ai-lab.com/tutorials/openpi_on_thor/)
- [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)（上游，分叉点 `15a9616`）
