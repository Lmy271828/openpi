# π₀.5 on Jetson Thor 性能分析操作手册

> 本手册记录 `report.md` 全部支撑数据的获取与处理过程，风格参照
> [OpenPi π₀.₅ on Jetson Thor | Jetson AI Lab](https://www.jetson-ai-lab.com/tutorials/openpi_on_thor/)。
> 所有步骤均在真实环境执行过；踩过的坑原样保留（见各节 Note 与 §6 排查表）。
>
> **角色划分**：
> - **Thor**：Jetson AGX Thor DevKit（JetPack 7.2，MAXN，nsys 2026.2.1），Docker 容器
>   `openpi-pi0.5:l4t-jp7.2`，workspace `~/lmy/openpi`（容器内 `/workspace`）
> - **host**：x86_64 WSL Ubuntu 24.04（nsys 2025.5.2），项目根 `~/pynoob/openpi`
>
> **前置状态**：教程 Step 1–12 已完成；已有 PyTorch / TensorRT 基线截图
> （`media/pytorch_baseline.png`、`media/tensorrt_fp8_nvfp4_baseline.png`）、
> 一份 PyTorch 路径的 nsys 报告 `pi05_pt.nsys-rep`、以及 Step 10 构建引擎时
> trtexec 自动生成的逐层 profile（`*_profile.json` / `_layers.json` / `.log`）。

---

## Step 0：分析思路与新增文件

分析遵循"不改任何原始脚本，只加新脚本"的约束。共新增 4 个文件（均在 host 编写，
传输到 Thor 执行其中 2 个）：

| 文件 | 运行侧 | 作用 |
|---|---|---|
| `deployment_scripts/pi05_numsteps_sweep.py` | Thor | num_steps 扫描，线性分解 `T(N) ≈ T_fixed + N·T_step` |
| `deployment_scripts/collect_perf_data.sh` | Thor | 一键采集（sweep + 3 路 nsys + trtexec 产物拷贝） |
| `deployment_scripts/analyze_perf.py` | host | 解析 `perf_data/` 生成 Markdown 分析表 |
| `report.md` | host | 最终报告 |

关键设计：

- **num_steps 扫描**不改源码：实例级覆盖 `policy._sample_kwargs["num_steps"]`，
  利用 `T(N) ≈ T_fixed + N·T_step` 把时延拆成"ViT+LLM prefill"与"expert 去噪循环×N"，
  不依赖 profiler 版本。
- **host 直读 `.sqlite`**：host 的 nsys 2025.5.2 打不开 Thor 的 2026.2.1 `.nsys-rep`
  （版本前向不兼容），但 nsys 导出的 `.sqlite` 是标准 SQLite，可跨版本直接查询。

---

## Step 1：两侧文件传输

Thor 与 host 的仓库内容保持一致，通过 `scp` 同步（Thor 地址 `hcclab@10.191.163.226`）：

```bash
# Thor → host：取回用户在 Thor 侧自加的 NVTX 埋点脚本（可选，用于保持仓库一致）
scp -p hcclab@10.191.163.226:~/lmy/openpi/deployment_scripts/pi05_inference_nvtx.py \
  ~/pynoob/openpi/deployment_scripts/

# host → Thor：上传采集脚本（两个都要，collect 脚本会调用 sweep 脚本）
scp -p ~/pynoob/openpi/deployment_scripts/collect_perf_data.sh \
  ~/pynoob/openpi/deployment_scripts/pi05_numsteps_sweep.py \
  hcclab@10.191.163.226:~/lmy/openpi/deployment_scripts/
```

> Note：`pi05_inference_nvtx.py` 是 Thor 侧已有的 `pi05_inference.py` NVTX 埋点变体
> （提供 `--profile-eager` 开关与 `S1_/S2_/S3_` 阶段埋点）。`collect_perf_data.sh` 检测到
> 它存在时优先使用，否则回退到原版 `pi05_inference.py`。

---

## Step 2：Thor 侧一键采集

容器内（工作目录 `/workspace`）：

```bash
bash deployment_scripts/collect_perf_data.sh pi05_pt.nsys-rep
```

脚本执行 5 步，产物全部写入 `perf_data/`：

| 步骤 | 内容 | 产物 |
|---|---|---|
| [1/5] | num_steps 扫描（PyTorch，N=1,2,4,6,8,10 × 3 warmup + 10 runs） | `numsteps_sweep_pytorch.csv` |
| [2/5] | 已有 `pi05_pt.nsys-rep`（**Eager**）的 nsys stats 导出 | `pi05_pt_*.csv` |
| [3/5] | torch.compile 路径 nsys 采样（`pi05_inference_nvtx.py`，不带 `--profile-eager`） | `pi05_ptc.nsys-rep` |
| [4/5] | TensorRT 路径 nsys 采样 | `pi05_trt.nsys-rep` |
| [5/5] | 拷贝 Step 10 的 trtexec 逐层 profile | `*_profile.json`、`*_layers.json`、`.log` |

脚本开头内置了两步环境自愈（适配 `--rm` 的新容器，替代教程 Step 5.1/5.3）：

```bash
export PYTHONPATH="packages/openpi-client/src:src:.:${PYTHONPATH:-}"
cp -r src/openpi/models_pytorch/transformers_replace/* \
  /usr/local/lib/python3.12/dist-packages/transformers/
```

> **踩坑记录 1**：首次运行时三个 python 步骤全部 `ModuleNotFoundError: No module named
> 'openpi'`——新容器 shell 没做 Step 5.1 的 `export PYTHONPATH`。nsys 对秒退的进程照样
> "成功"生成空报告，导致 `pi05_ptc_*`/`pi05_trt_*` 的 CSV 全是 0 字节。修复：把
> PYTHONPATH 与 transformers 补丁写进脚本。失败留下的空报告会被 `--force-overwrite=true`
> 自动覆盖，无需手清。

完成后拷回 host：

```bash
scp -r hcclab@10.191.163.226:~/lmy/openpi/perf_data ~/pynoob/openpi/
# engine.log 有 109 MB，非必需可只传两个 JSON：
# scp hcclab@10.191.163.226:~/lmy/openpi/perf_data/model_fp8_nvfp4.engine_{profile,layers}.json \
#   ~/pynoob/openpi/perf_data/
```

> **踩坑记录 2**：拷回的 `pi05_ptc_*`/`pi05_trt_*` CSV 仍为 0 字节（Thor 侧 stats 导出
> 对这两份报告失败），但 `.nsys-rep` 与 `.sqlite` 数据完好。后处理改为 host 直读
> `.sqlite`（见 Step 3.2），不影响任何结论。

---

## Step 3：host 侧数据处理

### 3.1 一键分析

```bash
python deployment_scripts/analyze_perf.py --perf-dir perf_data --output perf_data/analysis.md
```

输出四组表：① num_steps 扫描线性分解；② CUDA graph 使用情况（sqlite 直读）；
③ Eager kernel 分类汇总（读 `pi05_pt_*.csv`）；④ trtexec 逐层 profile 阶段归因。

### 3.2 sqlite 直读（绕过 nsys 版本不兼容）

host 的 nsys 2025.5.2 拒绝解析 Thor 的 2026.2.1 报告：

```
Exportation error: Report was created in Nsight Systems version (2026.2.1...),
newer than your current version (2025.5.2...).
```

但 `collect_perf_data.sh` 的 `nsys stats --force-export=true` 已在 Thor 侧生成
`.sqlite`，host 上可直接查询（注意用只读 URI，避免误建空库）：

```python
import sqlite3
con = sqlite3.connect('file:perf_data/pi05_trt.sqlite?mode=ro', uri=True)
# NVTX 窗口
con.execute("SELECT n.start,n.end FROM NVTX_EVENTS n JOIN StringIds s ON n.textId=s.id "
            "WHERE s.value=?", ('test_5',))
# 窗口内 CUDA API（发现 cudaGraphLaunch 的关键查询）
con.execute("SELECT s.value, COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
            "JOIN StringIds s ON r.nameId=s.id WHERE r.start>=? AND r.start<? "
            "GROUP BY s.value", (t0, t1))
# 窗口内 kernel
con.execute("SELECT COUNT(*), SUM(end-start) FROM CUPTI_ACTIVITY_KIND_KERNEL "
            "WHERE start>=? AND start<?", (t0, t1))
```

**关键发现**：torch.compile 稳态每次推理仅 39 次 `cudaGraphLaunch` + 83 次普通 launch
（TRT 为 1 + 23），graph 内 kernel 不被该 nsys 版本归因到
`CUPTI_ACTIVITY_KIND_KERNEL`——所以 compile/TRT 路径的 per-kernel 分析必须用
Eager 对照组或 trtexec 逐层 profile。该逻辑已固化为 `analyze_perf.py:
section_graph_launches`。

### 3.3 Eager 报告身份鉴定（strings 取证）

`pi05_pt.nsys-rep` 无法用 nsys 打开时，用 `strings` 直接取证：

```bash
strings -n 8 pi05_pt.nsys-rep | grep -i "inference-mode\|profile-eager\|triton_poi"
```

提取出的完整命令行证实它是 **Eager** 路径（`--inference-mode pytorch --profile-eager`），
且 `triton_*` kernel 出现 0 次（compile 必有 Inductor triton kernel）——与 126.7 ms 的
torch.compile 基线**不是同一条路径**，报告中仅作 Eager 对照组使用。

### 3.4 trtexec 逐层 profile 解析（执行序边界法）

`*_profile.json` 格式：首元素 `{"count": 49}`，其后 3847 条
`{"name", "timeMs", "averageMs", "medianMs", "percentage"}`。
Myelin 融合抹掉了大部分模块路径，阶段边界按 **`_myl0_N` 执行序**划分：

- **ViT**：`idx ≤ 最后一个 vision_tower 命名层`（~idx 301）
- **LLM prefill**：ViT 之后 ~一个 expert 层间距（`gateup` MatMul 间隔中位数）为止
  （`gateup_proj` 是导出时只对 expert 做的融合，是 expert 区域的指纹）
- **expert ×10**：其余全部（含 10 次展开的 180 个 expert 层）

注意 `timeMs` 是 49 次运行的总和，须用 `averageMs`；且 trtexec 逐层 profile 含层间
同步开销，合计 61.7 ms ≈ 实际 47.9 ms 的 1.29 倍，**只用占比**。

---

## Step 4：GPU metrics 补采（§7 资源饱和度）

### 4.1 权限问题（ERR_NVGPUCTRPERM）

直接采集报错：

```
Illegal --gpu-metrics-devices usage.
None of the installed GPUs are supported:
        Blackwell GB10B | NVIDIA Thor PCI[0000:01:00.0] - Insufficient privilege
```

容器内 root 默认缺 `CAP_SYS_ADMIN`，驱动拒绝开放性能计数器。**解决：重开容器时加
`--cap-add SYS_ADMIN`**（`exit` 退出旧容器，`--rm` 会自动删除；checkpoint/引擎在挂载卷
中不受影响）：

```bash
sudo docker run --rm -it --runtime nvidia \
  --cap-add SYS_ADMIN \
  -v "$PWD":/workspace \
  -v "$HOME/.cache/openpi":/root/.cache/openpi \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -w /workspace -p 8000:8000 \
  openpi-pi0.5:l4t-jp7.2
```

> **踩坑记录 3**：新容器里忘了 Step 5.3 的 transformers 补丁，推理秒退报
> `transformers_replace is not installed correctly`，nsys 照样生成空报告。
> 先 `cp -r ./src/openpi/models_pytorch/transformers_replace/*
> /usr/local/lib/python3.12/dist-packages/transformers/` 再重跑。

### 4.2 采集与导出

```bash
nsys profile -o perf_data/pi05_trt_gpu --force-overwrite=true -t cuda,nvtx \
  --gpu-metrics-devices=all \
  python deployment_scripts/pi05_inference.py \
    --config-name pi05_libero \
    --checkpoint-dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch \
    --engine-path ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch/engine/model_fp8_nvfp4.engine \
    --inference-mode tensorrt --num-warmup 3 --num-test-runs 10

nsys export --type sqlite --force-overwrite=true \
  -o perf_data/pi05_trt_gpu.sqlite perf_data/pi05_trt_gpu.nsys-rep
```

成功标志：结尾打出 `Model inference time: ~48 ms` 的 Results 块；`.sqlite` 约 480 MB
（含 1300 万行 GPU metrics，100 µs 采样）。

### 4.3 metrics 解析

`GPU_METRICS` 表 17 种指标（`TARGET_INFO_GPU_METRICS` 查 metricId）：本分析用
`8=GR Active`、`11=SMs Active`、`12=SM Issue`、`13=Tensor Active`（**无 DRAM 带宽项**）。
按 `test_i` NVTX 窗口切片，再按 profile 占比把窗口切成 prefill 段（前 47.8%）与
expert 段（后 52.1%）分别取均值：

```python
con.execute("SELECT AVG(value) FROM GPU_METRICS "
            "WHERE metricId=? AND timestamp>=? AND timestamp<?", (mid, a, b))
```

结果：prefill 段 SMs Active ~85% / SM Issue ~20% / Tensor ~20%；expert 段 ~71% / ~9% / ~10%
——两阶段计算资源均远未饱和，层级流水（§7）的瓶颈互补性得到实测确认。

> 备选方案：若 `--cap-add SYS_ADMIN` 仍无效（驱动模块参数锁死），可在 **Thor host** 用
> `tegrastats --interval 50 --logfile ...` 记录 `GR3D_FREQ`/`EMC_FREQ` 作为替代，零权限要求。

---

## Step 5：数据 → 报告章节映射

| report.md 章节 | 结论 | 数据源 | 获取方式 |
|---|---|---|---|
| §1 基线 | PT 130.4 / TRT 48.7 ms | `media/*_baseline.png` | 教程 Step 8/11 |
| §3.1 分解 | T_fixed=68.3、T_step=6.24 ms | `numsteps_sweep_pytorch.csv` | `pi05_numsteps_sweep.py`（Step 2 [1/5]） |
| §3.2 Eager | 316 ms/帧、GPU 空闲 45%、GEMM 占 50% | `pi05_pt_*.csv` | 既有 rep + nsys stats（Step 2 [2/5]） |
| §3.3 CUDA graph | compile=39 graph/帧、TRT=1 graph/帧 | `pi05_ptc/trt.sqlite` | sqlite 直读（Step 3.2） |
| §4.1 TRT 归因 | expert 52.1% / LLM 37.2% / ViT 10.7% | `*_engine_profile.json` | Step 10 产物 + 执行序边界法（Step 3.4） |
| §4.2-3 dynquant | NVFP4 动态量化链占 9.9% | 同上 Top-20（`MulReshDyna`） | 同上 |
| §7.2 重叠窗口 | 3.22 ms（5–7%） | 同上（每步 3.22 / 每层 1.27 ms） | 同上 |
| §7.3 饱和度 | prefill Issue 20%、expert 9% | `pi05_trt_gpu.sqlite` | Step 4 GPU metrics |

### 勘误记录（分析过程中修正过的判断）

1. `pi05_pt.nsys-rep` 初判为"PyTorch profile"，实为 **Eager**（Step 3.3 取证）。
2. prefix KV 流量初稿误按 8 个 KV head 算成 2.9 GB；GQA 实际 1 个 KV head，
   全帧仅 ~0.18 GB，"KV 量化"方向据此否决。
3. 初稿认为 compile 路径有"launch 风暴"；sqlite 直读发现其稳态已是 39 张 CUDA graph
   replay，launch 开销基本消除——launch 问题只存在于 Eager。
4. 初稿`analyze_perf.py` 解析 `_profile.json` 失败：首元素是 `{"count":49}` 而非 layer
   记录，且 `timeMs` 是 49 次总和（须用 `averageMs`）。

---

## Step 6：故障排查表

| 问题 | 现象 | 原因与解决 |
|---|---|---|
| `ModuleNotFoundError: No module named 'openpi'` | 采集脚本 python 步骤全挂 | 新容器未做 Step 5.1；`collect_perf_data.sh` 已内置 `export PYTHONPATH` |
| `transformers_replace is not installed correctly` | 新容器推理秒退 | 未做 Step 5.3；`cp -r src/openpi/models_pytorch/transformers_replace/* /usr/local/lib/python3.12/dist-packages/transformers/` |
| host nsys 打不开 `.nsys-rep` | `Exportation error: ... newer than your current version` | 2026.2.1 → 2025.5.2 前向不兼容；改用 Thor 侧导出 stats，或 host 直读 `.sqlite` |
| `Illegal --gpu-metrics-devices usage ... Insufficient privilege` | GPU metrics 采集被拒 | 容器缺 `CAP_SYS_ADMIN`；`docker run` 加 `--cap-add SYS_ADMIN` 重开 |
| nsys"成功"但 CSV 全 0 字节 | `pi05_ptc/trt_*.csv` 为空 | 被测进程秒退（上述两个环境坑）也会产生空报告；修环境后用 `--force-overwrite=true` 重采 |
| sqlite 查询返回空 / 误建新库 | 相对路径 + 非只读连接 | 用绝对路径只读 URI：`sqlite3.connect('file:<abspath>?mode=ro', uri=True)` |
| 逐层 profile 阶段归因错位 | LLM 被并入 expert | Myelin 抹掉模块路径；按 `_myl0_N` 执行序 + `gateup` 指纹划边界（Step 3.4） |

---

## 附：完整数据清单（`perf_data/`）

```
numsteps_sweep_pytorch.csv                  # Step 2 [1/5]，§3.1
pi05_pt_cuda_gpu_kern_sum.csv 等 5 个        # Step 2 [2/5]，§3.2（Eager）
pi05_pt_nvtx_pushpop_sum.csv                # 同上，Eager 阶段计时
pi05_ptc.nsys-rep / .sqlite                 # Step 2 [3/5]，§3.3（torch.compile）
pi05_trt.nsys-rep / .sqlite                 # Step 2 [4/5]，§3.3（TensorRT）
model_fp8_nvfp4.engine_profile.json         # Step 2 [5/5]，§4/§7（逐层耗时）
model_fp8_nvfp4.engine_layers.json          # 同上（逐层元数据，备用）
model_fp8_nvfp4.engine.log                  # 同上（构建日志，109 MB，可选）
pi05_trt_gpu.sqlite                         # Step 4，§7.3（GPU metrics）
analysis.md                                 # Step 3.1 一键重现
```
