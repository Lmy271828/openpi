#!/usr/bin/env python3
"""Host-side analysis of π₀.5 profiling artifacts collected on Thor.

Additive analysis script — reads ./perf_data/ (produced by
deployment_scripts/collect_perf_data.sh on Thor) and prints Markdown tables
for the perf report:

  1. num_steps sweep linear decomposition (PyTorch backend)
  2. DRAM / memory-subsystem bandwidth from nsys GPU metrics
  3. DRAM / EMC utilization from tegrastats logs (Thor)
  4. CUDA graph usage per steady-state inference
  5. steady-state memcpy/memset per inference (sqlite test window)
  6. nsys GPU kernel summary, rolled up by kernel category (per run prefix)
  7. trtexec per-layer profile rolled up by model stage (ViT / LLM / expert)

Artifact prefixes encode the capture hyperparameters (see collect_perf_data.sh):
pi05_<backend>[_<engine-tag>]_<config>_w<warmup>_r<runs>. Runs are discovered
by their *_cuda_gpu_kern_sum.csv, so renamed/new runs are picked up
automatically.

Usage:
    python deployment_scripts/analyze_perf.py [--perf-dir perf_data] [--dram-peak-gbps 273]
"""

import argparse
import csv
import json
import os
import re
from collections import defaultdict

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

KERNEL_CATEGORIES = [
    ("GEMM (cuBLAS/cutlass/nvjet)", re.compile(r"gemm|cutlass|matmul|nvjet|sm\d+_xmma|kernel_\d+x\d+", re.I)),
    ("Attention (fmha/sdpa/flash)", re.compile(r"fmha|attention|sdpa|flash", re.I)),
    ("Quant/convert (FP8/FP4/cvt)", re.compile(r"quant|dequant|fp8|fp4|nvfp4|cvt|convert|cast", re.I)),
    ("Norm (layer/rms)", re.compile(r"norm", re.I)),
    ("Softmax", re.compile(r"softmax", re.I)),
    ("Elementwise/reduction (triton)", re.compile(r"triton|elementwise|reduce|vectorized|CatArrayBatched|copy", re.I)),
    ("Memcpy/Memset", re.compile(r"memcpy|memset", re.I)),
]

STAGE_PATTERNS = [
    ("ViT (SigLIP x3 views)", re.compile(r"vision_tower|multi_modal_projector|siglip", re.I)),
    ("LLM prefix (Gemma 2B)", re.compile(r"paligemma[^/]*language_model|language_model", re.I)),
    ("Action expert (x10 denoise)", re.compile(r"gemma_expert|action_in_proj|action_out_proj|time_mlp", re.I)),
    ("Embedding/misc", re.compile(r"embed|proj", re.I)),
]


def categorize(name, rules, fallback="Other"):
    for label, pat in rules:
        if pat.search(name):
            return label
    return fallback


def read_nsys_csv(path):
    """Read an nsys stats CSV into (rows, total_ns). Robust to column naming."""
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows


def col(row, *candidates):
    for c in candidates:
        for k in row:
            if k.strip().lower() == c.lower():
                return row[k]
    return None


def kernel_rollup(rows, top_n=15):
    """Aggregate nsys cuda_gpu_kern_sum rows by category. Returns (rollup, top, grand_ns)."""
    rollup = defaultdict(lambda: [0.0, 0])  # category -> [ns, instances]
    named = []
    grand = 0.0
    for r in rows:
        name = col(r, "Name") or ""
        total = float(col(r, "Total Time (ns)", "Total Time") or 0)
        inst = int(float(col(r, "Instances") or 0))
        grand += total
        cat = categorize(name, KERNEL_CATEGORIES)
        rollup[cat][0] += total
        rollup[cat][1] += inst
        named.append((total, inst, name))
    named.sort(reverse=True)
    return rollup, named[:top_n], grand


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def fmt_ms(ns):
    return f"{ns / 1e6:.2f}"


def discover_prefixes(perf_dir):
    """Run prefixes in perf_dir, discovered from *_cuda_gpu_kern_sum.csv files."""
    suffix = "_cuda_gpu_kern_sum.csv"
    if not os.path.isdir(perf_dir):
        return []
    return sorted(f[: -len(suffix)] for f in os.listdir(perf_dir) if f.endswith(suffix))


def test_windows(con):
    """test_i NVTX windows from an nsys sqlite, ordered by start time."""
    try:
        return con.execute(
            "SELECT n.start, n.end FROM NVTX_EVENTS n JOIN StringIds s ON n.textId=s.id "
            "WHERE s.value GLOB 'test_[0-9]' ORDER BY n.start"
        ).fetchall()
    except Exception:  # no NVTX_EVENTS in this capture
        return []


def stage_ranges(con, t0, t1):
    """prefill/expert sub-ranges of one test window, from the NVTX stage probes.

    Returns (prefill_ranges, expert_ranges) using S2_paligemma_prefill and
    S3_denoise_step_XX (pi05_inference_nvtx.py stage probes). Both empty when
    the capture has no stage probes (e.g. the monolithic TRT engine path, where
    engine internals are not NVTX-annotated) — callers should fall back to the
    whole test window then.
    """
    try:
        rows = con.execute(
            "SELECT s.value, n.start, n.end FROM NVTX_EVENTS n JOIN StringIds s ON n.textId=s.id "
            "WHERE n.start>=? AND n.start<? "
            "AND (s.value='S2_paligemma_prefill' OR s.value GLOB 'S3_denoise_step_*')",
            (t0, t1),
        ).fetchall()
    except Exception:
        return [], []
    prefill = [(s, e) for v, s, e in rows if v == "S2_paligemma_prefill"]
    expert = [(s, e) for v, s, e in rows if v.startswith("S3_denoise_step_")]
    return prefill, expert


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------

def section_sweep(perf_dir):
    import glob

    candidates = sorted(glob.glob(os.path.join(perf_dir, "numsteps_sweep_*.csv")))
    if not candidates:
        return "_未找到 numsteps_sweep_*.csv_"
    path = candidates[0]
    import numpy as np

    with open(path) as f:
        rows = list(csv.DictReader(f))
    ns = np.array([float(r["num_steps"]) for r in rows])
    ts = np.array([float(r["mean_ms"]) for r in rows])
    t_step, t_fixed = np.polyfit(ns, ts, 1)
    t10 = t_fixed + 10 * t_step
    table = md_table(
        ["num_steps", "mean (ms)", "std (ms)", "min", "max"],
        [[r["num_steps"], f"{float(r['mean_ms']):.2f}", f"{float(r['std_ms']):.2f}",
          f"{float(r['min_ms']):.2f}", f"{float(r['max_ms']):.2f}"] for r in rows],
    )
    return (
        table
        + f"\n\n线性拟合 `T(N) ≈ T_fixed + N·T_step`：\n"
        + f"- **T_fixed（ViT + LLM prefill + host）= {t_fixed:.2f} ms**，占 T(10) 的 {100*t_fixed/t10:.1f}%\n"
        + f"- **T_step（单次 expert denoise）= {t_step:.2f} ms**，10 步合计 {10*t_step:.2f} ms，"
        + f"占 T(10) 的 {100*10*t_step/t10:.1f}%"
    )


def section_nsys(perf_dir, prefix, label):
    kern = read_nsys_csv(os.path.join(perf_dir, f"{prefix}_cuda_gpu_kern_sum.csv"))
    if kern is None:
        return f"_未找到 {prefix}_cuda_gpu_kern_sum.csv_"
    rollup, top, grand = kernel_rollup(kern)
    if grand == 0:
        return (f"**{label}** — 窗口内无可归因 kernel（稳态计算在 CUDA graph replay 内，"
                f"此 nsys 版本不归因 graph 内 kernel，见「CUDA graph 使用情况」一节）")
    lines = [f"**{label}** — GPU kernel 总耗时 {fmt_ms(grand)} ms（含 warmup，归一化见报告正文）\n"]
    lines.append(md_table(
        ["类别", "总耗时 (ms)", "占比", "launch 次数"],
        [[c, fmt_ms(v[0]), f"{100*v[0]/grand:.1f}%", v[1]]
         for c, v in sorted(rollup.items(), key=lambda kv: -kv[1][0])],
    ))
    lines.append("\nTop-15 kernel：\n")
    lines.append(md_table(
        ["总耗时 (ms)", "次数", "kernel"],
        [[fmt_ms(t), i, n[:110]] for t, i, n in top],
    ))
    mem = read_nsys_csv(os.path.join(perf_dir, f"{prefix}_cuda_gpu_mem_time_sum.csv"))
    if mem:
        lines.append("\nMemOps：\n")
        lines.append(md_table(
            ["操作", "总耗时 (ms)", "次数"],
            [[col(r, "Name", "Operation"), fmt_ms(float(col(r, "Total Time (ns)", "Total Time") or 0)),
              col(r, "Instances", "Count")] for r in mem],
        ))
    return "\n".join(lines)


def find_layer_records(obj):
    """Recursively find a list of per-layer dicts in a trtexec profile JSON."""
    if isinstance(obj, list):
        dicts = [x for x in obj if isinstance(x, dict)]
        dur_keys = {"timeMs", "totalMs", "averageMs", "computerTimeMs", "durationMs"}
        named = [x for x in dicts if "name" in x]
        if named and (dur_keys & set(named[0]) or ("startTimeMs" in named[0] and "endTimeMs" in named[0])):
            return named
        for x in obj:
            r = find_layer_records(x)
            if r:
                return r
    elif isinstance(obj, dict):
        for v in obj.values():
            r = find_layer_records(v)
            if r:
                return r
    return None


def layer_duration_ms(rec):
    for k in ("timeMs", "totalMs", "averageMs", "computerTimeMs", "durationMs"):
        if k in rec:
            return float(rec[k])
    if "startTimeMs" in rec and "endTimeMs" in rec:
        return float(rec["endTimeMs"]) - float(rec["startTimeMs"])
    return 0.0


def _myl_idx(name):
    m = re.search(r"_myl0_(\d+)$", name)
    return int(m.group(1)) if m else -1


def section_trt_layers(perf_dir):
    """TRT per-layer profile: stage attribution by execution-index boundaries.

    Myelin strips most module paths, but keeps the vision_tower prefix; the
    LLM region follows the ViT region, and the 10x unrolled expert region runs
    to the end (detected by the fused `gateup` MatMul cadence). Boundaries:
    ViT = idx <= last vision_tower; LLM = (vit_end, last MulReshDyna/gate_proj
    region]; expert = rest. Validated against the known structure (27 ViT
    layers / 18 LLM layers / 18x10 expert layers).
    """
    candidates = [f for f in os.listdir(perf_dir) if f.endswith("_profile.json")] if os.path.isdir(perf_dir) else []
    if not candidates:
        return "_未找到 *_profile.json（trtexec per-layer profile）_"
    path = os.path.join(perf_dir, candidates[0])
    with open(path) as f:
        data = json.load(f)
    recs = find_layer_records(data)
    if not recs:
        return f"_无法解析 {path} 的 per-layer 结构，需人工查看_"
    for r in recs:
        r["_idx"] = _myl_idx(r["name"])
        # prefer per-run average; timeMs in this format is the total over `count` runs
        r["_ms"] = float(r["averageMs"]) if "averageMs" in r else layer_duration_ms(r)
    total = sum(r["_ms"] for r in recs)

    vit_end = max((r["_idx"] for r in recs if re.search(r"vision_tower|multi_modal_projector", r["name"], re.I)), default=-1)
    # LLM/expert boundary: the expert is the only region with fused gateup_proj
    # MatMuls (export-time fuse_ae_projections). The LLM ends one expert-layer
    # cadence (median gateup-to-gateup gap) before the first gateup, since the
    # pre-MLP attention chunk of expert layer 0 sits before it.
    gateup_idx = sorted(r["_idx"] for r in recs if re.search(r"gateup", r["name"], re.I))
    if gateup_idx:
        gaps = sorted(b - a for a, b in zip(gateup_idx, gateup_idx[1:]))
        lead = gaps[len(gaps) // 2] if gaps else 0
        boundary = gateup_idx[0] - lead
    else:
        boundary = vit_end + 1

    stage = defaultdict(lambda: [0.0, 0])
    for r in recs:
        if r["_idx"] <= vit_end:
            s = "ViT (SigLIP x3 views)"
        elif r["_idx"] <= boundary:
            s = "LLM prefill (Gemma 2B, NVFP4)"
        else:
            s = "Action expert x10 denoise (FP8)"
        stage[s][0] += r["_ms"]
        stage[s][1] += 1

    lines = [f"来源：`{os.path.basename(path)}`，layer 记录 {len(recs)} 条，"
             f"合计 {total:.2f} ms（trtexec 逐层 profile 含层间同步开销，绝对值约为实际推理的 1.3 倍，看占比）\n"]
    lines.append(md_table(
        ["阶段", "耗时 (ms)", "占比", "layer 实例数"],
        [[s, f"{v[0]:.2f}", f"{100*v[0]/total:.1f}%", v[1]]
         for s, v in sorted(stage.items(), key=lambda kv: -kv[1][0])],
    ))
    exp_ms = stage["Action expert x10 denoise (FP8)"][0]
    llm_ms = stage["LLM prefill (Gemma 2B, NVFP4)"][0]
    lines.append(f"\n推算：expert 每步 ≈ {exp_ms/10:.2f} ms；LLM 每层 ≈ {llm_ms/18:.2f} ms；"
                 f"层级流水重叠窗口（§7）= min(每步, LLM 层2-18) ≈ **{min(exp_ms/10, llm_ms*17/18):.2f} ms**。")

    dyn = [r for r in recs if "MulReshDyna" in r["name"] or re.search(r"Dyna", r["name"])]
    if dyn:
        dms = sum(r["_ms"] for r in dyn)
        lines.append(f"\nNVFP4 动态量化开销（`*Dyna*` 类算子）：{len(dyn)} 个，合计 {dms:.2f} ms（{100*dms/total:.1f}%）。")

    per_layer = defaultdict(float)
    for r in recs:
        per_layer[r["name"]] += r["_ms"]
    lines.append("\nTop-20 layer（按耗时）：\n")
    top = sorted(per_layer.items(), key=lambda kv: -kv[1])[:20]
    lines.append(md_table(["耗时 (ms)", "占比", "layer"],
                          [[f"{d:.3f}", f"{100*d/total:.1f}%", n[:120]] for n, d in top]))
    return "\n".join(lines)


def section_dram(perf_dir, peak_gbps=None):
    """DRAM / memory-subsystem bandwidth from nsys GPU metrics samples.

    Reads the GPU_METRICS + TARGET_INFO_GPU_METRICS tables of every *.sqlite in
    perf_dir (present when the profile was captured with
    `nsys profile --gpu-metrics-device=all`; `nsys stats` re-exports them into
    the per-run <prefix>.sqlite automatically).

    The exposed counters are device-dependent: discrete GPUs report
    "DRAM Active [Throughput %]"; Thor/Tegra may expose no DRAM counter at
    all — then the copy-engine throughput is reported as the closest proxy and
    the gap is called out explicitly. Values are kept in the counter's native
    unit (taken from the metric name); with --dram-peak-gbps, Throughput-% rows
    additionally get an effective GB/s column.
    """
    import glob
    import sqlite3

    out = []
    for path in sorted(glob.glob(os.path.join(perf_dir, "*.sqlite"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        con = sqlite3.connect(f"file:{os.path.abspath(path)}?mode=ro", uri=True)
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "GPU_METRICS" not in tables:
            con.close()
            continue
        catalog = con.execute(
            "SELECT DISTINCT metricId, metricName FROM TARGET_INFO_GPU_METRICS"
        ).fetchall()
        wins = test_windows(con)

        def agg(metric_id, t0=None, t1=None):
            q = "SELECT AVG(value), MAX(value) FROM GPU_METRICS WHERE metricId=?"
            args = [metric_id]
            if t0 is not None:
                q += " AND timestamp>=? AND timestamp<?"
                args += [t0, t1]
            row = con.execute(q, args).fetchone()
            return (row[0], row[1]) if row and row[0] is not None else None

        dram = [(mid, name) for mid, name in catalog if re.search(r"dram", name, re.I)]
        if not dram:
            # no DRAM counter on this device — fall back to the copy engines
            dram = [(mid, name) for mid, name in catalog
                    if re.search(r"copy engine active.*throughput", name, re.I)]
            note = ("nsys GPU metrics 在本设备（Tegra iGPU）未暴露 DRAM 计数器——DRAM 挂在 SoC 侧 "
                    "MC/EMC，不归 GPU metrics 采样；以下为 Copy Engine 吞吐作为显存传输代理。"
                    "真实 DRAM 带宽请用 tegrastats（EMC%）或 NCU dram__* 指标。")
        else:
            note = None

        rows = []
        for mid, name in dram:
            unit_m = re.search(r"\[([^\]]+)\]\s*$", name)
            unit = unit_m.group(1) if unit_m else ""
            label_name = re.sub(r"\s*\[[^\]]+\]\s*$", "", name)
            stats_list = [agg(mid)]
            if wins:
                stats_list.append(agg(mid, *wins[len(wins) // 2]))
            cells = [label_name, unit]
            for stats in stats_list:
                if stats is None:
                    cells += ["-", "-"]
                    continue
                mean, peak = stats
                if peak_gbps and "%" in unit:
                    cells += [f"{mean:.1f}（≈{mean / 100 * peak_gbps:.0f} GB/s）", f"{peak:.0f}"]
                else:
                    cells += [f"{mean:.2f}", f"{peak:.0f}"]
            rows.append(cells)

        # Precise prefill/expert split via the S2/S3 NVTX stage probes
        # (pi05_inference_nvtx.py), replacing the cruder time-fraction
        # estimate (splitting the test window by measured prefill/expert
        # time shares). Only possible when the capture has stage probes —
        # the monolithic TRT engine path has none and keeps the window-level
        # stats above. The phase table also covers the saturation metrics
        # (SMs Active / SM Issue / Tensor Active).
        phase_rows = []
        if wins:
            prefill, expert = stage_ranges(con, *wins[len(wins) // 2])
            if prefill or expert:
                sat = [(mid, name) for mid, name in catalog
                       if re.search(r"SMs Active|SM Issue|Tensor Active", name)]
                seen = {mid for mid, _ in dram}
                phase_metrics = dram + [(mid, name) for mid, name in sat if mid not in seen]

                def agg_ranges(metric_id, ranges):
                    vals = [v for v in (agg(metric_id, a, b) for a, b in ranges) if v]
                    if not vals:
                        return None
                    return (sum(v[0] for v in vals) / len(vals), max(v[1] for v in vals))

                for mid, name in phase_metrics:
                    unit_m = re.search(r"\[([^\]]+)\]\s*$", name)
                    unit = unit_m.group(1) if unit_m else ""
                    label_name = re.sub(r"\s*\[[^\]]+\]\s*$", "", name)
                    for phase_name, ranges in (("prefill (S2)", prefill), ("expert (S3)", expert)):
                        stats = agg_ranges(mid, ranges)
                        if stats is None:
                            phase_rows.append([label_name, unit, phase_name, 0, "-", "-"])
                        else:
                            mean, peak = stats
                            phase_rows.append([label_name, unit, phase_name, len(ranges),
                                               f"{mean:.2f}", f"{peak:.0f}"])
        con.close()

        lines = [f"**{stem}**"]
        if note:
            lines.append(f"_{note}_\n")
        headers = ["指标", "单位", "整体均值", "整体峰值"]
        if wins:
            headers += ["test 窗口均值", "test 窗口峰值"]
        lines.append(md_table(headers, rows))
        if phase_rows:
            lines.append("\n分阶段饱和度（S2/S3 NVTX 探针精确切窗，中位 test 窗口）：\n")
            lines.append(md_table(["指标", "单位", "阶段", "窗口数", "均值", "峰值"], phase_rows))
        elif wins:
            lines.append("\n_采集不含 S2/S3 NVTX 探针（如 TRT 引擎路径内部不可标注），"
                         "无法精确切分 prefill/expert，仅有 test 窗口整体统计。_")
        out.append("\n".join(lines))

    if not out:
        return "_未找到含 GPU_METRICS 的 sqlite（采集时需 nsys profile --gpu-metrics-device=all）_"
    return "\n\n".join(out)


def section_emc(perf_dir, peak_gbps=None):
    """tegrastats EMC (DRAM controller) / GR3D utilization per run.

    On Thor the DRAM counters live on the SoC-side MC/EMC and are not part of
    the nsys GPU-metrics set, so tegrastats is the zero-privilege DRAM source.
    Reads <prefix>_emc.log (raw samples). When <prefix>_emc.log.windows.json
    exists (written by pi05_inference_nvtx.py's TegrastatsLogger once samples
    parse), samples are attributed to warmup/inference phases by wall-clock
    alignment: sample k ≈ t0 + k * interval.
    """
    import glob

    emc_re = re.compile(r"EMC_FREQ\s+(\d+)%")
    gr3d_re = re.compile(r"GR3D_FREQ\s+(\d+)%")
    peak = peak_gbps or 273.0  # Thor driver-reported peak; same default as the logger
    out = []
    for path in sorted(glob.glob(os.path.join(perf_dir, "*_emc.log"))):
        prefix = os.path.basename(path)[: -len("_emc.log")]
        with open(path) as f:
            lines = f.read().splitlines()
        samples = [
            (int(me.group(1)), int(mg.group(1)))
            for line in lines
            if (me := emc_re.search(line)) and (mg := gr3d_re.search(line))
        ]
        if not samples:
            out.append(
                f"**{prefix}** — {len(lines)} 行采样但零条 EMC/GR3D 字段：tegrastats 在容器内读不到 "
                "sysfs 节点时会静默省略这两个字段。`docker run` 加 `-v /sys:/sys:ro` 后重采"
                "（handbook Step 3）。"
            )
            continue

        def row(name, seg):
            if not seg:
                return [name, 0, "-", "-", "-", "-", "-"]
            se = [s[0] for s in seg]
            sg = [s[1] for s in seg]
            em = sum(se) / len(se)
            return [name, len(seg), f"{em:.1f}", max(se), f"{em / 100 * peak:.0f}",
                    f"{sum(sg) / len(sg):.1f}", max(sg)]

        rows = []
        win_path = path + ".windows.json"
        if os.path.exists(win_path):
            with open(win_path) as f:
                win = json.load(f)
            dt = win["interval_ms"] / 1000.0
            t0 = win["t0_wallclock"]
            for name, s, e in win["phases"]:
                i0 = max(0, int(-(-(s - t0) // dt)))  # ceil without math import
                i1 = min(len(samples), max(i0, int((e - t0) / dt)))
                rows.append(row(name, samples[i0:i1]))
        rows.append(row("overall", samples))
        out.append(
            f"**{prefix}**\n\n"
            + md_table(["阶段", "样本数", "EMC% mean", "EMC% max", "DRAM GB/s ≈",
                        "GR3D% mean", "GR3D% max"], rows)
            + f"\n\n_DRAM GB/s ≈ EMC% mean × {peak:.0f} GB/s（Thor 峰值，驱动上报值）_"
        )
    if not out:
        return "_未找到 *_emc.log（采集时加 --tegrastats-log，collect_perf_data.sh 已内置）_"
    return "\n\n".join(out)


def section_steady_memops(perf_dir):
    """Memcpy/memset inside the median test window, from the nsys sqlite.

    Unlike the *_cuda_gpu_mem_*_sum.csv totals (whole capture incl. warmup and
    engine deserialization), this attributes memory traffic to one steady-state
    inference. Graph-internal copies are still invisible to this nsys build,
    so what shows up here is the non-graphed glue traffic.
    """
    # CUPTI_ACTIVITY_MEMCPY_TYPE_*
    KIND = {0: "unknown", 1: "H2D", 2: "D2H", 3: "H2A", 4: "A2H", 5: "A2A",
            6: "A2D", 7: "D2A", 8: "D2D", 9: "H2H", 10: "P2P"}
    out = []
    for prefix in discover_prefixes(perf_dir):
        path = os.path.join(perf_dir, f"{prefix}.sqlite")
        if not os.path.exists(path):
            continue
        import sqlite3

        con = sqlite3.connect(f"file:{os.path.abspath(path)}?mode=ro", uri=True)
        wins = test_windows(con)
        if not wins:
            con.close()
            continue
        t0, t1 = wins[len(wins) // 2]
        try:
            mems = con.execute(
                "SELECT copyKind, COUNT(*), SUM(end-start)/1e6, SUM(bytes)/1e6 "
                "FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE start>=? AND start<? "
                "GROUP BY copyKind ORDER BY 3 DESC", (t0, t1)).fetchall()
        except Exception:
            mems = []
        try:
            ms_cnt, ms_ns = con.execute(
                "SELECT COUNT(*), COALESCE(SUM(end-start),0) FROM CUPTI_ACTIVITY_KIND_MEMSET "
                "WHERE start>=? AND start<?", (t0, t1)).fetchone()
        except Exception:
            ms_cnt, ms_ns = 0, 0
        con.close()

        rows = [[KIND.get(k, f"kind{k}"), c, f"{t:.2f}", f"{b:.1f}"] for k, c, t, b in mems]
        if ms_cnt:
            rows.append(["memset", ms_cnt, f"{ms_ns / 1e6:.2f}", "-"])
        body = (md_table(["类型", "次数", "总耗时 (ms)", "总量 (MB)"], rows)
                if rows else "_test 窗口内无 memcpy/memset_")
        out.append(f"**{prefix}**（test_i 窗口 {(t1 - t0) / 1e6:.1f} ms）\n\n{body}")
    if not out:
        return "_未找到可用的 sqlite / test 窗口_"
    return "\n\n".join(out)


def section_graph_launches(perf_dir):
    """CUDA graph usage per steady-state inference, from the nsys sqlite exports.

    Both backends replay captured CUDA graphs in steady state; graph-replayed
    kernels are NOT attributed in CUPTI_ACTIVITY_KIND_KERNEL by this nsys
    build, so per-kernel stats inside test windows only show the glue ops.
    """
    out = []
    prefixes = discover_prefixes(perf_dir)
    if not prefixes:
        return "_未找到任何 *_cuda_gpu_kern_sum.csv 对应的运行_"
    for prefix in prefixes:
        db = f"{prefix}.sqlite"
        label = prefix
        path = os.path.join(perf_dir, db)
        if not os.path.exists(path):
            out.append(f"_未找到 {db}_")
            continue
        import sqlite3

        con = sqlite3.connect(f"file:{os.path.abspath(path)}?mode=ro", uri=True)
        wins = test_windows(con)
        if not wins:
            out.append(f"_{db} 中无 test_i NVTX 窗口_")
            con.close()
            continue
        t0, t1 = wins[len(wins) // 2]
        apis = con.execute(
            "SELECT s.value, COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId=s.id "
            "WHERE r.start>=? AND r.start<? GROUP BY s.value ORDER BY COUNT(*) DESC", (t0, t1)).fetchall()
        kern_cnt, kern_ms = con.execute(
            "SELECT COUNT(*), COALESCE(SUM(end-start)/1e6,0) FROM CUPTI_ACTIVITY_KIND_KERNEL "
            "WHERE start>=? AND start<?", (t0, t1)).fetchone()
        con.close()
        graph = next((c for v, c in apis if "cudaGraphLaunch" in v), 0)
        launch = next((c for v, c in apis if v.startswith("cudaLaunchKernel")), 0)
        out.append(
            f"**{label}**（test_i 窗口 {(t1-t0)/1e6:.1f} ms）：cudaGraphLaunch × **{graph}**，"
            f"普通 cudaLaunchKernel × {launch}，窗口内可见 kernel 仅 {kern_cnt} 个 / {kern_ms:.2f} ms"
            f"——稳态计算几乎全部在 CUDA graph replay 内（graph 内 kernel 不被此 nsys 版本归因，"
            f"per-kernel 分析需用 Eager 对照组或 TRT 逐层 profile）"
        )
    return "\n\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perf-dir", default="perf_data")
    ap.add_argument("--output", default=None, help="write markdown here (default: stdout)")
    ap.add_argument("--dram-peak-gbps", type=float, default=None,
                    help="peak DRAM bandwidth of the device; converts Throughput-%% rows to GB/s")
    args = ap.parse_args()

    sections = [
        ("num_steps 扫描（PyTorch）", section_sweep(args.perf_dir)),
        ("DRAM / 显存带宽（GPU metrics）", section_dram(args.perf_dir, args.dram_peak_gbps)),
        ("DRAM / EMC（tegrastats）", section_emc(args.perf_dir, args.dram_peak_gbps)),
        ("CUDA graph 使用情况（sqlite 直读）", section_graph_launches(args.perf_dir)),
        ("稳态 MemOps（sqlite test 窗口）", section_steady_memops(args.perf_dir)),
    ]
    for prefix in discover_prefixes(args.perf_dir):
        sections.append((f"nsys kernel 分析 — {prefix}", section_nsys(args.perf_dir, prefix, prefix)))
    sections.append(("trtexec per-layer profile（TensorRT 阶段归因）", section_trt_layers(args.perf_dir)))
    out = "\n\n".join(f"## {t}\n\n{body}" for t, body in sections)
    if args.output:
        with open(args.output, "w") as f:
            f.write(out + "\n")
        print(f"written to {args.output}")
    else:
        print(out)


if __name__ == "__main__":
    main()
