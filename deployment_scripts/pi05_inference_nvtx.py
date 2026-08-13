#!/usr/bin/env python3
"""π0.5 inference benchmark with full NVTX stage instrumentation.

Differences vs. the stock deployment_scripts/pi05_inference.py:
  1. The PyTorch path now carries the same warmup/test NVTX ranges as the TRT path
     (plus an S0_policy_infer_total range wrapping every policy.infer call).
  2. Model-level stage probes are installed by monkey-patching the eager model:
       S0_policy_infer_total   (script level: total - model = S4 overhead)
       S4a_obs_preprocess      (_preprocess_observation: image resize/norm)
       S1_embed_prefix_total   (embed_prefix: SigLIP x3 + lang embed + concat)
       S1a_siglip_per_image    (embed_image: fires once per camera)
       S1b_lang_embed          (embed_language_tokens)
       S2_paligemma_prefill    (PaliGemmaWithExpertModel.forward, past_key_values=None)
       S3_denoise_step_XX      (denoise_step, fires num_steps times)
       S3_expert_forward       (PaliGemmaWithExpertModel.forward, past_key_values!=None)
       S_model_sample_actions  (whole model call; gap vs S0 = tokenizer/transforms/unnorm)
       S_trt_engine_forward    (TRT path only; engine internals are monolithic)
  3. --profile-eager strips the torch.compile wrapper from sample_actions so the
     in-graph stage probes actually reach the nsys timeline.
  4. --tegrastats-log runs tegrastats alongside inference and reports EMC (DRAM
     controller) / GR3D (GPU) utilization per phase — the zero-privilege DRAM
     source on Thor, where the nsys GPU-metrics set has no DRAM counter.

Patch points verified against openpi commit 15a9616a (src/openpi/models_pytorch/
pi0_pytorch.py): sample_actions -> embed_prefix -> PaliGemmaWithExpertModel.forward
(past_key_values=None) -> denoise_step x N -> PaliGemmaWithExpertModel.forward
(past_key_values=kv). torch.compile wraps sample_actions at PI0Pytorch.__init__.

Usage:
  sudo nsys profile -t cuda,nvtx,osrt --gpu-metrics-devices all \
    --cuda-memory-usage true -o pi05_pt --force-overwrite true \
    python deployment_scripts/pi05_inference_nvtx.py \
      --config-name ${CONFIG_NAME} \
      --checkpoint-dir ~/.cache/openpi/openpi-assets/checkpoints/${CONFIG_NAME}_pytorch \
      --inference-mode pytorch --profile-eager --num-warmup 3 --num-test-runs 10

  nsys stats --report nvtx_kern_sum pi05_pt.nsys-rep
"""

import argparse
import json
import logging
import math
import os
import re
import shutil
import subprocess
import time

import numpy as np
import nvtx
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from openpi.policies import policy_config
from openpi.policies.aloha_policy import make_aloha_example
from openpi.policies.droid_policy import make_droid_example
from openpi.policies.libero_policy import make_libero_example
from openpi.training import config as _config

# Configure logging to show INFO messages
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


# ---------------------------------------------------------------------------
# tegrastats logging (Thor EMC/GR3D sampling aligned to inference phases)
# ---------------------------------------------------------------------------

class TegrastatsLogger:
    """Run tegrastats during inference and report per-phase EMC/GR3D utilization.

    tegrastats is the zero-privilege DRAM source on Thor: the nsys GPU-metrics
    set on Tegra has no DRAM counter, but tegrastats reports EMC_FREQ (memory
    controller) and GR3D_FREQ (GPU) utilization on every sample.

    tegrastats log lines carry no timestamps, so sample k is approximated as
    covering [t0 + k*dt, t0 + (k+1)*dt) with t0 = process spawn time. Alignment
    is therefore done at phase granularity (warmup / inference_test): with a
    100 ms interval and ~50 ms per inference, per-test_i alignment is noise.

    A `<log>.windows.json` sidecar (t0, interval, phase boundaries) is written
    so the raw log can be re-aligned offline. No-ops with a warning when
    tegrastats is not on PATH (e.g. x86 dev host).
    """

    _EMC_RE = re.compile(r"EMC_FREQ\s+(\d+)%")
    _GR3D_RE = re.compile(r"GR3D_FREQ\s+(\d+)%")
    # GB10y/JP7.2 tegrastats: GR3D_FREQ has no utilization %, only per-engine
    # MHz — "GR3D_FREQ @[1574,1574,1574]". Fall back to mean MHz then.
    _GR3D_MHZ_RE = re.compile(r"GR3D_FREQ\s+@?\[([\d,\s]+)\]")

    def __init__(self, log_path, interval_ms=100, peak_gbps=273.0):
        self.log_path = log_path
        self.dt = interval_ms / 1000.0
        self.peak_gbps = peak_gbps  # Thor peak, driver-reported (TARGET_INFO_GPU.memoryBandwidth)
        self.proc = None
        self.t0 = None
        self.enabled = shutil.which("tegrastats") is not None
        if not self.enabled:
            print("  [tegrastats] WARNING: tegrastats not found on PATH; EMC/GR3D logging disabled")

    def start(self):
        if not self.enabled or self.proc is not None:
            return
        self._fh = open(self.log_path, "w")
        self.proc = subprocess.Popen(
            # stdbuf -oL: tegrastats block-buffers (~8 KB ≈ 1.8 s of samples)
            # when stdout is a file, and terminate() would silently drop the
            # tail — exactly the inference_test phase. Line-buffer it instead.
            ["stdbuf", "-oL", "tegrastats", "--interval", str(int(self.dt * 1000))],
            stdout=self._fh,
            stderr=subprocess.DEVNULL,
        )
        self.t0 = time.time()

    def stop(self):
        if self.proc is None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self._fh.close()
        self.proc = None

    def report(self, title, phases):
        """Print per-phase EMC/GR3D stats. phases: [(name, start_wall, end_wall)]."""
        if not self.enabled or self.t0 is None:
            return
        with open(self.log_path) as f:
            lines = f.read().splitlines()
        samples = []  # (emc_pct, gr3d) per line; gr3d is % or mean MHz (build-dependent)
        gr3d_mhz = False
        for line in lines:
            me = self._EMC_RE.search(line)
            if not me:
                continue
            mg = self._GR3D_RE.search(line)
            if mg:
                g = float(mg.group(1))
            else:
                mf = self._GR3D_MHZ_RE.search(line)
                if not mf:
                    continue
                freqs = [int(x) for x in mf.group(1).split(",") if x.strip()]
                g = sum(freqs) / len(freqs) if freqs else 0.0
                gr3d_mhz = True
            samples.append((int(me.group(1)), g))
        if not samples:
            print("  [tegrastats] no EMC/GR3D samples parsed from log")
            return

        with open(self.log_path + ".windows.json", "w") as f:
            json.dump(
                {
                    "tegrastats_log": self.log_path,
                    "interval_ms": int(self.dt * 1000),
                    "t0_wallclock": self.t0,
                    "phases": [(nm, s, e) for nm, s, e in phases],
                },
                f,
                indent=1,
            )

        def row(name, seg):
            if not seg:
                return (name, 0, "-", "-", "-", "-", "-")
            se = [s[0] for s in seg]
            sg = [s[1] for s in seg]
            emc_mean = sum(se) / len(se)
            return (
                name,
                len(seg),
                f"{emc_mean:.1f}",
                f"{max(se)}",
                f"{emc_mean / 100 * self.peak_gbps:.0f}",
                f"{sum(sg) / len(sg):.1f}",
                f"{max(sg)}",
            )

        def phase_seg(s, e):
            i0 = max(0, math.ceil((s - self.t0) / self.dt))
            i1 = min(len(samples), max(i0, int((e - self.t0) / self.dt)))
            return samples[i0:i1]

        rows = [row(name, phase_seg(s, e)) for name, s, e in phases]
        rows.append(row("overall", samples))

        print(f"\n== tegrastats — {title} ({len(samples)} samples @ {int(self.dt * 1000)} ms, log: {self.log_path}) ==")
        gu = "GR3D MHz" if gr3d_mhz else "GR3D%"
        print(f"{'phase':<16}{'samples':>8}{'EMC% mean':>10}{'EMC% max':>9}{'DRAM GB/s':>10}{gu + ' mean':>13}{gu + ' max':>12}")
        for r in rows:
            print(f"{r[0]:<16}{r[1]:>8}{r[2]:>10}{r[3]:>9}{r[4]:>10}{r[5]:>11}{r[6]:>10}")
        print(f"  DRAM GB/s ≈ EMC% mean × {self.peak_gbps:.0f} GB/s（Thor 峰值，驱动上报值）")


# ---------------------------------------------------------------------------
# NVTX stage probes
# ---------------------------------------------------------------------------

_NVTX = {"denoise_idx": 0}


def _nvtx_wrap(fn, name, color):
    """Wrap fn so every invocation emits one NVTX range."""

    def wrapped(*args, **kwargs):
        with nvtx.annotate(name, color=color):
            return fn(*args, **kwargs)

    return wrapped


# NOTE: the wrappers below are factories taking `fn` as a parameter. Defining
# closures inline in install_nvtx_stage_probes would capture the loop variable
# `fn` by reference (late binding) and every wrapper would end up calling the
# LAST looked-up attribute (model.sample_actions) regardless of its own tag.


def _make_pwe_forward(fn):
    """Dispatch the shared PaliGemmaWithExpertModel.forward by past_key_values:
    None -> S2 prefill (produces KV cache), otherwise S3 denoise forward."""

    def pwe_forward(*args, **kwargs):
        pkv = kwargs.get("past_key_values", None)
        if pkv is None and len(args) >= 3:
            pkv = args[2]
        if pkv is None:
            tag, color = "S2_paligemma_prefill", "orange"
        else:
            tag, color = "S3_expert_forward", "red"
        with nvtx.annotate(tag, color=color):
            return fn(*args, **kwargs)

    return pwe_forward


def _make_denoise_step(fn):
    # The per-step tag reads a mutable global, so dynamo guards on its value and
    # recompiles denoise_step once per step index. With num_steps=10 that exceeds
    # the default recompile_limit (8), after which the remaining steps silently
    # run EAGER (observed: "hit config.recompile_limit (8)" in the console log,
    # +4% on the ptcompile test window). Raising the limit lets all variants
    # compile during warmup, so steady state stays fully compiled AND graphed.
    # Do NOT use torch._dynamo.disable on this wrapper instead: the graph breaks
    # evict most kernels from CUDA graphs — measured 137 -> 274 ms on the
    # ptcompile test window (graph launches 39 -> 9, exposed kernels 83 -> 11k).
    torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 64)

    def denoise_step(*args, **kwargs):
        i = _NVTX["denoise_idx"]
        _NVTX["denoise_idx"] = i + 1
        with nvtx.annotate(f"S3_denoise_step_{i:02d}", color="red"):
            return fn(*args, **kwargs)

    return denoise_step


def _make_sample_actions(fn):
    def sample_actions(*args, **kwargs):
        _NVTX["denoise_idx"] = 0
        with nvtx.annotate("S_model_sample_actions", color="yellow"):
            return fn(*args, **kwargs)

    return sample_actions


def rebind_policy_sample_actions(policy, model):
    """openpi's Policy caches model.sample_actions as self._sample_actions at
    CONSTRUCTION time (policies/policy.py). Probes installed on
    model.sample_actions after policy creation therefore never fire, and the
    cached reference may still be the torch.compile wrapper. Rebind explicitly.
    """
    if hasattr(policy, "_sample_actions"):
        policy._sample_actions = model.sample_actions
        print("  [nvtx] policy._sample_actions rebound to patched model.sample_actions")
    else:
        print("  [nvtx] WARNING: policy has no _sample_actions attribute; check openpi policy internals")


def restore_eager_sample_actions(model):
    """Remove the torch.compile wrapper PI0Pytorch puts on sample_actions.

    Returns True if a compile wrapper was found and removed.
    """
    fn = getattr(model, "sample_actions", None)
    orig = getattr(fn, "_torchdynamo_orig_callable", None) if fn is not None else None
    if orig is not None:
        model.sample_actions = orig
        print("  [nvtx] torch.compile wrapper on sample_actions removed (eager mode).")
        return True
    return False


def install_nvtx_stage_probes(model):
    """Instrument the eager PI0Pytorch model with stage-level NVTX ranges."""
    installed = []
    pwe = getattr(model, "paligemma_with_expert", None)

    # S4a: observation preprocessing (image resize/normalize, state packing)
    fn = getattr(model, "_preprocess_observation", None)
    if fn is not None:
        model._preprocess_observation = _nvtx_wrap(fn, "S4a_obs_preprocess", "purple")
        installed.append("S4a_obs_preprocess")

    # S1: whole prefix embedding (3x SigLIP + language embed + concat)
    fn = getattr(model, "embed_prefix", None)
    if fn is not None:
        model.embed_prefix = _nvtx_wrap(fn, "S1_embed_prefix_total", "green")
        installed.append("S1_embed_prefix_total")

    if pwe is not None:
        # S1a: SigLIP vision tower + projector, one range per camera image
        fn = getattr(pwe, "embed_image", None)
        if fn is not None:
            pwe.embed_image = _nvtx_wrap(fn, "S1a_siglip_per_image", "green")
            installed.append("S1a_siglip_per_image")

        # S1b: language token embedding
        fn = getattr(pwe, "embed_language_tokens", None)
        if fn is not None:
            pwe.embed_language_tokens = _nvtx_wrap(fn, "S1b_lang_embed", "green")
            installed.append("S1b_lang_embed")

        # S2/S3: both stages share PaliGemmaWithExpertModel.forward.
        # sample_actions calls it with past_key_values=None for the prefix
        # (prefill, produces KV cache) and with the KV cache for every denoise
        # step. Dispatch the NVTX tag on that argument.
        fn = getattr(pwe, "forward", None)
        if fn is not None:
            pwe.forward = _make_pwe_forward(fn)
            installed.append("S2_paligemma_prefill / S3_expert_forward")

    # S3: full denoise step (embed_suffix + expert forward + action_out_proj)
    fn = getattr(model, "denoise_step", None)
    if fn is not None:
        model.denoise_step = _make_denoise_step(fn)
        installed.append("S3_denoise_step_XX")

    # S_model: whole sample_actions call; resets the denoise counter
    fn = getattr(model, "sample_actions", None)
    if fn is not None:
        model.sample_actions = _make_sample_actions(fn)
        installed.append("S_model_sample_actions")

    print(f"  [nvtx] stage probes installed: {', '.join(installed) if installed else 'NONE (check model structure)'}")
    return installed


def install_trt_probe(policy):
    """TRT path: the engine is monolithic, so we can only bracket its enqueue.

    Per-layer timing must come from trtexec --dumpProfile or IProfiler.
    """
    # The TRT hook may have replaced either model.sample_actions or the cached
    # policy._sample_actions; wrap whichever reference the policy actually calls.
    model = policy._model if hasattr(policy, "_model") else policy.model
    target = getattr(policy, "_sample_actions", None)
    if target is not None:
        policy._sample_actions = _nvtx_wrap(target, "S_trt_engine_forward", "yellow")
        print("  [nvtx] probe installed: S_trt_engine_forward (on policy._sample_actions)")
    else:
        fn = getattr(model, "sample_actions", None)
        if fn is not None:
            model.sample_actions = _nvtx_wrap(fn, "S_trt_engine_forward", "yellow")
            rebind_policy_sample_actions(policy, model)
            print("  [nvtx] probe installed: S_trt_engine_forward (on model.sample_actions)")
        else:
            print("  [nvtx] WARNING: no sample_actions reference found after TRT setup; no engine probe installed")
            return
    print("         (engine-internal stages are monolithic in the nsys timeline;")
    print("          use trtexec --dumpProfile / IProfiler for per-layer timing)")


# ---------------------------------------------------------------------------
# Example loading (unchanged from stock script)
# ---------------------------------------------------------------------------


def create_synthetic_example(config_name):
    """Create a synthetic example based on the config type."""
    print("  - Using synthetic example (random data)")

    # Determine which example maker to use based on config name
    if "libero" in config_name.lower():
        example = make_libero_example()
        print("  - Type: LIBERO")
        print(f"  - State shape: {example['observation/state'].shape}")
        print(f"  - Image shape: {example['observation/image'].shape}")
        print(f"  - Wrist image shape: {example['observation/wrist_image'].shape}")
    elif "droid" in config_name.lower():
        example = make_droid_example()
        print("  - Type: DROID")
        print(f"  - Joint position shape: {example['observation/joint_position'].shape}")
        print(f"  - Gripper position shape: {example['observation/gripper_position'].shape}")
        print(f"  - Exterior image shape: {example['observation/exterior_image_1_left'].shape}")
        print(f"  - Wrist image shape: {example['observation/wrist_image_left'].shape}")
    elif "aloha" in config_name.lower():
        example = make_aloha_example()
        print("  - Type: ALOHA")
        print(f"  - State shape: {example['state'].shape}")
        print(f"  - Number of cameras: {len(example['images'])}")
        for cam_name, img in example["images"].items():
            print(f"  - {cam_name} shape: {img.shape}")
    else:
        # Default to LIBERO if unknown
        print(f"  - Warning: Unknown config type '{config_name}', defaulting to LIBERO")
        example = make_libero_example()
        print(f"  - State shape: {example['observation/state'].shape}")
        print(f"  - Image shape: {example['observation/image'].shape}")
        print(f"  - Wrist image shape: {example['observation/wrist_image'].shape}")

    print(f"  - Prompt: {example.get('prompt', 'N/A')}")
    return example


def load_dataset_sample(config, sample_idx):
    """Load a sample from the LIBERO dataset."""
    repo_id = config.data.repo_id
    print(f"  - Dataset: {repo_id}")

    dataset = LeRobotDataset(repo_id)
    raw_example = dataset[sample_idx]

    print(f"  - Total samples: {len(dataset)}")
    print(f"  - Using sample index: {sample_idx}")
    print(f"  - Task: {raw_example.get('task', 'N/A')}")

    # Remap keys to match policy expectations (observation/ prefix)
    example = {
        "observation/image": raw_example["image"],
        "observation/wrist_image": raw_example["wrist_image"],
        "observation/state": raw_example["state"],
        "prompt": raw_example["task"],
    }

    return example


def load_example(config, use_dataset, sample_idx):
    """Load an example either from dataset or create a synthetic one."""
    if use_dataset:
        return load_dataset_sample(config, sample_idx)
    else:
        return create_synthetic_example(config.name)


# ---------------------------------------------------------------------------
# Inference runners (NVTX-instrumented)
# ---------------------------------------------------------------------------


def run_pytorch_inference(config, checkpoint_dir, example, noise=None, num_warmup=3, num_test_runs=10, profile_eager=False, ts_logger=None):
    """Run PyTorch inference with warmup and multiple test runs."""
    print("\n--- PyTorch Inference ---")
    print("Loading policy...")
    policy = policy_config.create_trained_policy(config, checkpoint_dir)
    print("Policy loaded successfully")

    model = policy._model if hasattr(policy, "_model") else policy.model

    # Deployment-side hook: align the additive attention mask dtype with the
    # attention compute dtype before torch.compile traces sample_actions,
    # otherwise the memory-efficient SDPA kernel raises
    # "invalid dtype for bias - should match query's dtype".
    from deployment_scripts.trt_model_forward import install_attention_mask_dtype_fix

    install_attention_mask_dtype_fix(model)

    # Stage probes need eager mode: a compiled graph swallows in-graph NVTX ranges.
    compile_active = getattr(getattr(model, "sample_actions", None), "_torchdynamo_orig_callable", None) is not None
    if profile_eager and compile_active:
        restore_eager_sample_actions(model)
        compile_active = False
    install_nvtx_stage_probes(model)
    # Policy caches sample_actions at construction; point it at the patched
    # (and possibly eager-restored) version, otherwise none of the above fires.
    rebind_policy_sample_actions(policy, model)
    if compile_active:
        print("  [nvtx] WARNING: torch.compile is ACTIVE on sample_actions.")
        print("         In-graph stage ranges (S1/S2/S3) will NOT appear on the nsys")
        print("         timeline. Re-run with --profile-eager for stage attribution.")
        print("         (Absolute latency will be higher in eager mode; use compiled")
        print("         runs for the latency ledger, eager runs for the timeline.)")

    if ts_logger:
        ts_logger.start()

    # Warmup runs
    print(f"\nWarming up ({num_warmup} runs)...")
    ts_warmup_start = time.time()
    with nvtx.annotate("warmup", color="blue"):
        for i in range(num_warmup):
            with nvtx.annotate(f"warmup_{i}", color="cyan"):
                _ = policy.infer(example, noise=noise)
            print(f"  Warmup {i + 1}/{num_warmup} completed")
    ts_test_start = time.time()

    # Test runs
    print(f"\nRunning inference tests ({num_test_runs} runs)...")
    inference_times = []
    model_times = []
    action_chunk = None

    with nvtx.annotate("inference_test", color="green"):
        for i in range(num_test_runs):
            if noise is not None and i == 0:
                print(f"  Using golden noise with shape: {noise.shape}")

            with nvtx.annotate(f"test_{i}", color="yellow"):
                with nvtx.annotate("S0_policy_infer_total", color="magenta"):
                    start_time = time.time()
                    result = policy.infer(example, noise=noise)
                    inference_time = (time.time() - start_time) * 1000

            inference_times.append(inference_time)
            policy_timing = result.get("policy_timing", {})
            model_time = policy_timing.get("infer_ms", inference_time)
            model_times.append(model_time)

            if i == 0:
                action_chunk = result["actions"]

            print(f"  Test {i + 1}/{num_test_runs}: {inference_time:.2f} ms")

    ts_test_end = time.time()

    if ts_logger:
        ts_logger.stop()
        ts_logger.report(
            "PyTorch",
            [("warmup", ts_warmup_start, ts_test_start), ("inference_test", ts_test_start, ts_test_end)],
        )

    del policy

    # Calculate statistics
    inference_stats = {
        "mean": np.mean(inference_times),
        "std": np.std(inference_times),
        "min": np.min(inference_times),
        "max": np.max(inference_times),
        "all": inference_times,
    }

    model_stats = {
        "mean": np.mean(model_times),
        "std": np.std(model_times),
        "min": np.min(model_times),
        "max": np.max(model_times),
        "all": model_times,
    }

    return action_chunk, inference_stats, model_stats


def run_tensorrt_inference(
    config, checkpoint_dir, engine_path, example, noise=None, num_warmup=3, num_test_runs=10, ts_logger=None
):
    """Run TensorRT inference with warmup and multiple test runs."""
    print("\n--- TensorRT Inference ---")

    if not os.path.exists(engine_path):
        raise FileNotFoundError(
            f"TensorRT engine not found at {engine_path}\n"
            "Please run ONNX to TensorRT conversion first:\n"
            "  bash deployment_scripts/build_engine.sh"
        )

    print("Loading policy...")
    policy = policy_config.create_trained_policy(config, checkpoint_dir)
    print("Policy loaded successfully")

    print("Setting up TensorRT engine...")
    from deployment_scripts.trt_model_forward import setup_pi0_tensorrt_engine

    policy = setup_pi0_tensorrt_engine(
        policy,
        engine_path,
    )
    print("TensorRT engine ready")

    # Bracket the (monolithic) engine forward after the TRT hooks replaced it
    install_trt_probe(policy)

    if ts_logger:
        ts_logger.start()

    # Warmup runs
    print(f"\nWarming up ({num_warmup} runs)...")
    ts_warmup_start = time.time()
    with nvtx.annotate("warmup", color="blue"):
        for i in range(num_warmup):
            with nvtx.annotate(f"warmup_{i}", color="cyan"):
                _ = policy.infer(example, noise=noise)
            print(f"  Warmup {i + 1}/{num_warmup} completed")
    ts_test_start = time.time()

    # Test runs
    print(f"\nRunning inference tests ({num_test_runs} runs)...")
    inference_times = []
    model_times = []
    action_chunk = None

    with nvtx.annotate("inference_test", color="green"):
        for i in range(num_test_runs):
            if noise is not None and i == 0:
                print(f"  Using golden noise with shape: {noise.shape}")

            with nvtx.annotate(f"test_{i}", color="yellow"):
                with nvtx.annotate("S0_policy_infer_total", color="magenta"):
                    start_time = time.time()
                    result = policy.infer(example, noise=noise)
                    inference_time = (time.time() - start_time) * 1000

            inference_times.append(inference_time)
            policy_timing = result.get("policy_timing", {})
            model_time = policy_timing.get("infer_ms", inference_time)
            model_times.append(model_time)

            if i == 0:
                action_chunk = result["actions"]

            print(f"  Test {i + 1}/{num_test_runs}: {inference_time:.2f} ms")

    ts_test_end = time.time()

    if ts_logger:
        ts_logger.stop()
        ts_logger.report(
            "TensorRT",
            [("warmup", ts_warmup_start, ts_test_start), ("inference_test", ts_test_start, ts_test_end)],
        )

    del policy

    # Calculate statistics
    inference_stats = {
        "mean": np.mean(inference_times),
        "std": np.std(inference_times),
        "min": np.min(inference_times),
        "max": np.max(inference_times),
        "all": inference_times,
    }

    model_stats = {
        "mean": np.mean(model_times),
        "std": np.std(model_times),
        "min": np.min(model_times),
        "max": np.max(model_times),
        "all": model_times,
    }

    return action_chunk, inference_stats, model_stats


# ---------------------------------------------------------------------------
# Comparison (unchanged from stock script)
# ---------------------------------------------------------------------------


def compare_outputs(pytorch_actions, tensorrt_actions):
    """Compare PyTorch and TensorRT outputs."""
    print("\n" + "=" * 60)
    print("Comparison Results:")
    print("=" * 60)

    # Shape comparison
    print(f"PyTorch shape:  {pytorch_actions.shape}")
    print(f"TensorRT shape: {tensorrt_actions.shape}")

    if pytorch_actions.shape != tensorrt_actions.shape:
        print("WARNING: Shapes don't match!")
        return

    # Cosine Similarity
    pytorch_flat = pytorch_actions.flatten()
    tensorrt_flat = tensorrt_actions.flatten()

    dot_product = np.dot(pytorch_flat, tensorrt_flat)
    pytorch_norm = np.linalg.norm(pytorch_flat)
    tensorrt_norm = np.linalg.norm(tensorrt_flat)
    cosine_similarity = dot_product / (pytorch_norm * tensorrt_norm + 1e-8)

    print("\nCosine Similarity:")
    print(f"  - Overall: {cosine_similarity:.8f}")

    # Per-timestep cosine similarity
    if len(pytorch_actions.shape) >= 2:
        timestep_similarities = []
        for t in range(pytorch_actions.shape[0]):
            pt_vec = pytorch_actions[t].flatten()
            trt_vec = tensorrt_actions[t].flatten()
            dot = np.dot(pt_vec, trt_vec)
            norm_pt = np.linalg.norm(pt_vec)
            norm_trt = np.linalg.norm(trt_vec)
            sim = dot / (norm_pt * norm_trt + 1e-8)
            timestep_similarities.append(sim)

        timestep_similarities = np.array(timestep_similarities)
        print(f"  - Per-timestep Mean: {timestep_similarities.mean():.8f}")
        print(f"  - Per-timestep Min:  {timestep_similarities.min():.8f}")
        print(f"  - Per-timestep Max:  {timestep_similarities.max():.8f}")

    # Statistical comparison
    diff = np.abs(pytorch_actions - tensorrt_actions)

    print("\nAbsolute Difference Statistics:")
    print(f"  - Mean:   {diff.mean():.6f}")
    print(f"  - Max:    {diff.max():.6f}")
    print(f"  - Min:    {diff.min():.6f}")
    print(f"  - Std:    {diff.std():.6f}")
    print(f"  - Median: {np.median(diff):.6f}")

    # Relative error
    rel_error = diff / (np.abs(pytorch_actions) + 1e-6)
    print("\nRelative Error:")
    print(f"  - Mean: {rel_error.mean():.6f}")
    print(f"  - Max:  {rel_error.max():.6f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    """Main entry point for pi05 inference."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test π₀.5 inference (NVTX-instrumented)")
    parser.add_argument(
        "--inference-mode",
        type=str,
        default="pytorch",
        choices=["pytorch", "tensorrt", "compare"],
        help="Inference mode: pytorch, tensorrt, or compare (default: pytorch)",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="pi05_libero",
        help="Model config name (e.g., pi05_libero, pi05_droid, pi05_aloha, default: pi05_libero)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="/root/converted_pytorch_checkpoint",
        help="Path to checkpoint directory (default: /root/converted_pytorch_checkpoint)",
    )
    parser.add_argument(
        "--engine-path",
        type=str,
        default=None,
        help="Path to TensorRT engine file (default: {checkpoint_dir}/model_fp16.engine)",
    )
    parser.add_argument(
        "--sample-idx",
        type=int,
        default=0,
        help="Dataset sample index to test (only used with --use-dataset, default: 0)",
    )
    parser.add_argument(
        "--use-dataset",
        action="store_true",
        help="Load example from LIBERO dataset instead of using synthetic example (default: False)",
    )
    parser.add_argument(
        "--golden-noise-path",
        type=str,
        default=None,
        help="Path to golden noise .npy file (optional, will auto-generate if not provided)",
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=3,
        help="Number of warmup runs (default: 3)",
    )
    parser.add_argument(
        "--num-test-runs",
        type=int,
        default=10,
        help="Number of test runs for timing (default: 10)",
    )
    parser.add_argument(
        "--profile-eager",
        action="store_true",
        help="Strip the torch.compile wrapper from sample_actions so in-graph NVTX stage "
        "ranges (S1/S2/S3) appear on the nsys timeline. Absolute latency will be higher "
        "than the compiled baseline; use this for stage attribution, not for the ledger.",
    )
    parser.add_argument(
        "--tegrastats-log",
        type=str,
        default=None,
        help="Thor only: run tegrastats during inference and write the raw log here; "
        "per-phase EMC (DRAM) / GR3D (GPU) utilization is printed after each backend run "
        "and <path>.windows.json records the phase boundaries for offline re-alignment.",
    )
    parser.add_argument(
        "--tegrastats-interval",
        type=int,
        default=100,
        help="tegrastats sampling interval in ms (default: 100)",
    )
    args = parser.parse_args()

    # Set engine path default if not provided
    if args.engine_path is None:
        args.engine_path = os.path.join(args.checkpoint_dir, "model_fp16.engine")

    print("=" * 60)
    print(f"π₀.5 Inference Test - Mode: {args.inference_mode.upper()}")
    print(f"Config: {args.config_name}")
    print("=" * 60)

    # Load config
    config = _config.get_config(args.config_name)
    checkpoint_dir = args.checkpoint_dir

    ts_logger = None
    if args.tegrastats_log:
        ts_logger = TegrastatsLogger(args.tegrastats_log, interval_ms=args.tegrastats_interval)

    if args.inference_mode == "compare":
        # Compare mode: run both and compare
        print("\n[1/4] Loading example...")
        example = load_example(config, args.use_dataset, args.sample_idx)

        # Generate or load golden noise for deterministic comparison
        if args.golden_noise_path:
            print(f"\n[2/4] Loading golden noise from {args.golden_noise_path}...")
            if not os.path.exists(args.golden_noise_path):
                raise FileNotFoundError(f"Golden noise file not found at {args.golden_noise_path}")
            golden_noise = np.load(args.golden_noise_path)
            print(f"Golden noise loaded with shape: {golden_noise.shape}")
        else:
            print("\n[2/4] Generating golden noise for deterministic comparison...")
            # Generate noise matching the model's action output shape
            # action_horizon=10, action_dim=32 for pi05_libero
            action_horizon = config.model.action_horizon
            action_dim = config.model.action_dim
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_dtype = torch.float32

            noise_tensor = torch.normal(
                mean=0.0,
                std=1.0,
                size=(1, action_horizon, action_dim),
                dtype=compute_dtype,
                device=device,
            )
            # Convert to numpy for compatibility with save/load and remove batch dimension
            golden_noise = noise_tensor.squeeze(0).cpu().numpy()
            print(f"Generated golden noise with shape: {golden_noise.shape}")
            print(f"  (action_horizon={action_horizon}, action_dim={action_dim})")

        print("\n[3/4] Running both PyTorch and TensorRT inference...")
        tensorrt_actions, tensorrt_inference_stats, tensorrt_model_stats = run_tensorrt_inference(
            config,
            checkpoint_dir,
            args.engine_path,
            example,
            noise=golden_noise,
            num_warmup=args.num_warmup,
            num_test_runs=args.num_test_runs,
            ts_logger=ts_logger,
        )
        pytorch_actions, pytorch_inference_stats, pytorch_model_stats = run_pytorch_inference(
            config,
            checkpoint_dir,
            example,
            noise=golden_noise,
            num_warmup=args.num_warmup,
            num_test_runs=args.num_test_runs,
            profile_eager=args.profile_eager,
            ts_logger=ts_logger,
        )

        print("\n[4/4] Comparing results...")

        # Print individual results
        print("\n" + "=" * 60)
        print("Individual Results:")
        print("=" * 60)
        print("\nPyTorch:")
        print(f"  - Actions range: [{pytorch_actions.min():.4f}, {pytorch_actions.max():.4f}]")
        print(f"  - Total time: {pytorch_inference_stats['mean']:.2f} ± {pytorch_inference_stats['std']:.2f} ms")
        print(f"    (min: {pytorch_inference_stats['min']:.2f}, max: {pytorch_inference_stats['max']:.2f})")
        print(f"  - Model time: {pytorch_model_stats['mean']:.2f} ± {pytorch_model_stats['std']:.2f} ms")
        print(f"    (min: {pytorch_model_stats['min']:.2f}, max: {pytorch_model_stats['max']:.2f})")

        print("\nTensorRT:")
        print(f"  - Actions range: [{tensorrt_actions.min():.4f}, {tensorrt_actions.max():.4f}]")
        print(f"  - Total time: {tensorrt_inference_stats['mean']:.2f} ± {tensorrt_inference_stats['std']:.2f} ms")
        print(f"    (min: {tensorrt_inference_stats['min']:.2f}, max: {tensorrt_inference_stats['max']:.2f})")
        print(f"  - Model time: {tensorrt_model_stats['mean']:.2f} ± {tensorrt_model_stats['std']:.2f} ms")
        print(f"    (min: {tensorrt_model_stats['min']:.2f}, max: {tensorrt_model_stats['max']:.2f})")

        print("\nSpeedup:")
        speedup_total = pytorch_inference_stats["mean"] / tensorrt_inference_stats["mean"]
        speedup_model = pytorch_model_stats["mean"] / tensorrt_model_stats["mean"]
        print(f"  - Total: {speedup_total:.2f}x")
        print(f"  - Model: {speedup_model:.2f}x")

        # Compare outputs
        compare_outputs(pytorch_actions, tensorrt_actions)

        # Add note about noise usage
        print("\n" + "=" * 60)
        print("NOTE: Golden Noise for Deterministic Comparison")
        print("=" * 60)
        if args.golden_noise_path:
            print(f"Used loaded golden noise from: {args.golden_noise_path}")
            print("  Both models used identical noise for exact comparison")
        else:
            print("Generated random golden noise automatically")
            print("  Both models used the same generated noise for fair comparison")
            print("\nTo save and reuse this noise:")
            print(f"  np.save('golden_noise.npy', noise)  # shape: {golden_noise.shape}")
            print("  Then run with: --golden-noise-path=golden_noise.npy")

    elif args.inference_mode == "pytorch":
        # PyTorch only
        print("\n[1/2] Loading example...")
        example = load_example(config, args.use_dataset, args.sample_idx)

        print("\n[2/2] Running PyTorch inference...")
        action_chunk, inference_stats, model_stats = run_pytorch_inference(
            config,
            checkpoint_dir,
            example,
            num_warmup=args.num_warmup,
            num_test_runs=args.num_test_runs,
            profile_eager=args.profile_eager,
            ts_logger=ts_logger,
        )

        print("\n" + "=" * 60)
        print("Results:")
        print("=" * 60)
        print(f"Actions shape: {action_chunk.shape}")
        print(f"Actions range: [{action_chunk.min():.4f}, {action_chunk.max():.4f}]")
        print(f"Total inference time: {inference_stats['mean']:.2f} ± {inference_stats['std']:.2f} ms")
        print(f"    (min: {inference_stats['min']:.2f}, max: {inference_stats['max']:.2f})")
        print(f"Model inference time: {model_stats['mean']:.2f} ± {model_stats['std']:.2f} ms")
        print(f"    (min: {model_stats['min']:.2f}, max: {model_stats['max']:.2f})")

    else:  # tensorrt
        # TensorRT only
        print("\n[1/2] Loading example...")
        example = load_example(config, args.use_dataset, args.sample_idx)

        print("\n[2/2] Running TensorRT inference...")
        action_chunk, inference_stats, model_stats = run_tensorrt_inference(
            config,
            checkpoint_dir,
            args.engine_path,
            example,
            num_warmup=args.num_warmup,
            num_test_runs=args.num_test_runs,
            ts_logger=ts_logger,
        )

        print("\n" + "=" * 60)
        print("Results:")
        print("=" * 60)
        print(f"Actions shape: {action_chunk.shape}")
        print(f"Actions range: [{action_chunk.min():.4f}, {action_chunk.max():.4f}]")
        print(f"Total inference time: {inference_stats['mean']:.2f} ± {inference_stats['std']:.2f} ms")
        print(f"    (min: {inference_stats['min']:.2f}, max: {inference_stats['max']:.2f})")
        print(f"Model inference time: {model_stats['mean']:.2f} ± {model_stats['std']:.2f} ms")
        print(f"    (min: {model_stats['min']:.2f}, max: {model_stats['max']:.2f})")

    print("\nTest completed successfully!")


if __name__ == "__main__":
    main()
