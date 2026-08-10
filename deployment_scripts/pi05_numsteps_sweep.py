#!/usr/bin/env python3
"""num_steps sweep experiment for π₀.5 latency decomposition (PyTorch backend).

Additive deployment-side experiment script — does NOT modify any existing file.

Idea
----
sample_actions latency is (to first order) affine in the number of flow-matching
denoising steps N:

    T(N) ≈ T_fixed + N · T_step

where T_fixed = ViT image encoding + LLM prefix prefill (+ host overhead), and
T_step = one action-expert denoise step over the cached prefix KV. Sweeping N and
fitting a line separates the two, giving a clean stage attribution that does not
depend on profiler versions.

Usage (inside the Thor Docker container, after Step 5 env setup):
    python deployment_scripts/pi05_numsteps_sweep.py \
        --config-name pi05_libero \
        --checkpoint-dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch \
        --num-steps-list 1,2,4,6,8,10 \
        --num-warmup 3 --num-test-runs 10 \
        --output-csv perf_data/numsteps_sweep_pytorch.csv

NOTE: the TensorRT engine has num_steps baked into the compiled graph
(pi0_tensorrt_sample_actions ignores the argument), so this sweep is meaningful
for the PyTorch backend only. For TRT stage attribution use the trtexec
per-layer profile (*_profile.json) collected by collect_perf_data.sh.
"""

import argparse
import csv
import os
import time

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description="π₀.5 num_steps latency sweep (PyTorch backend)")
    parser.add_argument("--config-name", type=str, default=os.getenv("CONFIG_NAME", "pi05_libero"))
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=os.path.expanduser("~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"),
    )
    parser.add_argument("--num-steps-list", type=str, default="1,2,4,6,8,10")
    parser.add_argument("--num-warmup", type=int, default=3)
    parser.add_argument("--num-test-runs", type=int, default=10)
    parser.add_argument("--output-csv", type=str, default="perf_data/numsteps_sweep_pytorch.csv")
    args = parser.parse_args()

    from openpi.policies import policy_config
    from openpi.training import config as _config
    from deployment_scripts.pi05_inference import create_synthetic_example
    from deployment_scripts.trt_model_forward import install_attention_mask_dtype_fix

    config = _config.get_config(args.config_name)
    example = create_synthetic_example(config.name)

    print(f"Loading policy ({args.config_name})...")
    policy = policy_config.create_trained_policy(config, args.checkpoint_dir)
    install_attention_mask_dtype_fix(policy._model if hasattr(policy, "_model") else policy.model)

    # Fixed golden noise so every N sees the same integration start point.
    action_horizon = config.model.action_horizon
    action_dim = config.model.action_dim
    noise = torch.normal(
        mean=0.0, std=1.0, size=(action_horizon, action_dim), dtype=torch.float32, device="cpu"
    ).numpy()

    steps_list = [int(s) for s in args.num_steps_list.split(",")]
    rows = []
    for n in steps_list:
        # Instance-level override only (no source change): the Policy forwards
        # _sample_kwargs to model.sample_actions.
        policy._sample_kwargs["num_steps"] = n

        for i in range(args.num_warmup):  # absorbs torch.compile re-tracing per new int guard
            _ = policy.infer(example, noise=noise)
            print(f"  [N={n}] warmup {i + 1}/{args.num_warmup}")

        times = []
        for _ in range(args.num_test_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = policy.infer(example, noise=noise)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)

        row = {
            "num_steps": n,
            "mean_ms": float(np.mean(times)),
            "std_ms": float(np.std(times)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
        }
        rows.append(row)
        print(f"  [N={n}] {row['mean_ms']:.2f} ± {row['std_ms']:.2f} ms")

    # Linear fit T(N) = T_fixed + N * T_step
    ns = np.array([r["num_steps"] for r in rows], dtype=np.float64)
    ts = np.array([r["mean_ms"] for r in rows], dtype=np.float64)
    t_step, t_fixed = np.polyfit(ns, ts, 1)
    t10 = t_fixed + 10.0 * t_step
    print("\n=== Linear decomposition  T(N) ≈ T_fixed + N · T_step ===")
    print(f"  T_fixed (ViT + LLM prefill + host): {t_fixed:.2f} ms  ({100.0 * t_fixed / t10:.1f}% of T(10))")
    print(f"  T_step  (one expert denoise step):  {t_step:.2f} ms  (x10 = {10.0 * t_step:.2f} ms,"
          f" {100.0 * 10.0 * t_step / t10:.1f}% of T(10))")
    print(f"  T(10) predicted: {t10:.2f} ms")

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) + ["t_fixed_ms", "t_step_ms"])
        writer.writeheader()
        for r in rows:
            r.update({"t_fixed_ms": t_fixed, "t_step_ms": t_step})
            writer.writerow(r)
    print(f"\nCSV written to {args.output_csv}")


if __name__ == "__main__":
    main()
