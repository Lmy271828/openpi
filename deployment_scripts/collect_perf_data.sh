#!/bin/bash
# Collect all profiling artifacts for the π₀.5 performance analysis.
#
# Additive deployment-side script — does NOT modify any existing file.
# Run INSIDE the Thor Docker container (after Step 5 env setup):
#
#   bash deployment_scripts/collect_perf_data.sh [path-to-pi05_pt.nsys-rep]
#
# Overridable env vars (with defaults):
#   CONFIG_NAME=pi05_libero   CKPT_DIR=<derived>   ENGINE_PATH=<derived>
#   NUM_WARMUP=3              NUM_TEST_RUNS=10
#
# Output naming — every artifact prefix encodes the backend and the capture
# hyperparameters, so runs with different settings never overwrite each other:
#   numsteps_sweep_<config>_w<warmup>_r<runs>.csv
#   pi05_ptcompile_<config>_w<warmup>_r<runs>{.nsys-rep,.sqlite,_*.csv}
#   pi05_trt_<engine-tag>_<config>_w<warmup>_r<runs>{.nsys-rep,.sqlite,_*.csv}
#   pi05_pteager_*  (from the pre-existing eager rep; its capture hyperparams
#                    are unknown, so the prefix stays fixed)
# where <engine-tag> is the engine basename minus "model_" and ".engine"
# (e.g. model_fp8_nvfp4.engine -> fp8_nvfp4).
#
# Everything is written to ./perf_data/ inside the workspace. Afterwards copy
# that directory back to the x86 host by pulling FROM the host side:
#   scp -r hcclab@10.191.163.226:~/lmy/openpi/perf_data ~/pynoob/openpi/
set -u

# Step 5.1 equivalent — self-sufficient even in a fresh container shell.
export PYTHONPATH="packages/openpi-client/src:src:.:${PYTHONPATH:-}"

# Step 5.3 equivalent — idempotent; needed in every fresh container (they run
# with --rm, so site-packages edits do not persist across restarts).
TF_DIR="/usr/local/lib/python3.12/dist-packages/transformers"
if [ -d "$TF_DIR" ] && [ -d src/openpi/models_pytorch/transformers_replace ]; then
    cp -r src/openpi/models_pytorch/transformers_replace/* "$TF_DIR/"
    echo "transformers patches applied (Step 5.3)"
fi

CONFIG_NAME="${CONFIG_NAME:-pi05_libero}"
CKPT="${CKPT_DIR:-$HOME/.cache/openpi/openpi-assets/checkpoints/${CONFIG_NAME}_pytorch}"
ENGINE="${ENGINE_PATH:-$CKPT/engine/model_fp8_nvfp4.engine}"
NUM_WARMUP="${NUM_WARMUP:-3}"
NUM_TEST_RUNS="${NUM_TEST_RUNS:-10}"
PT_REP="${1:-pi05_pt.nsys-rep}"   # existing PyTorch nsys report (arg 1 overrides)
OUT="perf_data"
mkdir -p "$OUT"

# Output prefixes: backend [+ engine precision] + config + capture hyperparams.
ENGINE_TAG=$(basename "$ENGINE" .engine); ENGINE_TAG="${ENGINE_TAG#model_}"
PT_TAG="pi05_pteager"   # legacy eager rep: capture hyperparams unknown, fixed prefix
PTC_TAG="pi05_ptcompile_${CONFIG_NAME}_w${NUM_WARMUP}_r${NUM_TEST_RUNS}"
TRT_TAG="pi05_trt_${ENGINE_TAG}_${CONFIG_NAME}_w${NUM_WARMUP}_r${NUM_TEST_RUNS}"
SWEEP_CSV="$OUT/numsteps_sweep_${CONFIG_NAME}_w${NUM_WARMUP}_r${NUM_TEST_RUNS}.csv"

echo "== config =="
echo "  CONFIG_NAME=$CONFIG_NAME"
echo "  CKPT=$CKPT"
echo "  ENGINE=$ENGINE (tag: $ENGINE_TAG)"
echo "  NUM_WARMUP=$NUM_WARMUP  NUM_TEST_RUNS=$NUM_TEST_RUNS"
echo "  PT_REP=$PT_REP"
echo "  OUT=$OUT"
echo "  prefixes: $PT_TAG / $PTC_TAG / $TRT_TAG"
echo "  sweep csv: $SWEEP_CSV"

run_stats () {  # $1=rep path, $2=output prefix
    # gpumetrics needs the rep to be captured with --gpu-metrics-device=all;
    # on reps without it (e.g. the legacy eager rep) nsys just skips the report.
    nsys stats "$1" \
        --report cuda_gpu_kern_sum \
        --report cuda_gpu_mem_time_sum \
        --report cuda_gpu_mem_size_sum \
        --report cuda_api_sum \
        --report nvtx_pushpop_sum \
        --report gpumetrics \
        --format csv --force-export=true \
        -o "$2" 2>&1 | tail -3
}

echo ""
echo "== [1/5] num_steps sweep (PyTorch backend) =="
python deployment_scripts/pi05_numsteps_sweep.py \
    --config-name "$CONFIG_NAME" \
    --checkpoint-dir "$CKPT" \
    --num-steps-list 1,2,4,6,8,10 \
    --num-warmup "$NUM_WARMUP" --num-test-runs "$NUM_TEST_RUNS" \
    --output-csv "$SWEEP_CSV"

echo ""
echo "== [2/5] nsys stats for existing PyTorch report =="
# NOTE: the existing pi05_pt.nsys-rep was captured with --profile-eager
# (Eager PyTorch, NOT torch.compile) — keep it as the eager reference.
if [ -f "$PT_REP" ]; then
    run_stats "$PT_REP" "$OUT/$PT_TAG"
else
    echo "  SKIP: $PT_REP not found (pass its path as arg 1)"
fi

echo ""
echo "== [3/5] nsys profile + stats for torch.compile PyTorch run =="
# Prefer the NVTX-annotated variant if present on Thor; without
# --profile-eager it runs the default torch.compile(max-autotune) path,
# matching the pytorch_baseline.png numbers.
PT_SCRIPT="deployment_scripts/pi05_inference_nvtx.py"
[ -f "$PT_SCRIPT" ] || PT_SCRIPT="deployment_scripts/pi05_inference.py"
echo "  using $PT_SCRIPT (default = torch.compile enabled)"
# --gpu-metrics-device=all samples GPU counters (clocks, SM/Tensor activity,
# DRAM throughput where the device exposes it) for the bandwidth analysis.
nsys profile -o "$OUT/$PTC_TAG" --force-overwrite=true -t cuda,nvtx \
    --gpu-metrics-device=all \
    python "$PT_SCRIPT" \
        --config-name "$CONFIG_NAME" \
        --checkpoint-dir "$CKPT" \
        --inference-mode pytorch \
        --num-warmup "$NUM_WARMUP" --num-test-runs "$NUM_TEST_RUNS" \
    2>&1 | tail -15
run_stats "$OUT/$PTC_TAG.nsys-rep" "$OUT/$PTC_TAG"

echo ""
echo "== [4/5] nsys profile + stats for TensorRT run =="
if [ -f "$ENGINE" ]; then
    nsys profile -o "$OUT/$TRT_TAG" --force-overwrite=true -t cuda,nvtx \
        --gpu-metrics-device=all \
        python deployment_scripts/pi05_inference.py \
            --config-name "$CONFIG_NAME" \
            --checkpoint-dir "$CKPT" \
            --engine-path "$ENGINE" \
            --inference-mode tensorrt \
            --num-warmup "$NUM_WARMUP" --num-test-runs "$NUM_TEST_RUNS" \
        2>&1 | tail -15
    run_stats "$OUT/$TRT_TAG.nsys-rep" "$OUT/$TRT_TAG"
else
    echo "  SKIP: engine not found at $ENGINE (set ENGINE_PATH)"
fi

echo ""
echo "== [5/5] trtexec build artifacts (per-layer profile) =="
for f in "${ENGINE}_profile.json" "${ENGINE}_layers.json" "${ENGINE}.log"; do
    if [ -f "$f" ]; then cp -v "$f" "$OUT/"; else echo "  SKIP: $f not found"; fi
done

echo ""
echo "Done. Copy back to host (run ON the host, WSL has no sshd):"
echo "  scp -r hcclab@10.191.163.226:~/lmy/openpi/$OUT ~/pynoob/openpi/"
ls -la "$OUT"
