#!/bin/bash
# Collect all profiling artifacts for the π₀.5 performance analysis.
#
# Additive deployment-side script — does NOT modify any existing file.
# Run INSIDE the Thor Docker container (after Step 5 env setup):
#
#   bash deployment_scripts/collect_perf_data.sh
#
# Overridable env vars (with defaults):
#   CONFIG_NAME=pi05_libero   CKPT_DIR=<derived>   ENGINE_PATH=<derived>
#   NUM_WARMUP=3              NUM_TEST_RUNS=10
#
# Each backend run is captured by two instruments from the same process:
#   - nsys (--gpu-metrics-devices=all): kernel times + SM/Tensor activity,
#     i.e. compute-side occupancy per NVTX stage (see analyze_perf.py).
#   - tegrastats (embedded via --tegrastats-log): EMC% (DRAM controller)
#     and GR3D% per phase — the only DRAM bandwidth source on Thor, where
#     the nsys GPU-metrics set has no DRAM counter.
#
# Output naming — every artifact prefix encodes the backend and the capture
# hyperparameters, so runs with different settings never overwrite each other:
#   numsteps_sweep_<config>_w<warmup>_r<runs>.csv
#   pi05_ptcompile_<config>_w<warmup>_r<runs>{.nsys-rep,.sqlite,_*.csv}
#   pi05_trt_<engine-tag>_<config>_w<warmup>_r<runs>{.nsys-rep,.sqlite,_*.csv}
# plus, per backend run:
#   <prefix>_emc.log      raw tegrastats samples (EMC%/GR3D%)
#   <prefix>_console.log  console output incl. the per-phase EMC/GR3D table
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
OUT="perf_data"
mkdir -p "$OUT"

# Output prefixes: backend [+ engine precision] + config + capture hyperparams.
ENGINE_TAG=$(basename "$ENGINE" .engine); ENGINE_TAG="${ENGINE_TAG#model_}"
PTC_TAG="pi05_ptcompile_${CONFIG_NAME}_w${NUM_WARMUP}_r${NUM_TEST_RUNS}"
TRT_TAG="pi05_trt_${ENGINE_TAG}_${CONFIG_NAME}_w${NUM_WARMUP}_r${NUM_TEST_RUNS}"
SWEEP_CSV="$OUT/numsteps_sweep_${CONFIG_NAME}_w${NUM_WARMUP}_r${NUM_TEST_RUNS}.csv"

echo "== config =="
echo "  CONFIG_NAME=$CONFIG_NAME"
echo "  CKPT=$CKPT"
echo "  ENGINE=$ENGINE (tag: $ENGINE_TAG)"
echo "  NUM_WARMUP=$NUM_WARMUP  NUM_TEST_RUNS=$NUM_TEST_RUNS"
echo "  OUT=$OUT"
echo "  prefixes: $PTC_TAG / $TRT_TAG"
echo "  sweep csv: $SWEEP_CSV"

run_stats () {  # $1=rep path, $2=output prefix
    # gpumetrics needs the rep to be captured with --gpu-metrics-device=all.
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

run_infer () {  # $1=output prefix, rest=extra args for pi05_inference_nvtx.py
    # One process, two instruments: nsys for compute-side metrics, embedded
    # tegrastats for DRAM/EMC. Full console output is tee'd to <prefix>_console.log
    # (and streamed live — do NOT re-add a `| tail -N`: it buffers everything
    # until process exit and makes the run look hung).
    local tag="$1"; shift
    nsys profile -o "$OUT/$tag" --force-overwrite=true -t cuda,nvtx \
        --gpu-metrics-devices=all \
        python deployment_scripts/pi05_inference_nvtx.py \
            --config-name "$CONFIG_NAME" \
            --checkpoint-dir "$CKPT" \
            --num-warmup "$NUM_WARMUP" --num-test-runs "$NUM_TEST_RUNS" \
            --tegrastats-log "$OUT/${tag}_emc.log" \
            "$@" \
        2>&1 | tee "$OUT/${tag}_console.log"
    run_stats "$OUT/$tag.nsys-rep" "$OUT/$tag"
}

echo ""
echo "== [1/4] num_steps sweep (PyTorch backend) =="
python deployment_scripts/pi05_numsteps_sweep.py \
    --config-name "$CONFIG_NAME" \
    --checkpoint-dir "$CKPT" \
    --num-steps-list 1,2,4,6,8,10 \
    --num-warmup "$NUM_WARMUP" --num-test-runs "$NUM_TEST_RUNS" \
    --output-csv "$SWEEP_CSV"

echo ""
echo "== [2/4] torch.compile PyTorch: nsys + tegrastats =="
# Default path = torch.compile(max-autotune), matching the PyTorch baseline.
# In-graph stage probes do not fire here; for stage attribution run
# pi05_inference_nvtx.py with --profile-eager separately.
run_infer "$PTC_TAG" --inference-mode pytorch

echo ""
echo "== [3/4] TensorRT: nsys + tegrastats =="
if [ -f "$ENGINE" ]; then
    run_infer "$TRT_TAG" --inference-mode tensorrt --engine-path "$ENGINE"
else
    echo "  SKIP: engine not found at $ENGINE (set ENGINE_PATH)"
fi

echo ""
echo "== [4/4] trtexec build artifacts (per-layer profile) =="
for f in "${ENGINE}_profile.json" "${ENGINE}_layers.json" "${ENGINE}.log"; do
    if [ -f "$f" ]; then cp -v "$f" "$OUT/"; else echo "  SKIP: $f not found"; fi
done

echo ""
echo "Done. Copy back to host (run ON the host, WSL has no sshd):"
echo "  scp -r hcclab@10.191.163.226:~/lmy/openpi/$OUT ~/pynoob/openpi/"
ls -la "$OUT"
