# FlashRT Thor (JetPack 7.2, SM110) 环境配置
# 用法: source deployment_scripts/thor_jp72_env.sh   （在 openpi 仓库任意目录执行）
#
# 实测环境（2026-08，Thor 宿主机，非容器）：
#   python 3.12 venv: third_party/flashrt/.venv/
#   torch 2.11.0 (cu130, sbsa wheel)
#     pip install torch --index-url https://pypi.jetson-ai-lab.io/sbsa/cu130
#   torch.cuda.is_available() = True, capability = (11, 0)
#
# 已踩过的缺失库及修复（jetson-ai-lab torch wheel 为 --no-deps 风格）：
#   1) ImportError: libnvpl_lapack_lp64_gomp.so.0
#      -> pip install nvpl-lapack nvpl-blas --index-url https://pypi.jetson-ai-lab.io/sbsa/cu130/
#      安装版本: nvpl-lapack 0.4.0.1, nvpl-blas 0.6.0
#      库落点: $SP/nvpl/lib/  （注意不在 nvidia/ 目录下）
#   2) ImportError: libcudss.so.0
#      -> 已在 $SP/nvidia/cu13/lib/ 随 wheel 提供，只需加进 LD_LIBRARY_PATH，无需另装
#   通用排查: ldd $SP/torch/lib/libtorch_cuda.so | grep "not found"
#   缺什么补什么: pip install nvidia-<name>-cu13 --index-url https://pypi.jetson-ai-lab.io/sbsa/cu130/

# 定位 flashrt 目录（本脚本在 deployment_scripts/，third_party/flashrt 与其同级）
FLASHRT_DIR="${FLASHRT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../third_party/flashrt" && pwd)}"

# 激活 venv（已在其它 venv 中则跳过）
if [ -z "$VIRTUAL_ENV" ]; then
  source "$FLASHRT_DIR/.venv/bin/activate"
fi
SP=$(python -c "import site; print(site.getsitepackages()[0])")

# 缺失库搜索路径（见上方注释）
export LD_LIBRARY_PATH="$SP/nvidia/cu13/lib:$SP/nvpl/lib:$LD_LIBRARY_PATH"

# FA4（thor-fa4 可选加速）：arch 别名必须在 import cutlass 之前 export，
# loader 的 setdefault 在 import 之后才执行，靠不住。
# dsl 4.5+ 的 sm_110a 路径有 NVVM chip-string bug（Failed translating the module to ISA），
# 必须用 sm_101a 别名；FLASH_ATTENTION_ARCH=sm_100a 选 SM100 兼容前向 kernel。
export CUTE_DSL_ARCH=sm_101a
export FLASH_ATTENTION_ARCH=sm_100a
# FA4 依赖安装（一次）：
#   pip install "nvidia-cutlass-dsl==4.5.1" "quack-kernels==0.4.1" nvidia-cuda-nvcc
#   （nvidia-cuda-nvcc 提供 ptxas；nvidia-cuda-nvcc-cu13 已 deprecated 勿装）
# 验证：python -c "from flash_rt.hardware.thor import fa4_backend as f; print(f.status())"  # 期望 active
# 注意：bench_pi05_thor_views.py 直构 frontend 不带 use_fa4；官方口径数字用
#   tests/bench_pi05_decoder_fp4_e2e.py（强制 FA4 + 要求 flashrt 工作区干净，本地补丁先 git stash）

# LIBERO 评测（eval_libero.py 同进程起仿真，子进程继承环境变量）
export PYTHONPATH="$FLASHRT_DIR/../libero:$PYTHONPATH"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1   # LIBERO init_states 为老版 torch.load 写法

# 构建（首次，详见 README-zh.md §4/§5）：
#   pip install pybind11 cmake "numpy>=1.24" safetensors "transformers<4.56" pandas pillow pyarrow
#   git clone --depth 1 --branch v4.4.2 https://github.com/NVIDIA/cutlass.git third_party/cutlass
#   pip install -e ".[torch]"
#   cmake -B build -S . -DGPU_ARCH=110 && cmake --build build -j$(nproc)
#   bash scripts/download_paligemma_tokenizer.sh
# LIBERO 仿真依赖（eval_libero.py 用）：
#   pip install "robosuite==1.4.1" "mujoco==3.2.3" bddl easydict gym matplotlib opencv-python-headless PyOpenGL
# 上游 [torch] extra 漏装的（pipeline_rtx.py 无条件 import ml_dtypes；tqdm 在 [eval]）：
#   pip install ml_dtypes tqdm
