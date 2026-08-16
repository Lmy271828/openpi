#!/usr/bin/env python3
"""Generate the LIBERO obs fixture for FlashRT's bench_pi05_thor_views.py.

FlashRT ships benchmarks that expect /tmp/libero_obs_<Nv>n8.npz (N=8 LIBERO
observations) but not the fixture itself. This script builds the 2-view
variant from our LIBERO checkout. Run on Thor inside the flashrt venv
(after `source deployment_scripts/thor_jp72_env.sh`):

    python deployment_scripts/gen_flashrt_obs_fixture.py

Output: /tmp/libero_obs_2v_n8.npz  (keys: n, img_{i}, wrist_{i}, state_{i})
Image preprocessing matches flashrt examples/thor/eval_libero.py exactly:
180-degree rotation + resize_with_pad to 224x224. State follows the openpi
LIBERO convention: concat(robot0_joint_pos, robot0_gripper_qpos).
"""

import os
import pathlib
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

# Reuse get_libero_env / resize_with_pad / DUMMY_ACTION from FlashRT's eval.
_FLASHRT_ROOT = pathlib.Path(__file__).resolve().parents[1] / "third_party" / "flashrt"
sys.path.insert(0, str(_FLASHRT_ROOT / "examples" / "thor"))
from eval_libero import DUMMY_ACTION, get_libero_env, resize_with_pad  # noqa: E402

OUT = "/tmp/libero_obs_2v_n8.npz"
N = 8
NUM_STEPS_WAIT = 10


def capture(env, obs):
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = resize_with_pad(img, 224, 224)
    wrist = resize_with_pad(wrist, 224, 224)
    state = np.concatenate([obs["robot0_joint_pos"], obs["robot0_gripper_qpos"]])
    return img, wrist, state


def main():
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()["libero_10"]()
    task = suite.get_task(0)
    env, _ = get_libero_env(task, 256, seed=7)
    init_states = suite.get_task_init_states(0)

    bundle = {"n": N}
    idxs = np.linspace(0, len(init_states) - 1, N).astype(int)
    for i, si in enumerate(idxs):
        env.reset()
        obs = env.set_init_state(init_states[si])
        for _ in range(NUM_STEPS_WAIT):
            obs, *_ = env.step(DUMMY_ACTION)
        img, wrist, state = capture(env, obs)
        bundle[f"img_{i}"] = img
        bundle[f"wrist_{i}"] = wrist
        bundle[f"state_{i}"] = state
        print(f"sample {i}: init_state {si}, img {img.shape} {img.dtype}, state {state.shape}")

    np.savez(OUT, **bundle)
    print(f"saved {OUT}")
    env.close()


if __name__ == "__main__":
    main()
