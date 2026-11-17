"""Check an exported ACT-JEPA checkpoint against the training code's own predictions.

    python3 -m ros2_policy_inference.policies.act_jepa.parity_check /checkpoints/checkpoint_act_jepa_50k

Feeds parity.npz's frames (uint8 RGB as the node gets them, the recorded state) through this
package's network and preprocessing, and compares the result with the chunks the training code
predicted for the same frames. Needs no ROS and no camera.
"""
import os
import sys

import numpy as np
import torch

from .model import load_network


def main(ckpt_path, device=None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    net, cfg, norm = load_network(ckpt_path, device)
    d = np.load(os.path.join(ckpt_path, "parity.npz"))

    def t(x):
        return torch.tensor(x, dtype=torch.float32, device=device)

    with torch.inference_mode():
        # older single-camera fixtures store one "rgb" array; newer ones "images" = (n_cams, B, H, W, 3)
        stacks = [d["rgb"]] if "rgb" in d.files else list(d["images"])
        if "image_keys" in d.files:
            assert list(d["image_keys"]) == cfg["image_keys"], "parity.npz cameras differ from config.json"
        images = [torch.from_numpy(x).to(device).permute(0, 3, 1, 2).float().div_(255.0) for x in stacks]
        state = (t(d["state"]) - t(norm["observation.state"]["mean"])) / (t(norm["observation.state"]["std"]) + 1e-8)
        pred = net(images, state, torch.from_numpy(d["task_index"]).to(device).long())
        pred = (pred * (t(norm["action"]["std"]) + 1e-8) + t(norm["action"]["mean"])).cpu().numpy()

    exp = d["expected_actions"]
    scale = np.abs(exp).reshape(-1, exp.shape[-1]).max(0) + 1e-12
    rel = np.abs(pred - exp).reshape(-1, exp.shape[-1]).max(0) / scale
    grip_agree = float(((pred[..., 6] < 0.5) == (exp[..., 6] < 0.5)).mean())
    print(f"{len(exp)} frames x {exp.shape[1]} steps on {device}, cameras {cfg['image_keys']}, "
          f"state dims hidden: {cfg.get('context_state_mask') or []}")
    print("max |diff| / max |expected| per action dim:", np.array2string(rel, precision=2, separator=', '))
    print(f"gripper open/close agreement: {grip_agree:.3f}")
    ok = bool(rel.max() < 1e-2 and grip_agree == 1.0)
    print("PARITY OK" if ok else "PARITY FAILED")
    return ok


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    sys.exit(0 if main(sys.argv[1], *sys.argv[2:3]) else 1)
