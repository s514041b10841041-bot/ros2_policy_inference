"""ACT-JEPA: runs a checkpoint exported by delta_robot_friday/JEPA/act-jepa/scripts/export_for_ros.py,
configured by config.yaml.

The node calls reset() and then predict() sync_chunk_size times per request. The first predict()
after a reset runs the network once and queues the first n_action_steps of the predicted chunk;
the following calls pop from that queue. This matches how the act policy uses lerobot's
select_action().
"""
import os
from collections import deque

import numpy as np
import torch
import yaml
from rclpy.logging import get_logger

from ..base_policy import ACTION_DIM, PolicyModel
from ..cameras import build_cameras
from .model import load_network

_STATE_SEGMENTS = {"joint_pos": 6, "gripper_state": 1, "ft_wrench": 6}


def _config_path(config_dir):
    """policies/<config_dir>/config.yaml. colcon only installs .py files, so fall back to the
    mounted source tree in the container."""
    candidates = (
        os.path.join(os.path.dirname(os.path.dirname(__file__)), config_dir, "config.yaml"),
        f"/workspace/src/ros2_policy_inference/ros2_policy_inference/policies/{config_dir}/config.yaml",
    )
    path = next((p for p in candidates if os.path.isfile(p)), None)
    if path is None:
        raise FileNotFoundError(f"no config.yaml for '{config_dir}' (looked in {list(candidates)})")
    return path


class ACTJepaPolicyModel(PolicyModel):
    def __init__(self, config_dir="act_jepa"):
        with open(_config_path(config_dir)) as f:
            cfg = yaml.safe_load(f) or {}
        self._log = get_logger("act_jepa_policy")
        self.request_topic = cfg.get("request_topic")
        self.chunk_topic = cfg.get("chunk_topic")
        self.head_rgb_topic = cfg.get("head_rgb_topic")

        ckpt_path = cfg["ckpt_path"]
        self._device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._net, net_cfg, norm = load_network(ckpt_path, self._device)
        assert net_cfg["action_dim"] == ACTION_DIM, "checkpoint action dim mismatch"

        self._chunk = net_cfg["chunk_size"]
        self._n_action_steps = int(cfg.get("n_action_steps") or self._chunk)
        assert 1 <= self._n_action_steps <= self._chunk, "n_action_steps out of [1, chunk_size]"
        self.sync_chunk_size = self._n_action_steps

        self._state_keys = cfg["state_keys"]
        assert sum(_STATE_SEGMENTS[k] for k in self._state_keys) == net_cfg["state_dim"], \
            "state_keys don't match the checkpoint's state_dim"
        self._image_size = int(net_cfg["image_size"])
        self._image_keys = list(net_cfg["image_keys"])
        if "cameras" in cfg:
            # same block as the act/diffusion configs: image feature -> obs source + encoding
            # (rgb, or depth encoded with the dataset's depth_info.json, e.g. turbo for Le-007)
            self._cameras = build_cameras(cfg["cameras"], self._image_keys, ckpt_path)
        else:  # single-camera configs from before multi-camera support
            assert len(self._image_keys) == 1, "multi-camera checkpoint needs a cameras block in config.yaml"
            src = cfg.get("image_source", "rgb")
            self._cameras = {self._image_keys[0]: (src, lambda x: np.ascontiguousarray(x, dtype=np.uint8))}
        masked = net_cfg.get("context_state_mask") or []
        self._task_index = torch.tensor([int(cfg.get("task_index", 0))], device=self._device)

        def stats(key):
            s = norm[key]
            return (torch.tensor(s["mean"], dtype=torch.float32, device=self._device),
                    torch.tensor(s["std"], dtype=torch.float32, device=self._device) + s.get("eps", 1e-8))
        self._state_mean, self._state_std = stats("observation.state")
        self._action_mean, self._action_std = stats("action")

        self._queue = deque()
        self._log.info(
            f"act_jepa loaded from {ckpt_path} (trained {net_cfg.get('trained_steps', '?')} steps): "
            f"state={list(self._state_keys)} (dims {masked} hidden from the policy), "
            f"cameras={ {k: src for k, (src, _) in self._cameras.items()} } {self._image_size}px, "
            f"chunk_size={self._chunk}, n_action_steps={self._n_action_steps}, device={self._device}"
        )
        if cfg.get("warmup", True):
            n = self._image_size
            self.predict({"rgb": np.zeros((n, n, 3), np.uint8), "head_rgb": np.zeros((n, n, 3), np.uint8),
                          "depth": np.zeros((n, n), np.uint16),
                          "joint_pos": np.zeros(6, np.float32), "gripper_state": np.float32(1.0),
                          "ft_wrench": np.zeros(6, np.float32)})
            self.reset()

    def reset(self):
        self._queue.clear()

    @torch.inference_mode()
    def predict_chunk(self, obs: dict) -> np.ndarray:
        """obs -> the full (chunk_size, ACTION_DIM) predicted chunk, denormalised."""
        images = []
        for key in self._image_keys:
            source, encode = self._cameras[key]
            img = encode(obs[source])
            if img.shape != (self._image_size, self._image_size, 3) or img.dtype != np.uint8:
                raise ValueError(f"{key}: expected {self._image_size}x{self._image_size}x3 uint8 from '{source}', "
                                 f"got {img.shape} {img.dtype} (set the node's image_size to {self._image_size})")
            # as in training: uint8 / 255, no ImageNet mean/std
            images.append(torch.from_numpy(img).to(self._device).permute(2, 0, 1)[None].float().div_(255.0))
        state = torch.from_numpy(self._build_state(obs)).to(self._device)[None]
        state = (state - self._state_mean) / self._state_std
        action = self._net(images, state, self._task_index)[0]         # (chunk, 7), normalised
        return (action * self._action_std + self._action_mean).float().cpu().numpy()

    def predict(self, obs: dict) -> np.ndarray:
        if not self._queue:
            self._queue.extend(self.predict_chunk(obs)[: self._n_action_steps])
        return self._queue.popleft()

    def _build_state(self, obs):
        return np.concatenate([np.asarray(obs[k], dtype=np.float32).ravel() for k in self._state_keys])
