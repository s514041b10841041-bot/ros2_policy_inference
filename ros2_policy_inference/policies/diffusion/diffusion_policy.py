"""Diffusion Policy (lerobot): runs a LeRobot diffusion checkpoint, configured by config.yaml."""
import os
import time

import numpy as np
import yaml
from rclpy.logging import get_logger

from ..base_policy import ACTION_DIM, PolicyModel
from ..cameras import build_cameras

import torch
from lerobot.configs import PreTrainedConfig
from lerobot.policies import get_policy_class, make_pre_post_processors, prepare_observation_for_inference
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from .rtc_ddim import guided_conditional_sample, prefix_weights

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


_STATE_SEGMENTS = {"joint_pos": 6, "gripper_state": 1, "ft_wrench": 6}


class DiffusionPolicyModel(PolicyModel):
    def __init__(self, config_dir="diffusion"):
        """config_dir: the policies/ subfolder whose config.yaml to use (diffusion, sim_diffusion_50k,
        sim_diffusion_ema_125k, ...), so several diffusion checkpoints can share this code."""
        with open(_config_path(config_dir)) as f:
            cfg = yaml.safe_load(f) or {}
        self._log = get_logger("diffusion_policy")
        self.request_topic = cfg.get("request_topic")
        self.chunk_topic = cfg.get("chunk_topic")
        self.head_rgb_topic = cfg.get("head_rgb_topic")

        ckpt_path = cfg["ckpt_path"]
        self._task = cfg.get("task", "")
        self._robot_type = cfg.get("robot_type", "")

        policy_cfg = PreTrainedConfig.from_pretrained(ckpt_path)
        policy_cfg.pretrained_path = ckpt_path
        if cfg.get("device"):
            policy_cfg.device = cfg["device"]
        if cfg.get("n_action_steps"):
            policy_cfg.n_action_steps = cfg["n_action_steps"]
        if cfg.get("num_inference_steps"):
            policy_cfg.num_inference_steps = cfg["num_inference_steps"]
        assert policy_cfg.output_features[ACTION].shape[0] == ACTION_DIM, "checkpoint action dim mismatch"
        self.sync_chunk_size = policy_cfg.n_action_steps
        self._device = torch.device(policy_cfg.device)

        self._cameras = build_cameras(cfg["cameras"], policy_cfg.image_features, ckpt_path)

        state_dim = policy_cfg.input_features[OBS_STATE].shape[0]
        self._state_keys = cfg["state_keys"]
        assert sum(_STATE_SEGMENTS[k] for k in self._state_keys) == state_dim, "state_keys don't match checkpoint"

        self._policy = get_policy_class(policy_cfg.type).from_pretrained(ckpt_path, config=policy_cfg)
        self._policy.to(self._device).eval()
        self._preprocessor, self._postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg, pretrained_path=ckpt_path,
            preprocessor_overrides={"device_processor": {"device": policy_cfg.device}},
        )

        # Real-Time Chunking (rtc_ddim.py), used by inference_node when the sim runs async chunks
        rtc = cfg.get("rtc") or {}
        self.supports_rtc = bool(rtc.get("enabled", True))
        self._rtc_max_guidance = float(rtc.get("max_guidance_weight", 50.0))
        self._rtc_schedule = str(rtc.get("schedule", "linear"))
        self.horizon = policy_cfg.horizon
        self.first_action_index = policy_cfg.n_obs_steps - 1  # horizon row of the current step

        self._log.info(
            f"diffusion loaded from {ckpt_path}: task='{self._task}', state={list(self._state_keys)}, "
            f"cameras={list(self._cameras)}, n_action_steps={policy_cfg.n_action_steps}"
        )
        if cfg.get("warmup", True):
            self._warmup({k: policy_cfg.input_features[k].shape[1:] for k in self._cameras})

    def reset(self):
        self._policy.reset()
        self._preprocessor.reset()
        self._postprocessor.reset()

    def predict(self, obs: dict) -> np.ndarray:
        # DiffusionPolicy.select_action() caches its own obs-history and action-chunk queues
        # internally, so unlike pi/groot this needs no deque of our own.
        frame = {key: encode(obs[source]) for key, (source, encode) in self._cameras.items()}
        frame[OBS_STATE] = self._build_state(obs)
        batch = prepare_observation_for_inference(frame, self._device, self._task, self._robot_type)
        with torch.inference_mode():
            action = self._policy.select_action(self._preprocessor(batch))
        action = self._postprocessor(action)
        return action.squeeze(0).float().cpu().numpy()

    def predict_chunk_rtc(self, obs: dict, prefix_norm, frozen: int):
        """One full-horizon chunk with Real-Time Chunking guidance.

        prefix_norm: (n, ACTION_DIM) normalized actions of the previous chunk re-indexed to this
        chunk's horizon rows (row 0 = the step before the current one, row first_action_index =
        the current step), or None for the first chunk (no guidance). frozen: rows [0, frozen)
        execute while this chunk is computed and get weight 1; the rest of the prefix decays to 0.
        Returns (actions (horizon - first_action_index, ACTION_DIM) unnormalized, from the current
        step on; normalized (horizon, ACTION_DIM), to pass back as the next chunk's prefix).
        """
        frame = {key: encode(obs[source]) for key, (source, encode) in self._cameras.items()}
        frame[OBS_STATE] = self._build_state(obs)
        batch = self._preprocessor(prepare_observation_for_inference(frame, self._device, self._task, self._robot_type))
        diffusion = self._policy.diffusion
        n_obs = diffusion.config.n_obs_steps
        # same obs history select_action builds after reset(): the current obs repeated n_obs times
        model_batch = {OBS_STATE: batch[OBS_STATE].unsqueeze(1).repeat(1, n_obs, 1)}
        if diffusion.config.image_features:
            images = torch.stack([batch[k] for k in diffusion.config.image_features], dim=-4)
            model_batch[OBS_IMAGES] = images.unsqueeze(1).repeat(1, n_obs, *([1] * (images.ndim - 1)))
        with torch.inference_mode():
            global_cond = diffusion._prepare_global_conditioning(model_batch)
        global_cond = global_cond.clone()  # leave inference mode so autograd can run through the U-Net
        horizon, adim = self.horizon, ACTION_DIM
        prefix = torch.zeros(1, horizon, adim, device=self._device)
        if prefix_norm is None or len(prefix_norm) == 0:
            weights = torch.zeros(horizon)
        else:
            n = min(len(prefix_norm), horizon)
            prefix[0, :n] = torch.as_tensor(np.asarray(prefix_norm[:n]), dtype=torch.float32, device=self._device)
            weights = prefix_weights(min(frozen, n), n, horizon, self._rtc_schedule)
        if weights.sum() > 0:
            norm = guided_conditional_sample(diffusion, global_cond, prefix, weights, self._rtc_max_guidance)
        else:
            with torch.inference_mode():
                norm = diffusion.conditional_sample(1, global_cond=global_cond)
        norm = norm.detach()
        with torch.inference_mode():
            actions = self._postprocessor(norm[:, self.first_action_index:].clone())
        return actions.squeeze(0).float().cpu().numpy(), norm.squeeze(0).float().cpu().numpy()

    def _build_state(self, obs):
        return np.concatenate([np.asarray(obs[k], dtype=np.float32).ravel() for k in self._state_keys])

    def _warmup(self, shapes):
        """shapes: image feature -> (H, W). Both encodings produce (H, W, 3) uint8."""
        frame = {key: np.zeros((*hw, 3), np.uint8) for key, hw in shapes.items()}
        frame[OBS_STATE] = np.zeros(sum(_STATE_SEGMENTS[k] for k in self._state_keys), np.float32)
        batch = prepare_observation_for_inference(frame, self._device, self._task, self._robot_type)
        started = time.perf_counter()
        with torch.inference_mode():
            self._policy.select_action(self._preprocessor(batch))
        self._log.info(f"warmup inference in {(time.perf_counter() - started) * 1e3:.0f} ms")
        self.reset()
