"""Registry mapping policy_name (ROS param) -> PolicyModel implementation.

Add a branch here when you add a new policies/<x>.py module. Imports are
local to each branch so unrelated policies' dependencies (e.g. a heavy ML
framework) don't need to be installed unless that policy is selected.
"""
from .base_policy import PolicyModel


def create_policy(name: str) -> PolicyModel:
    if name == "dummy":
        from .dummy_policy import DummyPolicyModel
        return DummyPolicyModel()
    if name == "act":
        from .act.act_policy import ACTPolicyModel
        return ACTPolicyModel()
    if name == "pi":
        from .pi.pi_policy import PIPolicyModel
        return PIPolicyModel()
    if name == "liteflow":
        from .liteflow.liteflow_policy import LiteFlowPolicyModel
        return LiteFlowPolicyModel()
    if name == "groot":
        from .groot.groot_policy import GrootPolicyModel
        return GrootPolicyModel()
    if name == "diffusion":
        from .diffusion.diffusion_policy import DiffusionPolicyModel
        return DiffusionPolicyModel()
    if name in ("sim_diffusion_50k", "sim_diffusion_125k", "sim_diffusion_ema_50k", "sim_diffusion_ema_125k",
                "sim_diffusion_ema_50k_ema_weights", "sim_diffusion_ema_125k_ema_weights"):
        # same diffusion code, different checkpoint: policies/<name>/config.yaml. sim_diffusion_ema_*
        # load the EMA run's raw weights, *_ema_weights the same checkpoints' EMA weights.
        from .diffusion.diffusion_policy import DiffusionPolicyModel
        return DiffusionPolicyModel(config_dir=name)
    if name in ("sim_diffusion_aug_50k", "sim_diffusion_aug_125k", "sim_diffusion_ema_aug_50k_ema_weights", "sim_diffusion_ema_aug_125k_ema_weights"):
        # augmented EMA run (wrist colour jitter): sim_diffusion_aug_50k / _125k = its raw weights,
        # *_ema_weights = 50k / 125k EMA weights
        from .diffusion.diffusion_policy import DiffusionPolicyModel
        return DiffusionPolicyModel(config_dir=name)
    if name in ("sim_act_aug_50k", "sim_act_aug_125k", "sim_act_ema_aug_50k_ema_weights", "sim_act_ema_aug_125k_ema_weights"):
        # ACT with EMA + wrist colour jitter on Le-007-turbo: sim_act_aug_50k / _125k = its raw weights,
        # *_ema_weights = 50k / 125k EMA weights
        from .act.act_policy import ACTPolicyModel
        return ACTPolicyModel(config_dir=name)
    if name == "act_jepa":
        # ACT-JEPA exported from delta_robot_friday/JEPA/act-jepa (scripts/export_for_ros.py)
        from .act_jepa.act_jepa_policy import ACTJepaPolicyModel
        return ACTJepaPolicyModel()
    if name in ("sim_act_jepa_50k", "sim_act_jepa_87_5k", "sim_act_jepa_125k"):
        # ACT-JEPA on Le-007-turbo (wrist RGB + turbo depth, gripper hidden from the policy):
        # same act_jepa code, checkpoint from policies/<name>/config.yaml
        from .act_jepa.act_jepa_policy import ACTJepaPolicyModel
        return ACTJepaPolicyModel(config_dir=name)
    if name in ("le008_act", "le008_diffusion", "le008_act_jepa_50k", "le008_act_jepa_35k"):
        # le_008_head (real robot, 10 fps): wrist RGB + head RGB (head_rgb_topic) + wrist depth.
        # Same act / diffusion / act_jepa code, checkpoint from policies/<name>/config.yaml.
        if name == "le008_act":
            from .act.act_policy import ACTPolicyModel
            return ACTPolicyModel(config_dir=name)
        if name == "le008_diffusion":
            from .diffusion.diffusion_policy import DiffusionPolicyModel
            return DiffusionPolicyModel(config_dir=name)
        from .act_jepa.act_jepa_policy import ACTJepaPolicyModel
        return ACTJepaPolicyModel(config_dir=name)
    if name == "omnireset":
        from .omnireset.omnireset_policy import OmniResetPolicyModel
        return OmniResetPolicyModel()
    raise ValueError(
        f"unknown policy_name '{name}' (options: dummy, act, pi, liteflow, groot, diffusion, "
        "sim_diffusion_50k, sim_diffusion_125k, sim_diffusion_ema_50k, sim_diffusion_ema_125k, "
        "sim_diffusion_ema_50k_ema_weights, sim_diffusion_ema_125k_ema_weights, "
        "sim_diffusion_aug_50k, sim_diffusion_aug_125k, sim_diffusion_ema_aug_50k_ema_weights, sim_diffusion_ema_aug_125k_ema_weights, sim_act_aug_50k, sim_act_aug_125k, sim_act_ema_aug_50k_ema_weights, sim_act_ema_aug_125k_ema_weights, act_jepa, sim_act_jepa_50k, sim_act_jepa_87_5k, sim_act_jepa_125k, le008_act, le008_diffusion, le008_act_jepa_50k, le008_act_jepa_35k, omnireset)"
    )
