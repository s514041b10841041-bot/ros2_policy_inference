"""Registry mapping policy_name (ROS param) -> PolicyModel implementation.

Add a branch here when you add a new policies/<x>.py module. Imports are
local to each branch so unrelated policies' dependencies (e.g. a heavy ML
framework) don't need to be installed unless that policy is selected.
"""
from .base_policy import PolicyModel


def create_policy(name: str, **kwargs) -> PolicyModel:
    """kwargs are forwarded to backends that take them (currently only 'diffusion':
    checkpoint_path, device, task, logger). Backends that take no arguments ignore them."""
    if name == "dummy":
        from .dummy_policy import DummyPolicyModel
        return DummyPolicyModel()
    if name == "act":
        from .act_policy import ACTPolicyModel
        return ACTPolicyModel()
    if name == "pi":
        from .pi_policy import PIPolicyModel
        return PIPolicyModel()
    if name == "diffusion":
        from .diffusion_policy import DEFAULT_TASK, DiffusionPolicyModel
        if not kwargs.get("checkpoint_path"):
            raise ValueError("policy_name 'diffusion' requires the 'checkpoint_path' parameter")
        return DiffusionPolicyModel(
            checkpoint_path=kwargs["checkpoint_path"],
            device=kwargs.get("device", "cuda"),
            task=kwargs.get("task") or DEFAULT_TASK,
            logger=kwargs.get("logger"),
        )
    raise ValueError(f"unknown policy_name '{name}' (options: dummy, act, pi, diffusion)")
