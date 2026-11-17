"""
Real-Time Chunking (Black et al. 2025, "Real-Time Execution of Action Chunking Flow Policies") for
LeRobot's Diffusion Policy (epsilon-prediction U-Net + DDIM/DDPM scheduler).

lerobot.policies.rtc only implements RTC for flow-matching policies (pi0, pi05, smolvla): it guides
the velocity field. This is the same inference-time guidance written for an epsilon-prediction
diffusion sampler. At each denoising step:

    x0_hat = (x_t - sigma * eps(x_t)) / alpha                  alpha = sqrt(abar_t), sigma = sqrt(1 - abar_t)
    g      = J^T (W * (Y - x0_hat)),   J = d x0_hat / d x_t     (one vector-Jacobian product through the U-Net)
    x0'    = x0_hat + sigma * min(beta, 1 / (alpha * sigma)) * g
    eps'   = (x_t - alpha * x0') / sigma  -> scheduler.step(eps', t, x_t)

Y is the previous chunk's not-yet-executed actions, re-indexed to the new chunk's timesteps, and W
the RTC prefix weights: 1 for the actions that will execute while this chunk is computed (frozen),
decaying linearly to 0 over the rest of the overlap, 0 after it. The step size is RTC's
min(beta, (1-tau)/tau * 1/r^2) guidance weight with the rectified-flow tau mapped onto the VP path
(tau = alpha / (alpha + sigma)); with alpha^2 + sigma^2 = 1 it reduces to the expression above.
Everything is in the policy's normalized action space.
"""
import torch
from lerobot.configs import RTCAttentionSchedule
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.modeling_rtc import RTCProcessor


def prefix_weights(frozen: int, overlap_end: int, horizon: int, schedule: str = "linear") -> torch.Tensor:
    """(horizon,) RTC weights: 1 for [0, frozen), decaying over [frozen, overlap_end), 0 after
    (lerobot's RTCProcessor.get_prefix_weights)."""
    proc = RTCProcessor(RTCConfig(prefix_attention_schedule=RTCAttentionSchedule(schedule.upper())))
    w = proc.get_prefix_weights(frozen, overlap_end, horizon)
    return w[:horizon]


def guided_conditional_sample(diffusion, global_cond, prefix, weights, max_guidance_weight=50.0, generator=None):
    """DiffusionModel.conditional_sample with RTC guidance.

    diffusion: lerobot DiffusionModel (policy.diffusion); global_cond: (B, cond_dim) from
    diffusion._prepare_global_conditioning; prefix: (B, horizon, action_dim) normalized target
    actions (anything where weights == 0); weights: (horizon,).
    Returns (B, horizon, action_dim) normalized actions.
    """
    cfg = diffusion.config
    sched = diffusion.noise_scheduler
    device, dtype = global_cond.device, global_cond.dtype
    batch = global_cond.shape[0]
    x = torch.randn((batch, cfg.horizon, cfg.action_feature.shape[0]), dtype=dtype, device=device,
                    generator=generator)
    w = weights.to(device=device, dtype=dtype).view(1, -1, 1)
    prefix = prefix.to(device=device, dtype=dtype)
    abar_all = sched.alphas_cumprod.to(device=device, dtype=dtype)

    sched.set_timesteps(diffusion.num_inference_steps)
    for t in sched.timesteps:
        abar = abar_all[t]
        alpha, sigma = abar.sqrt(), (1 - abar).sqrt()
        t_batch = torch.full((batch,), t, dtype=torch.long, device=device)
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            eps = diffusion.unet(x_in, t_batch, global_cond=global_cond)
            x0_hat = (x_in - sigma * eps) / alpha
            err = (prefix - x0_hat) * w
            (g,) = torch.autograd.grad(x0_hat, x_in, grad_outputs=err.detach())
        step = sigma * torch.clamp(1.0 / (alpha * sigma), max=max_guidance_weight)
        x0_guided = x0_hat.detach() + step * g
        eps_guided = (x - alpha * x0_guided) / sigma
        x = sched.step(eps_guided, t, x, generator=generator).prev_sample
    return x
