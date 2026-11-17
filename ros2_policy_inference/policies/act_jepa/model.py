"""ACT-JEPA inference network: only the parts used at deployment (context encoder, action
decoder, action head). The target encoder and predictor only matter during training and are
not part of the exported checkpoint.

This mirrors the official code (github.com/act-jepa/act-jepa: models/act_model_original.py,
models/act_jepa.py, transformer_utils/original.py) module for module, so the exported
state_dict loads with strict=True. It's written in plain torch, plus transformers' ResNetModel
built from a saved config (no download), because the training code targets transformers 4.x
and this image ships 5.x.

Checkpoints come from delta_robot_friday/JEPA/act-jepa/scripts/export_for_ros.py.
"""
import json
import os

import torch
from torch import nn

_ACT = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}


class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.up_proj = nn.Linear(c["hidden_size"], c["intermediate_size"], bias=False)
        self.act_fn = _ACT[c["hidden_act"]]()
        self.dropout1 = nn.Dropout(c["attention_dropout"])
        self.down_proj = nn.Linear(c["intermediate_size"], c["hidden_size"], bias=False)
        self.dropout2 = nn.Dropout(c["attention_dropout"])

    def forward(self, x):
        return self.dropout2(self.down_proj(self.dropout1(self.act_fn(self.up_proj(x)))))


class _Attention(nn.Module):
    """SelfAttention / CrossAttention of the original: one nn.MultiheadAttention, no mask at inference."""

    def __init__(self, c):
        super().__init__()
        self.attention_interface = nn.MultiheadAttention(
            c["hidden_size"], c["num_attention_heads"], c["attention_dropout"], batch_first=True)

    def forward(self, x, memory=None):
        kv = x if memory is None else memory
        return self.attention_interface(x, kv, kv, need_weights=False)[0]


class EncoderLayer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.self_attn = _Attention(c)
        self.norm1 = nn.LayerNorm(c["hidden_size"])
        self.mlp = MLP(c)
        self.norm2 = nn.LayerNorm(c["hidden_size"])

    def forward(self, x):  # post-norm, as in the original
        x = self.norm1(x + self.self_attn(x))
        return self.norm2(x + self.mlp(x))


class DecoderLayer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.self_attn = _Attention(c)
        self.norm1 = nn.LayerNorm(c["hidden_size"])
        self.cross_attn = _Attention(c)
        self.norm2 = nn.LayerNorm(c["hidden_size"])
        self.mlp = MLP(c)
        self.norm3 = nn.LayerNorm(c["hidden_size"])

    def forward(self, x, memory):
        x = self.norm1(x + self.self_attn(x))
        x = self.norm2(x + self.cross_attn(x, memory))
        return self.norm3(x + self.mlp(x))


class PositionalEmbedding(nn.Module):
    """2D sinusoidal embedding, copied as is (including its 2 * 3.14)."""

    def __init__(self, hidden_size):
        super().__init__()
        self.dim = hidden_size // 2
        self._2pi = 2 * 3.14
        self._eps = 1e-6

    def forward(self, x):
        mask = torch.ones_like(x[0, :1])
        y_pos = mask.cumsum(1, dtype=torch.float32)
        x_pos = mask.cumsum(2, dtype=torch.float32)
        y_pos = y_pos / (y_pos[:, -1:, :] + self._eps) * self._2pi
        x_pos = x_pos / (x_pos[:, :, -1:] + self._eps) * self._2pi
        denominator = 10000 ** (2 * (torch.arange(self.dim, dtype=torch.float32, device=x.device) // 2) / self.dim)
        x_pos = x_pos.unsqueeze(-1) / denominator
        y_pos = y_pos.unsqueeze(-1) / denominator
        x_embed = torch.stack((x_pos[..., 0::2].sin(), x_pos[..., 1::2].cos()), dim=-1).flatten(3)
        y_embed = torch.stack((y_pos[..., 0::2].sin(), y_pos[..., 1::2].cos()), dim=-1).flatten(3)
        return torch.cat((y_embed, x_embed), dim=3).permute(0, 3, 1, 2)


class ContextEncoder(nn.Module):
    def __init__(self, c, resnet_config):
        super().__init__()
        from transformers import ResNetConfig, ResNetModel
        h = c["hidden_size"]
        self.state_emb = nn.Linear(c["state_dim"], h)
        self.img_emb = ResNetModel(ResNetConfig(**resnet_config))  # weights come from the checkpoint
        self.img_proj = nn.Conv2d(self.img_emb.config.hidden_sizes[-1], h, kernel_size=1)
        self.pos_emb = PositionalEmbedding(h)
        self.task_emb = nn.Embedding(c["num_task_embeddings"], h)
        self.layers = nn.ModuleList([EncoderLayer(c) for _ in range(c["num_encoder_layers"])])
        self.norm = nn.LayerNorm(h)

        # state dims hidden from the policy (e.g. the gripper command), set to 0 = the normalised mean
        self.state_mask = list(c.get("context_state_mask") or [])

    def _img_tokens(self, image):
        x = self.img_emb(pixel_values=image).last_hidden_state   # (B, 512, h, w)
        x = self.img_proj(x)
        x = x + self.pos_emb(x)
        return x.flatten(2).transpose(1, 2)                      # (B, h*w, C)

    def forward(self, images, state, task_index):
        """images: one (B, 3, H, W) tensor per camera, in config["image_keys"] order. All cameras
        share the backbone, projection and positional embedding; their tokens are concatenated."""
        if self.state_mask:
            state = state.clone()
            state[..., self.state_mask] = 0.0
        img_tokens = torch.cat([self._img_tokens(im) for im in images], dim=1)
        tokens = torch.cat([self.task_emb(task_index)[:, None], self.state_emb(state)[:, None], img_tokens], dim=1)
        for layer in self.layers:
            tokens = layer(tokens)
        return self.norm(tokens)


class ActionDecoder(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.zeros(c["chunk_size"], c["hidden_size"]))
        self.layers = nn.ModuleList([DecoderLayer(c) for _ in range(c["num_decoder_layers"])])
        self.norm = nn.LayerNorm(c["hidden_size"])

    def forward(self, memory):
        x = self.pos_emb.unsqueeze(0).expand(memory.shape[0], -1, -1)
        for layer in self.layers:
            x = layer(x, memory)
        return self.norm(x)


class ActJepaInference(nn.Module):
    """(list of images in [0, 1], each (B, 3, H, W), in image_keys order; normalised state
    (B, state_dim); task index (B,)) -> normalised action chunk (B, chunk_size, action_dim)."""

    def __init__(self, config, resnet_config):
        super().__init__()
        self.encoder = ContextEncoder(config, resnet_config)
        self.decoder = ActionDecoder(config)
        self.action_head = nn.Linear(config["hidden_size"], config["action_dim"])

    def forward(self, images, state, task_index):
        return self.action_head(self.decoder(self.encoder(images, state, task_index)))


def load_network(ckpt_path, device):
    """Exported checkpoint dir -> (network in eval mode, its config, normalization stats)."""
    with open(os.path.join(ckpt_path, "config.json")) as f:
        net_cfg = json.load(f)
    if net_cfg.get("format") != "act_jepa_inference_v1":
        raise ValueError(f"{ckpt_path}: not an act_jepa export (format={net_cfg.get('format')})")
    with open(os.path.join(ckpt_path, "resnet_config.json")) as f:
        resnet_cfg = json.load(f)
    with open(os.path.join(ckpt_path, "normalization.json")) as f:
        norm = json.load(f)
    # exports before multi-camera support have a single image_key
    net_cfg.setdefault("image_keys", [net_cfg["image_key"]] if net_cfg.get("image_key") else [])
    from safetensors.torch import load_file
    net = ActJepaInference(net_cfg, resnet_cfg)
    net.load_state_dict(load_file(os.path.join(ckpt_path, "model.safetensors"), device="cpu"), strict=True)
    return net.to(device).eval(), net_cfg, norm
