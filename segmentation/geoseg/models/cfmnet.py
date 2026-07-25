import os
import copy
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import antialiased_cnns
from einops import rearrange

try:
    from timm.layers import DropPath, trunc_normal_
except ImportError:
    from timm.models.layers import DropPath, trunc_normal_

try:
    from mmcv.cnn import build_norm_layer as _mmcv_build_norm_layer
except ImportError:
    def _mmcv_build_norm_layer(cfg, num_features, postfix=''):
        cfg = cfg or dict(type='BN')
        layer_type = cfg.get('type', 'BN') if isinstance(cfg, dict) else str(cfg)
        requires_grad = cfg.get('requires_grad', True) if isinstance(cfg, dict) else True
        if layer_type in ('BN', 'BN2d', 'BatchNorm2d'):
            layer = nn.BatchNorm2d(num_features)
        elif layer_type == 'SyncBN':
            layer = nn.SyncBatchNorm(num_features)
        elif layer_type == 'GN':
            layer = nn.GroupNorm(cfg.get('num_groups', 32), num_features)
        elif layer_type == 'LN':
            layer = nn.LayerNorm(num_features)
        else:
            raise ValueError(f"Unsupported norm layer without mmcv: {layer_type}")
        for param in layer.parameters():
            param.requires_grad = requires_grad
        return f'norm{postfix}', layer

def _load_state_dict_from_checkpoint(ckpt_path: str, map_location="cpu"):
    """Load a state dictionary and strip common wrapper prefixes."""
    ckpt = torch.load(ckpt_path, map_location=map_location)

    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        elif "model" in ckpt:
            sd = ckpt["model"]
        elif "model_state_dict" in ckpt:
            sd = ckpt["model_state_dict"]
        elif "ema_state_dict" in ckpt:
            sd = ckpt["ema_state_dict"]
        else:
            sd = ckpt
    else:
        sd = ckpt

    if not hasattr(sd, "items"):
        raise TypeError(f"Checkpoint at {ckpt_path} does not contain a state_dict-like object.")

    prefixes = ("net.", "model.", "backbone.", "module.")
    new_sd = {}
    for k, v in sd.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
        new_sd[nk] = v

    return new_sd


def build_norm_layer(cfg, num_features, postfix=''):
    """Build a normalization layer with or without MMCV installed."""
    if cfg is None:
        return f'norm{postfix}', nn.Identity()

    if isinstance(cfg, type) and issubclass(cfg, nn.Module):
        if cfg is nn.BatchNorm2d:
            cfg = dict(type='BN')
        elif cfg is nn.SyncBatchNorm:
            cfg = dict(type='SyncBN')
        elif cfg is nn.GroupNorm:
            cfg = dict(type='GN', num_groups=32)
        elif cfg is nn.LayerNorm:
            cfg = dict(type='LN')
        else:
            cfg = dict(type=cfg.__name__)
        return _mmcv_build_norm_layer(cfg, num_features, postfix)

    return _mmcv_build_norm_layer(cfg, num_features, postfix)


# ----------------------------
# Utils
# ----------------------------
def _geo_ensemble(k: torch.Tensor) -> torch.Tensor:
    """Geometric ensemble for kernels: reduce directional bias."""
    k_hflip = k.flip([3])
    k_vflip = k.flip([2])
    k_hvflip = k.flip([2, 3])
    k_rot90 = torch.rot90(k, -1, [2, 3])
    k_rot90_hflip = k_rot90.flip([3])
    k_rot90_vflip = k_rot90.flip([2])
    k_rot90_hvflip = k_rot90.flip([2, 3])
    k = (k + k_hflip + k_vflip + k_hvflip +
         k_rot90 + k_rot90_hflip + k_rot90_vflip + k_rot90_hvflip) / 8
    return k


# ----------------------------
# TCRM
# ----------------------------
class TCRM(nn.Module):
    def __init__(self, dim: int, num_heads: int, bias: bool):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1,
            groups=dim * 3, bias=bias
        )

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn_drop = nn.Dropout(0.)

        self.attn1 = nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = nn.Parameter(torch.tensor([0.2]), requires_grad=True)

        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_in = x

        dtype = x.dtype
        x = x.to(torch.float32)

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head d) h w -> b head d (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head d) h w -> b head d (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head d) h w -> b head d (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1, eps=1e-6)
        k = F.normalize(k, dim=-1, eps=1e-6)

        _, _, D, _ = q.shape

        gate_map = self.gate(x)
        gate_mean = gate_map.mean()
        raw_k = torch.clamp(D * gate_mean, min=1.0, max=float(D))
        dynamic_k = int(raw_k.detach().item())

        temp = torch.clamp(self.temperature, 0.01, 10.0)
        attn = (q @ k.transpose(-2, -1)) * temp / (D ** 0.5)

        mask = torch.zeros_like(attn, device=attn.device, requires_grad=False)
        index = torch.topk(attn, k=dynamic_k, dim=-1, largest=True)[1]
        mask.scatter_(-1, index, 1.0)

        very_neg = -1e4
        attn = torch.where(mask > 0, attn, torch.full_like(attn, very_neg))
        attn = attn - attn.max(dim=-1, keepdim=True)[0]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        base_out = attn @ v
        alpha = self.attn1 + self.attn2 + self.attn3 + self.attn4
        out = alpha * base_out

        x_out = rearrange(out, 'b head d (h w) -> b (head d) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(x_out)

        out = x_in.to(torch.float32) + self.gamma * out
        out = out.to(dtype)
        return out


# ----------------------------
# LDFM
# ----------------------------
class LDFM(nn.Module):
    def __init__(self, dim: int, growth_rate: float = 2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        self.conv_0 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1, groups=dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0)
        )
        self.act = nn.GELU()
        self.conv_1 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_0(x)
        x = self.act(x)
        x = self.conv_1(x)
        return x


# ----------------------------
# EGPCM
# ----------------------------
class EGPCM(nn.Module):
    def __init__(self, pdim: int, proj_dim_in: Optional[int] = None,
                 sk_size: int = 3, kernel_scale: float = 0.5):
        super().__init__()
        self.pdim = pdim
        self.proj_dim_in = proj_dim_in if proj_dim_in is not None else pdim
        self.sk_size = sk_size

        self.dwc_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.pdim, pdim // 2, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(pdim // 2, pdim * self.sk_size * self.sk_size, 1, 1, 0)
        )
        nn.init.zeros_(self.dwc_proj[-1].weight)
        nn.init.zeros_(self.dwc_proj[-1].bias)

        self.kernel_scale = kernel_scale
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, lk_filter: torch.Tensor) -> torch.Tensor:
        x_in = x
        dtype = x.dtype

        x = x.to(torch.float32)
        lk_filter = lk_filter.to(torch.float32)

        x1, x2 = torch.split(x, [self.pdim, x.shape[1] - self.pdim], dim=1)
        bs = x1.shape[0]

        dyn = self.dwc_proj(x[:, :self.pdim])
        dyn = torch.tanh(dyn) * self.kernel_scale
        dynamic_kernel = dyn.reshape(-1, 1, self.sk_size, self.sk_size)

        x1_ = rearrange(x1, 'b c h w -> 1 (b c) h w')
        x1_ = F.conv2d(
            x1_, dynamic_kernel, stride=1, padding=self.sk_size // 2,
            groups=bs * self.pdim
        )
        x1_ = rearrange(x1_, '1 (b c) h w -> b c h w', b=bs, c=self.pdim)

        lk = lk_filter / (lk_filter.norm(dim=(2, 3), keepdim=True) + 1e-6)
        x1_lk = F.conv2d(
            x1, lk, stride=1, padding=lk.shape[-1] // 2, groups=self.pdim
        )

        x1_new = x1_lk + x1_
        x_out = torch.cat([x1_new, x2], dim=1)

        x_out = x_in.to(torch.float32) + self.gamma * (x_out - x_in.to(torch.float32))
        x_out = x_out.to(dtype)
        return x_out


class DownsampleBlock(nn.Module):
    def __init__(self, dim, norm_layer, act_layer):
        super().__init__()
        self.outdim = dim * 2

        self.conv = nn.Conv2d(dim, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim)
        self.conv_c = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=2, padding=1, groups=dim * 2)
        self.act_c = act_layer()
        self.norm_c = build_norm_layer(norm_layer, dim * 2)[1]

        self.max_m = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.norm_m = build_norm_layer(norm_layer, dim * 2)[1]

        self.fusion = nn.Conv2d(dim * 4, self.outdim, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        maxv = self.norm_m(self.max_m(x))
        convv = self.norm_c(self.act_c(self.conv_c(x)))
        x = torch.cat([convv, maxv], dim=1)
        x = self.fusion(x)
        return x


# ----------------------------
# SSRM
# ----------------------------
class SSRM(nn.Module):
    def __init__(self, channel, att_kernel, norm_layer):
        super().__init__()
        att_padding = att_kernel // 2
        self.gate_fn = nn.Sigmoid()
        self.channel = channel

        self.max_m1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.max_m2 = antialiased_cnns.BlurPool(channel, stride=3)

        self.H_att1 = nn.Conv2d(channel, channel, (att_kernel, 3), 1, (att_padding, 1), groups=channel, bias=False)
        self.V_att1 = nn.Conv2d(channel, channel, (3, att_kernel), 1, (1, att_padding), groups=channel, bias=False)
        self.H_att2 = nn.Conv2d(channel, channel, (att_kernel, 3), 1, (att_padding, 1), groups=channel, bias=False)
        self.V_att2 = nn.Conv2d(channel, channel, (3, att_kernel), 1, (1, att_padding), groups=channel, bias=False)

        self.norm = build_norm_layer(norm_layer, channel)[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_tem = self.max_m1(x)
        x_tem = self.max_m2(x_tem)
        x_h1 = self.H_att1(x_tem)
        x_w1 = self.V_att1(x_tem)
        x_h2 = self.inv_h_transform(self.H_att2(self.h_transform(x_tem)))
        x_w2 = self.inv_v_transform(self.V_att2(self.v_transform(x_tem)))

        att = self.norm(x_h1 + x_w1 + x_h2 + x_w2)
        out = x[:, :self.channel, :, :] * F.interpolate(
            self.gate_fn(att), size=(x.shape[-2], x.shape[-1]), mode='nearest'
        )
        return out

    def h_transform(self, x):
        shape = x.size()
        x = F.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2 * shape[3] - 1)
        return x

    def inv_h_transform(self, x):
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = F.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2 * shape[-2])
        x = x[..., 0: shape[-2]]
        return x

    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = F.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2 * shape[3] - 1)
        return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1)
        x = F.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2 * shape[-2])
        x = x[..., 0: shape[-2]]
        return x.permute(0, 1, 3, 2)


# ----------------------------
# Block / Stage / Stem
# ----------------------------
class CFMBlock(nn.Module):
    def __init__(self, dim, stage, att_kernel, mlp_ratio, drop_path, act_layer, norm_layer):
        super().__init__()
        self.stage = stage
        self.module_order = ('TCRM', 'SSRM', 'LDFM', 'EGPCM')
        if dim % len(self.module_order) != 0:
            raise ValueError(f'dim={dim} must be divisible by the four CFMNet modules')
        self.dim_split = dim // len(self.module_order)
        self.split_sizes = [self.dim_split] * len(self.module_order)
        self.branch_dims = {name: self.dim_split for name in self.module_order}
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            build_norm_layer(norm_layer, mlp_hidden_dim)[1],
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        )

        self.SSRM = SSRM(self.branch_dims['SSRM'], att_kernel, norm_layer)
        self.TCRM = TCRM(self.branch_dims['TCRM'], num_heads=4, bias=True)
        self.LDFM = LDFM(self.branch_dims['LDFM'], growth_rate=2.0)

        C = self.branch_dims['EGPCM']
        if stage == 0:
            ratio = 1 / 8
        elif stage == 1:
            ratio = 1 / 4
        else:
            ratio = 0.5

        C1_min = 8
        self.C1 = max(C1_min, int(C * ratio))
        self.C2 = C - self.C1
        self.EGPCM = EGPCM(pdim=self.C1, proj_dim_in=self.C2)

        plk = torch.randn(self.C1, 1, 13, 13)
        plk = _geo_ensemble(plk)
        self.plk_filter = nn.Parameter(plk)

        self.norm1 = build_norm_layer(norm_layer, dim)[1]

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        chunks = torch.split(x, self.split_sizes, dim=1)
        branch_outputs = []
        for module_name, x_branch in zip(self.module_order, chunks):
            if module_name == 'TCRM':
                x_branch = self.TCRM(x_branch)
            elif module_name == 'SSRM':
                x_branch = self.SSRM(x_branch)
            elif module_name == 'LDFM':
                x_branch = self.LDFM(x_branch)
            elif module_name == 'EGPCM':
                x_branch = self.EGPCM(x_branch, self.plk_filter)
            else:
                raise RuntimeError(f'Unexpected active module: {module_name}')
            branch_outputs.append(x_branch)

        x_att = torch.cat(branch_outputs, 1)
        x = shortcut + self.norm1(self.drop_path(self.mlp(x_att)))
        return x


class BasicStage(nn.Module):
    def __init__(self, dim, stage, depth, att_kernel, mlp_ratio, drop_path, norm_layer, act_layer):
        super().__init__()
        blocks_list = [
            CFMBlock(
                dim=dim, stage=stage, att_kernel=att_kernel,
                mlp_ratio=mlp_ratio, drop_path=drop_path[i],
                act_layer=act_layer, norm_layer=norm_layer,
            )
            for i in range(depth)
        ]
        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x: Tensor) -> Tensor:
        x = self.blocks(x)
        return x


class Stem(nn.Module):
    def __init__(self, in_chans, stem_dim, norm_layer):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, stem_dim, kernel_size=4, stride=4, bias=False)
        if norm_layer is not None:
            self.norm = build_norm_layer(norm_layer, stem_dim)[1]
        else:
            self.norm = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(self.proj(x))
        return x


class CFMNet(nn.Module):
    """Canonical CFMNet backbone used in the paper experiments."""

    def __init__(self,
                 in_chans=3,
                 stem_dim=96,
                 depths=(1, 4, 4, 2),
                 att_kernel=(11, 11, 11, 11),
                 norm_layer=nn.BatchNorm2d,
                 act_layer=nn.ReLU,
                 mlp_ratio=2.,
                 stem_norm=True,
                 drop_path_rate=0.1,
                 init_cfg=None,
                 **kwargs):
        super().__init__()

        self.num_stages = len(depths)

        self.stem = Stem(
            in_chans=in_chans, stem_dim=stem_dim,
            norm_layer=norm_layer if stem_norm else None
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        stages_list = []
        for i_stage in range(self.num_stages):
            stage = BasicStage(
                dim=int(stem_dim * 2 ** i_stage),
                stage=i_stage,
                depth=depths[i_stage],
                att_kernel=att_kernel[i_stage],
                mlp_ratio=mlp_ratio,
                drop_path=dpr[sum(depths[:i_stage]):sum(depths[:i_stage + 1])],
                norm_layer=norm_layer,
                act_layer=act_layer,
            )
            stages_list.append(stage)

            if i_stage < self.num_stages - 1:
                stages_list.append(
                    DownsampleBlock(
                        dim=int(stem_dim * 2 ** i_stage),
                        norm_layer=norm_layer,
                        act_layer=act_layer)
                )

        self.stages = nn.Sequential(*stages_list)

        self.out_indices = [0, 2, 4, 6]
        for i_emb, i_layer in enumerate(self.out_indices):
            layer = build_norm_layer(norm_layer, int(stem_dim * 2 ** i_emb))[1]
            self.add_module(f'norm{i_layer}', layer)

        self.apply(self.cls_init_weights)
        self.init_cfg = copy.deepcopy(init_cfg)
        if self.init_cfg is not None:
            self.init_weights()

    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self):
        if not isinstance(self.init_cfg, dict) or 'checkpoint' not in self.init_cfg:
            raise ValueError('CFMNet init_cfg must contain a checkpoint path.')
        ckpt_path = self.init_cfg['checkpoint']
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f'CFMNet pretrained checkpoint not found: {ckpt_path}')

        sd = _load_state_dict_from_checkpoint(ckpt_path, map_location="cpu")
        model_sd = self.state_dict()
        filtered_sd = {}
        for key, value in sd.items():
            if key in model_sd and getattr(value, 'shape', None) != model_sd[key].shape:
                continue
            filtered_sd[key] = value
        self.load_state_dict(filtered_sd, strict=False)

    def forward(self, x: Tensor) -> List[Tensor]:
        x = self.stem(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                outs.append(norm_layer(x))
        return outs
