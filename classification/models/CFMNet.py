# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
try:
    from timm.layers import DropPath
except ImportError:
    try:
        from timm.models.layers import DropPath
    except ImportError:
        class DropPath(nn.Module):
            def __init__(self, drop_prob=0.):
                super().__init__()
                self.drop_prob = drop_prob

            def forward(self, x):
                if self.drop_prob == 0. or not self.training:
                    return x
                keep_prob = 1 - self.drop_prob
                shape = (x.shape[0],) + (1,) * (x.ndim - 1)
                random_tensor = keep_prob + torch.rand(
                    shape, dtype=x.dtype, device=x.device
                )
                random_tensor.floor_()
                return x.div(keep_prob) * random_tensor
from typing import List
from torch import Tensor
try:
    import antialiased_cnns
except ImportError:
    class _BlurPool(nn.Module):
        def __init__(self, channels, stride=2):
            super().__init__()
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)

        def forward(self, x):
            return self.pool(x)

    class antialiased_cnns:
        BlurPool = _BlurPool
import torch.nn.functional as F

try:
    from einops import rearrange
except ImportError:
    def rearrange(x, pattern, **kwargs):
        if pattern == 'b (head d) h w -> b head d (h w)':
            b, c, h, w = x.shape
            head = kwargs['head']
            return x.reshape(b, head, c // head, h * w)
        if pattern == 'b head d (h w) -> b (head d) h w':
            b, head, d, _ = x.shape
            h, w = kwargs['h'], kwargs['w']
            return x.reshape(b, head * d, h, w)
        if pattern == 'b c h w -> 1 (b c) h w':
            b, c, h, w = x.shape
            return x.reshape(1, b * c, h, w)
        if pattern == '1 (b c) h w -> b c h w':
            _, _, h, w = x.shape
            b, c = kwargs['b'], kwargs['c']
            return x.reshape(b, c, h, w)
        raise NotImplementedError(f"Unsupported rearrange pattern: {pattern}")
from torch.nn.init import trunc_normal_
from typing import Optional

def build_norm_layer(norm_layer, num_features, postfix=''):
    """Build a normalization layer and return its generated name."""
    if isinstance(norm_layer, str):
        nl = norm_layer.lower()
        if nl in ['bn', 'batchnorm', 'batchnorm2d']:
            layer = nn.BatchNorm2d(num_features)
            name = f'bn{postfix}'
        elif nl in ['syncbn', 'sync_batchnorm']:
            layer = nn.SyncBatchNorm(num_features)
            name = f'syncbn{postfix}'
        elif nl in ['gn', 'groupnorm']:
            num_groups = 32
            while num_features % num_groups != 0 and num_groups > 1:
                num_groups //= 2
            layer = nn.GroupNorm(num_groups, num_features)
            name = f'gn{postfix}'
        elif nl in ['ln', 'layernorm']:
            layer = nn.GroupNorm(1, num_features)
            name = f'ln{postfix}'
        else:
            raise ValueError(f"Unsupported norm_layer str: {norm_layer}")
    else:
        layer = norm_layer(num_features)
        name = layer.__class__.__name__.lower() + str(postfix)

    return name, layer


# -----TCRM----------------------------

class TCRM(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(TCRM, self).__init__()
        assert dim % num_heads == 0, 'dim must be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.linear_0 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.linear_2 = nn.Conv2d(dim, dim, 1, 1, 0)

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3,
            kernel_size=3, stride=1, padding=1,
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

    def forward(self, x):
        b, c, h, w = x.shape
        x_in = x

        dtype = x.dtype
        x = x.to(torch.float32)

        qkv = self.qkv_dwconv(self.qkv(x))      # (B, 3C, H, W)
        q, k, v = qkv.chunk(3, dim=1)

        # reshape -> (B, head, d, HW)
        q = rearrange(q, 'b (head d) h w -> b head d (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head d) h w -> b head d (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head d) h w -> b head d (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1, eps=1e-6)
        k = F.normalize(k, dim=-1, eps=1e-6)

        B, Hn, D, N = q.shape

        gate_map = self.gate(x)          # (B,1,H,W) in (0,1)
        gate_mean = gate_map.mean()

        raw_k = D * gate_mean            # [0, D]
        raw_k = torch.clamp(raw_k, min=1.0, max=float(D))
        dynamic_k = int(raw_k.detach().item())

        temp = torch.clamp(self.temperature, 0.01, 10.0)

        # (B, head, D, D)
        attn = (q @ k.transpose(-2, -1)) * temp / (D ** 0.5)

        # mask: (B, head, D, D)
        mask = torch.zeros_like(attn, device=attn.device, requires_grad=False)
        index = torch.topk(attn, k=dynamic_k, dim=-1, largest=True)[1]
        mask.scatter_(-1, index, 1.0)

        very_neg = -1e4 if attn.dtype == torch.float32 else -1e3
        attn = torch.where(mask > 0, attn, torch.full_like(attn, very_neg))

        attn = attn - attn.max(dim=-1, keepdim=True)[0]

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        base_out = attn @ v    # (B, head, D, HW)

        alpha = self.attn1 + self.attn2 + self.attn3 + self.attn4
        out = alpha * base_out

        x_out = rearrange(out, 'b head d (h w) -> b (head d) h w',
                          head=self.num_heads, h=h, w=w)

        out = self.project_out(x_out)

        out = x_in.to(torch.float32) + self.gamma * out
        out = out.to(dtype)

        return out
class LDFM(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        self.conv_0 = nn.Sequential(
            nn.Conv2d(dim,hidden_dim,3,1,1,groups=dim),
            nn.Conv2d(hidden_dim,hidden_dim,1,1,0)
        )
        self.act =nn.GELU()
        self.conv_1 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv_0(x) # 3×3Conv: (B,C,H,W)--conv_0-->(B,D,H,W)
        x = self.act(x) # (B,D,H,W)-GELU->(B,D,H,W)
        x = self.conv_1(x) # 1×1Conv: (B,D,H,W)-->(B,C,H,W)
        return x
# --------------------------------------------
# ----------------------EGPCM------------------------





def _geo_ensemble(k):
    k_hflip = k.flip([3])
    k_vflip = k.flip([2])
    k_hvflip = k.flip([2, 3])
    k_rot90 = torch.rot90(k, -1, [2, 3])
    k_rot90_hflip = k_rot90.flip([3])
    k_rot90_vflip = k_rot90.flip([2])
    k_rot90_hvflip = k_rot90.flip([2, 3])
    k = (k + k_hflip + k_vflip + k_hvflip + k_rot90 + k_rot90_hflip + k_rot90_vflip + k_rot90_hvflip) / 8
    return k


class EGPCM(nn.Module):
    def __init__(self, pdim: int, proj_dim_in: Optional[int] = None,
                 sk_size: int = 3, kernel_scale: float = 0.5,
                 lk_size: int = 13):
        super().__init__()
        self.pdim = pdim
        self.proj_dim_in = proj_dim_in if proj_dim_in is not None else pdim
        self.sk_size = sk_size
        self.lk_size = lk_size
        self.lk_padding = lk_size // 2

        hidden_dim = max(1, pdim // 2)
        self.dwc_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.pdim, hidden_dim, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(hidden_dim, pdim * self.sk_size * self.sk_size, 1, 1, 0)
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

        x1_dynamic = rearrange(x1, 'b c h w -> 1 (b c) h w')
        x1_dynamic = F.conv2d(
            x1_dynamic,
            dynamic_kernel,
            stride=1,
            padding=self.sk_size // 2,
            groups=bs * self.pdim
        )
        x1_dynamic = rearrange(
            x1_dynamic, '1 (b c) h w -> b c h w', b=bs, c=self.pdim
        )

        lk = lk_filter / (lk_filter.norm(dim=(2, 3), keepdim=True) + 1e-6)
        x1_large = F.conv2d(
            x1,
            lk,
            stride=1,
            padding=self.lk_padding,
            groups=self.pdim
        )
        x1_new = x1_dynamic + x1_large

        x_out = torch.cat([x1_new, x2], dim=1)

        x_out = x_in.to(torch.float32) + self.gamma * (x_out - x_in.to(torch.float32))
        x_out = x_out.to(dtype)
        return x_out

class DownsampleBlock(nn.Module):
    def __init__(self, dim, norm_layer, act_layer):
        super().__init__()
        self.dim = dim
        self.outdim = dim * 2

        self.conv = nn.Conv2d(dim, dim*2, kernel_size=3, stride=1, padding=1, groups=dim)

        self.conv_c = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=2, padding=1, groups=dim*2)
        self.act_c = act_layer()
        self.norm_c = build_norm_layer(norm_layer, dim*2)[1]

        self.max_m = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.norm_m = build_norm_layer(norm_layer, dim*2)[1]

        self.fusion = nn.Conv2d(dim*4, self.outdim, kernel_size=1, stride=1)

    def forward(self, x):  # x: [B, C, H, W]
        x = self.conv(x)

        max = self.norm_m(self.max_m(x))                  # [B, 2C, H/2, W/2]

        conv = self.norm_c(self.act_c(self.conv_c(x)))    # [B, 2C, H/2, W/2]

        x = torch.cat([conv, max], dim=1)

        x = self.fusion(x)                                # [B, 2C, H/2, W/2]

        return x

class SSRM(nn.Module):
    def __init__(self, channel, att_kernel, norm_layer):
        super().__init__()
        att_padding = att_kernel // 2
        self.gate_fn = nn.Sigmoid()
        self.channel = channel

        self.max_m1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.max_m2 = antialiased_cnns.BlurPool(channel, stride=3)

        self.H_att1 = nn.Conv2d(channel, channel, (att_kernel, 3), 1, (att_padding, 1),
                                groups=channel, bias=False)
        self.V_att1 = nn.Conv2d(channel, channel, (3, att_kernel), 1, (1, att_padding),
                                groups=channel, bias=False)

        self.H_att2 = nn.Conv2d(channel, channel, (att_kernel, 3), 1, (att_padding, 1),
                                groups=channel, bias=False)
        self.V_att2 = nn.Conv2d(channel, channel, (3, att_kernel), 1, (1, att_padding),
                                groups=channel, bias=False)

        self.norm = build_norm_layer(norm_layer, channel)[1]

    def forward(self, x):
        x_tem = self.max_m1(x)
        x_tem = self.max_m2(x_tem)

        x_h1 = self.H_att1(x_tem)
        x_w1 = self.V_att1(x_tem)

        x_h2 = self.inv_h_transform(self.H_att2(self.h_transform(x_tem)))
        x_w2 = self.inv_v_transform(self.V_att2(self.v_transform(x_tem)))

        att = self.norm(x_h1 + x_w1 + x_h2 + x_w2)

        out = x[:, :self.channel, :, :] * F.interpolate(
            self.gate_fn(att),
            size=(x.shape[-2], x.shape[-1]),
            mode='nearest'
        )
        return out

    def h_transform(self, x):
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x

    def inv_h_transform(self, x):
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x

    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1)
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x.permute(0, 1, 3, 2)


class CFMBlock(nn.Module):

    def __init__(self,
                 dim,
                 stage,
                 att_kernel,
                 mlp_ratio,
                 drop_path,
                 act_layer,
                 norm_layer,
                 egpcm_sk_size: int = 3,
                 egpcm_lk_size: int = 13,
                 egpcm_ratios=(1/8, 1/4, 0.5, 0.5)
                 ):
        super().__init__()
        self.stage = stage
        self.stage_id = stage

        self.module_order = ('TCRM', 'SSRM', 'LDFM', 'EGPCM')
        if dim % len(self.module_order) != 0:
            raise ValueError(f'dim={dim} must be divisible by the four CFMNet modules')
        self.dim_split = dim // len(self.module_order)
        self.split_sizes = [self.dim_split] * len(self.module_order)
        self.branch_dims = {name: self.dim_split for name in self.module_order}
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer: List[nn.Module] = [
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            build_norm_layer(norm_layer, mlp_hidden_dim)[1],
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]
        self.mlp = nn.Sequential(*mlp_layer)

        self.SSRM = SSRM(self.branch_dims['SSRM'], att_kernel, norm_layer)
        self.TCRM = TCRM(self.branch_dims['TCRM'], num_heads=4, bias=True)
        self.LDFM = LDFM(self.branch_dims['LDFM'], growth_rate=2.0)

        self.C = self.branch_dims['EGPCM']
        ratio = egpcm_ratios[stage]
        C1_min = 8
        self.C1 = max(C1_min, int(self.C * ratio))
        self.C2 = self.C - self.C1
        self.EGPCM = EGPCM(
            pdim=self.C1,
            proj_dim_in=self.C2,
            sk_size=egpcm_sk_size,
            lk_size=egpcm_lk_size,
        )
        plk = torch.randn(self.C1, 1, egpcm_lk_size, egpcm_lk_size)
        self.plk_filter = nn.Parameter(_geo_ensemble(plk))

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
        x_out = self.mlp(x_att)
        x = shortcut + self.norm1(self.drop_path(x_out))
        return x



class BasicStage(nn.Module):
    """A stack of CFM blocks at one feature resolution."""
    def __init__(self,
                 dim,
                 stage,
                 depth,
                 att_kernel,
                 mlp_ratio,
                 drop_path,
                 norm_layer,
                 act_layer,
                 egpcm_sk_size: int = 3,
                 egpcm_lk_size: int = 13,
                 egpcm_ratios=(1/8, 1/4, 0.5, 0.5)
                 ):
        super().__init__()

        blocks_list = []
        for i in range(depth):
            block = CFMBlock(
                dim=dim,
                stage=stage,
                att_kernel=att_kernel,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                egpcm_sk_size=egpcm_sk_size,
                egpcm_lk_size=egpcm_lk_size,
                egpcm_ratios=egpcm_ratios,
            )
            blocks_list.append(block)

        self.blocks = nn.Sequential(*blocks_list)
        self.stage_id = stage

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)



class Stem(nn.Module):
    """Patch embedding stem with a stride-4 convolution."""
    def __init__(self, in_chans, stem_dim, norm_layer):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, stem_dim,
                              kernel_size=4, stride=4, bias=False)
        if norm_layer is not None:
            self.norm = build_norm_layer(norm_layer, stem_dim)[1]
        else:
            self.norm = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.proj(x))


class CFMNet(nn.Module):
    """Canonical CFMNet backbone used in the paper experiments."""
    def __init__(self,
                 in_chans=3,
                 num_classes=1000,
                 stem_dim=96,
                 depths=(1, 4, 4, 2),
                 att_kernel=(11, 11, 11, 11),
                 norm_layer=nn.BatchNorm2d,
                 act_layer=nn.ReLU,
                 mlp_ratio=2.,
                 stem_norm=True,
                 feature_dim=1280,
                 drop_path_rate=0.1,
                 egpcm_sk_size: int = 3,
                 egpcm_lk_size: int = 13,
                 egpcm_ratios=(1/8, 1/4, 0.5, 0.5),
                 **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_stages = len(depths)
        self.num_features = int(stem_dim * 2 ** (self.num_stages - 1))

        self.stem = Stem(
            in_chans=in_chans, stem_dim=stem_dim,
            norm_layer=norm_layer if stem_norm else None
        )

        dpr = [x.item()
               for x in torch.linspace(0, drop_path_rate, sum(depths))]

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
                egpcm_sk_size=egpcm_sk_size,
                egpcm_lk_size=egpcm_lk_size,
                egpcm_ratios=egpcm_ratios,
            )
            stages_list.append(stage)

            if i_stage < self.num_stages - 1:
                stages_list.append(
                    DownsampleBlock(dim=int(stem_dim * 2 ** i_stage),
                                    norm_layer=norm_layer,
                                    act_layer=act_layer)
                )

        self.stages = nn.Sequential(*stages_list)

        self.avgpool_pre_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.num_features, feature_dim, 1, bias=False),
            act_layer()
        )
        self.head = nn.Linear(feature_dim, num_classes) \
            if num_classes > 0 else nn.Identity()

        self.apply(self.cls_init_weights)

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

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.avgpool_pre_head(x)
        return self.head(torch.flatten(x, 1))
