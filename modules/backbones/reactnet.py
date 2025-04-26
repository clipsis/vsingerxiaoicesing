# ReactNet: Diffusion MoE RevNet Singing Voice Synthesis
# Copyright (C) 2025 Project Vsinger-Xiaoice Group of ICA Co.Ltd.
# Released in WTFPL License.

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from modules.commons.common_layers import SinusoidalPosEmb, SwiGLU, Transpose
from utils.hparams import hparams

class MoEContainer(nn.Module):
    def __init__(self, experts: list, dim: int, actives: int):
        super().__init__()
        self.dim_size = dim
        self.experts = nn.ModuleList(experts)
        self.num_experts = len(experts)
        self.num_actives = actives
        self.router = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, len(experts))
        )
    def forward(self, x):
        num_batchs = x.shape[0]
        num_frames = x.shape[1]
        x = x.reshape(x.shape[0] * x.shape[1], self.dim_size)
        # Router
        route_value = self.router(x)
        if self.training:
            route_indice = torch.topk(route_value + torch.randn(route_value.shape, device=x.device) * 0.1, k=self.num_actives, dim=1).indices
        else:
            route_indice = torch.topk(route_value, k=self.num_actives, dim=1).indices
        route_mask = torch.scatter(
            input=torch.full_like(route_value, False, dtype=torch.bool), dim=1,
            index=route_indice,
            src=torch.full_like(route_indice, True, dtype=torch.bool)
        )
        # Add Safety Token
        x = torch.cat([
            torch.zeros(1, self.dim_size, device=x.device), x
        ], dim=0)
        route_mask = torch.cat([
            torch.ones(1, self.num_experts, dtype=torch.bool, device=x.device), route_mask
        ], dim=0)
        route_value = torch.cat([
            torch.zeros(1, self.num_experts, device=x.device), route_value
        ], dim=0)
        # Softmax
        route_weight = F.softmax(torch.where(route_mask, route_value, float("-inf")), dim=1)
        # Experts compute
        y = x
        for e_id, e_module in enumerate(self.experts):
            e_indice = torch.nonzero(route_mask[:, e_id])
            e_input = torch.gather(
                input=x, dim=0, index=e_indice.repeat(1, self.dim_size)
            )
            e_output = e_module(e_input)
            e_weight = torch.gather(
                input=route_weight[:, e_id], dim=0,
                index=e_indice.squeeze(1)
            )[:, None]
            y = torch.scatter_add(
                input=y, dim=0,
                index=e_indice.repeat(1, self.dim_size),
                src=e_output * e_weight
            )
        # Remove Safety Token
        y = y[1:, :]
        return y.reshape(num_batchs, num_frames, self.dim_size)


class ReactNetBlock(nn.Module):
    def __init__(self, dim, expansion_factor, kernel_size=31, dropout=0., num_experts=5, num_actives=2, senet_squeeze=8):
        super().__init__()
        inner_dim = int(dim * expansion_factor)
        if float(dropout) > 0.:
            _dropout = nn.Dropout(dropout)
        else:
            _dropout = nn.Identity()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            Transpose((1, 2)),
            nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim),
            Transpose((1, 2)),
            MoEContainer(
                [
                    nn.Sequential(
                        nn.Linear(dim, inner_dim * 2),
                        SwiGLU(),
                        nn.Linear(inner_dim, inner_dim * 2),
                        SwiGLU(),
                        nn.Linear(inner_dim, dim)
                    ) for _ in range(0, num_experts)
                ],
                dim=dim, actives=num_actives
            ),
            _dropout
        )
        self.senet = nn.Sequential(
            nn.Linear(dim, dim // senet_squeeze),
            nn.SiLU(),
            nn.Linear(dim // senet_squeeze, dim),
            nn.Sigmoid()
        )
        self.senet_res = nn.Parameter(torch.ones(dim), requires_grad=True)
    def forward(self, res):
        x = self.norm(res)
        y = self.net(x)
        gate = self.senet(torch.mean(x * self.senet_res[None, None, :] + y, dim=-2))
        return res + y * gate

class RevNetAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, layers):
        ctx.layers = layers
        ctx.rng_seed = [ random.randint(0, 65535) for _ in layers ]
        x1, x2 = torch.split(x, [x.size(-1) // 2, x.size(-1) // 2], dim=-1)
        for layer, seed in zip(layers, ctx.rng_seed):
            torch.cuda.manual_seed_all(seed)
            x2 = x2 + layer(x1)
            x1, x2 = x2, x1
        ctx.save_for_backward(x1, x2)
        return torch.cat([x1, x2], dim=-1)
    @staticmethod
    def backward(ctx, grad_x):
        x1, x2 = ctx.saved_tensors
        grad_x1, grad_x2 = torch.split(grad_x, [x1.size(-1), x2.size(-1)], dim=-1)
        for layer, seed in list(zip(ctx.layers, ctx.rng_seed))[::-1]:
            # Switch
            x1, x2 = x2, x1
            grad_x1, grad_x2 = grad_x2, grad_x1
            # Block
            with torch.enable_grad():
                x1_4g = x1.clone().detach().requires_grad_(True)
                torch.cuda.manual_seed_all(seed)
                dx2 = layer(x1_4g)
            dx2.backward(grad_x2)
            x2 = x2 - dx2
            grad_x1 = grad_x1 + x1_4g.grad
        return torch.cat([grad_x1, grad_x2], dim=-1), None

def RevNetInfer(x, layers):
    x1, x2 = torch.split(x, [x.size(-1) // 2, x.size(-1) // 2], dim=-1)
    for layer in layers:
        x2 += layer(x1)
        x1, x2 = x2, x1
    return torch.cat([x1, x2], dim=-1)

class ReactNet(nn.Module):
    def __init__(self, in_dims, n_feats, *, num_layers=6, num_channels=512, expansion_factor=1, kernel_size=31,
                 dropout=0.0, num_experts=5, num_active_experts=2, senet_squeeze=8):
        super().__init__()
        self.in_dims = in_dims
        self.n_feats = n_feats
        self.input_projection = nn.Linear(in_dims * n_feats, num_channels * 2)
        self.conditioner_projection = nn.Linear(hparams['hidden_size'], num_channels * 2)
        self.diffusion_embedding = nn.Sequential(
            SinusoidalPosEmb(num_channels),
            nn.Linear(num_channels, num_channels * 4),
            nn.GELU(),
            nn.Linear(num_channels * 4, num_channels * 2),
        )
        self.residual_layers = nn.ModuleList(
            [
                ReactNetBlock(
                    dim=num_channels,
                    expansion_factor=expansion_factor,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    num_experts=num_experts,
                    num_actives=num_active_experts,
                    senet_squeeze=senet_squeeze
                )
                for i in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(num_channels * 2)
        self.output_projection = nn.Linear(num_channels * 2, in_dims * n_feats)
        nn.init.kaiming_normal_(self.input_projection.weight)
        nn.init.kaiming_normal_(self.conditioner_projection.weight)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, spec, diffusion_step, cond):
        """
        :param spec: [B, F, M, T]
        :param diffusion_step: [B, 1]
        :param cond: [B, H, T]
        :return:
        """

        if self.n_feats == 1:
            x = spec[:, 0]  # [B, M, T]
        else:
            x = spec.flatten(start_dim=1, end_dim=2)  # [B, F x M, T]

        x = self.input_projection(x.transpose(1, 2)) # [B, T, F x M]
        x = x + self.conditioner_projection(cond.transpose(1, 2))
        x = x + self.diffusion_embedding(diffusion_step).unsqueeze(1)

        if torch.onnx.is_in_onnx_export() or not torch.is_grad_enabled():
            x = RevNetInfer(x, self.residual_layers)
        else:
            x = RevNetAutograd.apply(x, self.residual_layers)

        # post-norm
        x = self.norm(x)

        # output projection
        x = self.output_projection(x).transpose(1, 2)  # [B, 128, T]

        if self.n_feats == 1:
            x = x[:, None, :, :]
        else:
            # This is the temporary solution since PyTorch 1.13
            # does not support exporting aten::unflatten to ONNX
            # x = x.unflatten(dim=1, sizes=(self.n_feats, self.in_dims))
            x = x.reshape(-1, self.n_feats, self.in_dims, x.shape[2])
        return x
