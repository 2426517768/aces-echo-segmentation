from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        groups = min(8, cout)
        while cout % groups:
            groups -= 1
        self.net = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                                 nn.GroupNorm(groups, cout), nn.ReLU(inplace=True),
                                 nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                                 nn.GroupNorm(groups, cout), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class AnatomyGraph(nn.Module):
    def __init__(self, channels, dim=128, grid=16, heads=4, conditioned=True):
        super().__init__()
        if dim % heads:
            raise ValueError("graph_dim must be divisible by graph_heads")
        self.grid, self.heads, self.dim, self.conditioned = grid, heads, dim, conditioned
        self.embed = nn.Embedding(2 if conditioned else 1, 16)
        self.proj = nn.Linear(channels + 2 + 16, dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.bias = nn.Parameter(torch.zeros(2 if conditioned else 1, heads, 4, 4))
        self.out = nn.Linear(dim, channels)

    def forward(self, features, assignments, view):
        b, c, height, width = features.shape
        g = F.adaptive_avg_pool2d(features, self.grid)
        chi = F.adaptive_avg_pool2d(assignments.softmax(1), self.grid).flatten(2).transpose(1, 2)
        nodes = g.flatten(2).transpose(1, 2)
        axis = torch.linspace(-1, 1, self.grid, device=g.device, dtype=g.dtype)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        positions = torch.stack([xx, yy], -1).reshape(1, -1, 2).expand(b, -1, -1)
        ids = view if self.conditioned else torch.zeros_like(view)
        emb = self.embed(ids).unsqueeze(1).expand(-1, self.grid**2, -1)
        x = F.relu(self.proj(torch.cat([nodes, positions, emb], -1)))
        qkv = self.qkv(x).reshape(b, -1, 3, self.heads, self.dim // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        # All four semantic probabilities, including background, have a defined dimension.
        relation_bias = torch.einsum("bmi,bhij,bnj->bhmn", chi, self.bias[ids], chi)
        relation_bias = relation_bias.to(q.dtype)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=relation_bias, dropout_p=0.0)
        y = y.transpose(1, 2).reshape(b, self.grid**2, self.dim)
        y = self.out(y).transpose(1, 2).reshape(b, c, self.grid, self.grid)
        return features + F.interpolate(y, size=(height, width), mode="bilinear", align_corners=False)


class CompactUNet(nn.Module):
    def __init__(self, base=32, graph=False, conditioned=True, graph_dim=128, graph_grid=16, graph_heads=4):
        super().__init__()
        channels = [base * 2**i for i in range(5)]
        self.enc = nn.ModuleList([ConvBlock(1 if i == 0 else channels[i-1], c)
                                  for i, c in enumerate(channels)])
        self.aux = nn.Conv2d(channels[-1], 4, 1)
        self.graph = AnatomyGraph(channels[-1], graph_dim, graph_grid, graph_heads, conditioned) if graph else None
        self.dec = nn.ModuleList([ConvBlock(channels[i+1] + channels[i], channels[i])
                                  for i in range(3, -1, -1)])
        self.head = nn.Conv2d(channels[0], 4, 1)

    def forward(self, x, view):
        skips = []
        for i, block in enumerate(self.enc):
            if i:
                x = F.max_pool2d(x, 2)
            x = block(x)
            skips.append(x)
        aux = self.aux(x)
        if self.graph is not None:
            x = self.graph(x, aux, view)
        for block, skip in zip(self.dec, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat([x, skip], dim=1))
        return {"logits": self.head(x), "aux": aux}


class SegFormerB0(nn.Module):
    """Official Transformers MiT-B0 architecture, trained from scratch on grayscale inputs."""
    def __init__(self):
        super().__init__()
        try:
            from transformers import SegformerConfig, SegformerForSemanticSegmentation
        except ImportError as e:
            raise RuntimeError("SegFormer requires requirements_optional.txt") from e
        self.net = SegformerForSemanticSegmentation(SegformerConfig(
            num_channels=1, num_labels=4, depths=[2, 2, 2, 2],
            hidden_sizes=[32, 64, 160, 256], num_attention_heads=[1, 2, 5, 8],
            decoder_hidden_size=256, classifier_dropout_prob=0.1))

    def forward(self, x, view):
        logits = self.net(pixel_values=x).logits
        return {"logits": F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False), "aux": None}


METHODS = {
    "unet": {"graph": False, "ssl": False, "filter": "none"},
    "mean_teacher": {"graph": False, "ssl": True, "filter": "confidence"},
    "graph_supervised": {"graph": True, "ssl": False, "filter": "none"},
    "graph_confidence": {"graph": True, "ssl": True, "filter": "confidence"},
    "graph_gaussian": {"graph": True, "ssl": True, "filter": "gaussian"},
    "aces_view_agnostic": {"graph": True, "conditioned": False, "ssl": True, "filter": "student_t"},
    "aces": {"graph": True, "ssl": True, "filter": "student_t"},
    "segformer_b0": {"graph": False, "ssl": False, "filter": "none"},
}


def build_model(method, cfg):
    if method not in METHODS:
        raise ValueError(f"Unsupported method {method}. AP-Net is not silently replaced by a different network.")
    if method == "segformer_b0":
        return SegFormerB0()
    opts = METHODS[method]
    return CompactUNet(base=cfg.get("base_channels", 32), graph=opts["graph"],
                       conditioned=opts.get("conditioned", True),
                       graph_dim=cfg.get("graph_dim", 128), graph_grid=cfg.get("graph_grid", 16),
                       graph_heads=cfg.get("graph_heads", 4))
