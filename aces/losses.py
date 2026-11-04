from __future__ import annotations
import math
import torch
import torch.nn.functional as F


def supervised_loss(output, target, auxiliary_weight=0.1):
    logits = output["logits"].float()
    ce = F.cross_entropy(logits, target)
    p = logits.softmax(1)
    y = F.one_hot(target, 4).permute(0, 3, 1, 2).float()
    dice = (2*(p[:, 1:]*y[:, 1:]).sum((0, 2, 3)) + 1e-6) / ((p[:, 1:]+y[:, 1:]).sum((0, 2, 3)) + 1e-6)
    aux = output.get("aux")
    aux_loss = logits.new_zeros(())
    if aux is not None:
        pooled = F.adaptive_avg_pool2d(y, aux.shape[-2:])
        aux_loss = -(pooled * aux.float().log_softmax(1)).sum(1).mean()
    return ce + 1 - dice.mean() + auxiliary_weight*aux_loss


def random_geometry(batch, generator=None):
    """affine_grid matrices map augmented output coordinates to canonical input coordinates."""
    b, _, _, _ = batch.shape
    r = torch.rand(b, 4, device=batch.device, generator=generator)
    angle = (r[:, 0]*20-10)*math.pi/180
    scale = 0.9 + r[:, 1]*0.2
    theta = torch.zeros(b, 2, 3, device=batch.device)
    theta[:, 0, 0] = angle.cos()/scale
    theta[:, 0, 1] = -angle.sin()/scale
    theta[:, 1, 0] = angle.sin()/scale
    theta[:, 1, 1] = angle.cos()/scale
    theta[:, :, 2] = (r[:, 2:4]*2-1)*0.1
    matrix = torch.eye(3, device=batch.device).expand(b, 3, 3).clone()
    matrix[:, :2] = theta
    inv = torch.linalg.inv(matrix)[:, :2]
    return theta, inv


def warp(x, theta, mode="bilinear"):
    grid = F.affine_grid(theta, x.size(), align_corners=False)
    return F.grid_sample(x, grid, mode=mode, align_corners=False, padding_mode="zeros")


def intensity(x, strong=False):
    gamma = torch.empty(x.shape[0], 1, 1, 1, device=x.device).uniform_(0.7 if strong else 0.9, 1.5 if strong else 1.1)
    y = x.clamp(0, 1).pow(gamma)
    if strong:
        sigma = torch.rand(x.shape[0], 1, 1, 1, device=x.device)*0.03
        y = y + torch.randn_like(y)*sigma
    return y.clamp(0, 1)


def augment_supervised(image, mask):
    theta, _ = random_geometry(image)
    return intensity(warp(image, theta)), warp(mask[:, None].float(), theta, "nearest")[:, 0].long()


@torch.no_grad()
def pseudo_targets(teacher, canonical_image, view, prior, family, confidence=0.9, score_threshold=0.5):
    theta, inv = random_geometry(canonical_image)
    geom = warp(canonical_image, theta)
    weak, strong = intensity(geom), intensity(geom, strong=True)
    prob = teacher(weak, view)["logits"].float().softmax(1)
    conf, pseudo = prob.max(1)
    in_field = warp(torch.ones_like(canonical_image), theta)[:, 0] > 0.999
    weights = (conf >= confidence).float()*in_field
    score = torch.ones(canonical_image.shape[0], device=canonical_image.device)
    if family in {"gaussian", "student_t"}:
        if prior is None:
            raise ValueError("Anatomical filter needs a fitted training-only prior")
        restored_prob = warp(prob, inv)
        restored_mask = restored_prob.argmax(1)
        score = prior.score(restored_mask, view, family)
        weights *= ((score >= score_threshold)*score)[:, None, None]
    return strong.detach(), pseudo.detach(), weights.detach(), score.detach()


def unsupervised_loss(logits, pseudo, weights):
    losses = F.cross_entropy(logits.float(), pseudo, reduction="none")
    # Explicitly zero when no valid pixels; gradients are finite and zero.
    return (losses*weights).sum()/weights.sum().clamp_min(1)


@torch.no_grad()
def update_teacher(teacher, student, decay):
    for a, b in zip(teacher.parameters(), student.parameters()):
        a.lerp_(b.detach(), 1-decay)
    for a, b in zip(teacher.buffers(), student.buffers()):
        a.copy_(b)
