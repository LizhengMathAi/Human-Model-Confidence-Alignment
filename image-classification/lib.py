import torch
from torch import nn
import torchvision.transforms as T
import torchvision.transforms.functional as F

# from torchvision.models.mobilenetv2 import MobileNet_V2_Weights

def disable_bn_update(m):
    if isinstance(m, nn.modules.batchnorm._BatchNorm):
        m.track_running_stats = False


class DownScaleTransform:
    """
    Downscale image with magnitude in [0, 1].

    mag = 0.0 -> scale = 1.0
    mag = 1.0 -> scale = min_scale (e.g. 0.25)
    """
    def __init__(
        self,
        min_scale=0.1,
        # min_scale=0.2,  # ratio=0.3286
        # min_scale=0.25,  # ratio: 0.3264
        # min_scale=0.5,  # ratio: 0.4236
        max_scale=1.0,
        interpolation=F.InterpolationMode.BILINEAR,
    ):
        assert 0 < min_scale <= max_scale <= 1.0
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.interpolation = interpolation

    def __call__(self, img, mag):
        # mag ∈ [0, 1]
        scale = self.max_scale - mag * (self.max_scale - self.min_scale)
        _, h, w = img.shape  # Tensor (C, H, W)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        # downscale
        img_small = F.resize(img, (new_h, new_w), interpolation=self.interpolation)

        # resize back to original resolution
        img = F.resize(img_small, (h, w), interpolation=self.interpolation)
        return img


def causal_collate_fn(iterations=4):

    def func(batch):
        images, labels = zip(*batch)

        # normalized magnitude from low → high
        mags = torch.linspace(0, 1, iterations)
        transform = DownScaleTransform()

        images = [transform(img, mag=mags[j].item()) for img in images for j in range(iterations)]
        labels = [label for label in labels for j in range(iterations)]

        return torch.stack(images), torch.tensor(labels)
    return func


def causal_criterion(label_smoothing, iterations=4, gamma=0.9, beta=0.1):
    def func(prediction, target):
        loss_seq = torch.nn.functional.cross_entropy(prediction, target, reduction='none', label_smoothing=label_smoothing).view(-1, iterations)
        violations = torch.relu(loss_seq[:, :-1] - loss_seq[:, 1:].detach())  # max(0, l_t - l_{t+1}). if l_t > l_{t+1}, detect a random guessing on t-step
        ratio = (violations > 0).sum() / violations.numel()

        # # Option A: strongest signal, good for most cases
        # loss = violations.mean()

        # Option B: very sparse supervision, only cares about existence of violation
        loss = violations.max(dim=-1).values.mean()
        
        # # Option C: smooth-ish version, penalizes bigger drops more
        # loss = torch.square(violations).sum(dim=-1).mean()
        
        # # Option D: take the first positive value
        # mask = violations > 0
        # # Find first positive violation index for each sample
        # first_pos_idx = torch.argmax(mask.float(), dim=-1)
        # # Get the violation values at those indices
        # first_violations = torch.gather(violations, -1, first_pos_idx.unsqueeze(-1)).squeeze(-1)
        # # Only use samples that have at least one positive violation
        # has_violation = mask.any(dim=-1)
        # loss = torch.where(has_violation, first_violations, torch.zeros_like(first_violations)).mean()

        return loss, ratio

        # loss = loss_seq[0]
        # for k in range(1, iterations):
        #     penalty = torch.abs(loss_seq[k] - loss_seq[k-1].detach())
        #     loss = loss + gamma**k * ((1-beta) * loss_seq[k] + beta * penalty)
        # return loss.mean()
    return func



# def violation_loss(losses: torch.Tensor) -> torch.Tensor:
#     """
#     losses: shape [n] or [batch, n]
#     Returns a scalar loss that is
#     •  0                whenever losses is non-decreasing
#     •  > 0 (and has gradient) whenever there is at least one decrease
#     """
#     diffs = losses[..., 1:] - losses[..., :-1]          # l_{t+1} - l_t
#     violations = torch.relu(-diffs)                     # max(0, l_t - l_{t+1})
#     # Option A: strongest signal, good for most cases
#     loss = violations.sum(dim=-1)
    
#     # Option B: very sparse supervision, only cares about existence of violation
#     # loss = violations.max(dim=-1).values
    
#     # Option C: smooth-ish version, penalizes bigger drops more
#     # loss = (violations ** 2).sum(dim=-1)               # or .mean()
    
#     return loss.mean()   # or .sum(), depending on your reduction preference