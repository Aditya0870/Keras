import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
import numpy as np

KEY_OUTPUT = 'metric_depth'

# Helper Functions
def extract_key(prediction, key):
    if isinstance(prediction, dict):
        return prediction[key]
    return prediction

def interpolate_if_needed(input, target, mode='bilinear', align_corners=True):
    if input.shape[-1] != target.shape[-1]:
        return nn.functional.interpolate(input, target.shape[-2:], mode=mode, align_corners=align_corners)
    return input

def debug_nan(name, tensor):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}: min={tensor.min()}, max={tensor.max()}, shape={tensor.shape}")

# Loss Functions
class SILogLoss(nn.Module):
    """SILog loss (pixel-wise)"""
    def __init__(self, beta=0.15, epsilon=1e-6):
        super(SILogLoss, self).__init__()
        self.name = 'SILog'
        self.beta = beta
        self.epsilon = epsilon

    def forward(self, input, target, mask=None, interpolate=True, return_interpolated=False):
        input = extract_key(input, KEY_OUTPUT)
        if interpolate:
            input = interpolate_if_needed(input, target)
        intr_input = input

        if target.ndim == 3:
            target = target.unsqueeze(1)

        if mask is not None:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            input, target = input[mask], target[mask]

        with amp.autocast(enabled=False):  # amp causes NaNs in this loss function
            g = torch.log(input + self.epsilon) - torch.log(target + self.epsilon)
            Dg = torch.var(g) + self.beta * torch.pow(torch.mean(g), 2)
            loss = 10 * torch.sqrt(Dg)

        debug_nan("SILog Loss", loss)

        if not return_interpolated:
            return loss
        return loss, intr_input


def grad(x):
    diff_x = x[..., 1:, 1:] - x[..., 1:, :-1]
    diff_y = x[..., 1:, 1:] - x[..., :-1, 1:]
    return diff_x**2 + diff_y**2, torch.atan(diff_y / (diff_x + 1e-10))


def grad_mask(mask):
    return mask[..., 1:, 1:] & mask[..., 1:, :-1] & mask[..., :-1, 1:]


class GradL1Loss(nn.Module):
    """Gradient loss"""
    def __init__(self):
        super(GradL1Loss, self).__init__()
        self.name = 'GradL1'

    def forward(self, input, target, mask=None, interpolate=True, return_interpolated=False):
        input = extract_key(input, KEY_OUTPUT)
        if interpolate:
            input = interpolate_if_needed(input, target)
        intr_input = input

        grad_gt = grad(target)
        grad_pred = grad(input)
        if mask is not None:
            mask_g = grad_mask(mask)
        else:
            mask_g = torch.ones_like(grad_gt[0], dtype=torch.bool)

        loss = nn.functional.l1_loss(grad_pred[0][mask_g], grad_gt[0][mask_g]) + \
               nn.functional.l1_loss(grad_pred[1][mask_g], grad_gt[1][mask_g])
        if not return_interpolated:
            return loss
        return loss, intr_input


class OrdinalRegressionLoss(object):
    def __init__(self, ord_num, beta, discretization="SID"):
        self.ord_num = ord_num
        self.beta = beta
        self.discretization = discretization

    def _create_ord_label(self, gt):
        N, _, H, W = gt.shape
        ord_c0 = torch.ones(N, self.ord_num, H, W).to(gt.device)
        if self.discretization == "SID":
            label = self.ord_num * torch.log(gt) / np.log(self.beta)
        else:
            label = self.ord_num * (gt - 1.0) / (self.beta - 1.0)
        label = label.long()

        mask = torch.linspace(0, self.ord_num - 1, self.ord_num, device=gt.device).view(1, self.ord_num, 1, 1).expand(N, -1, H, W).long()
        ord_c0[mask > label] = 0
        ord_c1 = 1 - ord_c0
        ord_label = torch.cat((ord_c0, ord_c1), dim=1)
        return ord_label, mask

    def __call__(self, prob, gt):
        valid_mask = gt > 0.
        ord_label, _ = self._create_ord_label(gt)
        entropy = -prob * ord_label
        loss = torch.sum(entropy, dim=1)[valid_mask.squeeze(1)]
        return loss.mean()


class DiscreteNLLLoss(nn.Module):
    """Cross entropy loss"""
    def __init__(self, min_depth=1e-3, max_depth=10, depth_bins=64):
        super(DiscreteNLLLoss, self).__init__()
        self.name = 'CrossEntropy'
        self.ignore_index = -(depth_bins + 1)
        self._loss_func = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.depth_bins = depth_bins
        self.alpha = 1
        self.zeta = 1 - min_depth
        self.beta = max_depth + self.zeta

    def quantize_depth(self, depth):
        depth = torch.log(depth / self.alpha) / np.log(self.beta / self.alpha)
        depth = torch.round(depth * (self.depth_bins - 1)).long()
        return depth

    def _dequantize_depth(self, depth):
        centers = torch.linspace(self.min_depth, self.max_depth, self.depth_bins, device=depth.device)
        return centers[depth.clamp(0, self.depth_bins - 1)]

    def forward(self, input, target, mask=None, interpolate=True, return_interpolated=False):
        input = extract_key(input, KEY_OUTPUT)
        if interpolate:
            input = interpolate_if_needed(input, target)
        intr_input = input

        if target.ndim == 3:
            target = target.unsqueeze(1)
        target = self.quantize_depth(target)

        if mask is not None:
            mask = mask.long()
            input, target = input * mask + (1 - mask) * self.ignore_index, target * mask + (1 - mask) * self.ignore_index

        input = input.flatten(2)
        target = target.flatten(1)
        loss = self._loss_func(input, target)

        if not return_interpolated:
            return loss
        return loss, intr_input


class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, input, target, mask=None):
        input = extract_key(input, KEY_OUTPUT)
        
        if mask is not None:
            input = input[mask]
            target = target[mask]
        
        loss = torch.sqrt((input - target) ** 2 + self.epsilon ** 2)
        return loss.mean()


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, input, target, mask=None):
        input = extract_key(input, KEY_OUTPUT)
        
        if mask is not None:
            input = input[mask]
            target = target[mask]
        
        loss = torch.abs(input - target)
        return loss.mean()


class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, input, target, mask=None):
        input = extract_key(input, KEY_OUTPUT)
        
        if mask is not None:
            input = input[mask]
            target = target[mask]
        
        loss = (input - target) ** 2
        return loss.mean()
