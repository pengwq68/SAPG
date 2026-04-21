#Training script for the Posterior Cue Predictor (PCP).
#This script trains the cue predictor only and does not train the diffusion backbone.

from __future__ import annotations
import os, argparse, random, time, math
from typing import List, Tuple, Optional, Dict

import numpy as np
import inspect
from PIL import Image

import torch
import torch.nn.functional as F

_SOBEL_CACHE = {}

def _sobel_mag_gray01_cached(x01: torch.Tensor) -> torch.Tensor:
    """Sobel magnitude for [B,1,H,W] gray in [0,1] with cached kernels."""
    assert x01.ndim == 4 and x01.shape[1] == 1
    device = x01.device
    dtype = x01.dtype
    key = (device.type, device.index if device.type == 'cuda' else -1, dtype)
    kx, ky = _SOBEL_CACHE.get(key, (None, None))
    if kx is None:
        kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], device=device, dtype=dtype).view(1, 1, 3, 3)
        ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], device=device, dtype=dtype).view(1, 1, 3, 3)
        _SOBEL_CACHE[key] = (kx, ky)
    gx = F.conv2d(x01, kx, padding=1)
    gy = F.conv2d(x01, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


def edge_focal_loss_from_irvis(ir_a: torch.Tensor,
                               vis_xt: torch.Tensor,
                               roi_full: torch.Tensor | None,
                               edge_focal_pow: float = 1.7,
                               edge_bg_only: bool = False,
                               edge_ground_boost: float = 0.0,
                               edge_ground_y0: float = 0.55,
                               eps: float = 1e-6) -> torch.Tensor:
    assert ir_a.ndim == 4 and vis_xt.ndim == 4
    B, _, H, W = ir_a.shape

    ir01 = ((ir_a + 1.0) * 0.5).clamp(0, 1)
    # IR is already 1ch
    e_ir = _sobel_mag_gray01_cached(ir01)

    vis01 = ((vis_xt.clamp(-1, 1) + 1.0) * 0.5).clamp(0, 1)
    vis_y = (0.299 * vis01[:, 0:1] + 0.587 * vis01[:, 1:2] + 0.114 * vis01[:, 2:3]).clamp(0, 1)
    e_vis = _sobel_mag_gray01_cached(vis_y)

    w = e_ir.clamp(min=0) ** float(max(edge_focal_pow, 0.0))
    if edge_bg_only and (roi_full is not None):
        # background emphasis
        w = w * (1.0 - roi_full.detach().clamp(0, 1))

    if edge_ground_boost and edge_ground_boost > 0:
        y = torch.linspace(0, 1, H, device=vis_xt.device, dtype=vis_xt.dtype).view(1, 1, H, 1)
        ramp = ((y - float(edge_ground_y0)) / max(1e-6, (1.0 - float(edge_ground_y0)))).clamp(0, 1)
        w = w * (1.0 + float(edge_ground_boost) * ramp)

    diff = (e_ir - e_vis).abs()
    return (diff * w).mean() / (w.mean() + eps)

from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from pcp_model import (
    PosteriorCuePredictor,
    roi_l2_losses, edge_l1_loss, orthogonal_loss,
    patch_infonce_cross_hardneg_v17, patch_infonce_intra,
    lowpass_rgb
)

# dataset
IMG_EXT = {".png",".jpg",".jpeg",".bmp",".webp"}

def list_images(root: str) -> List[str]:
    out = []
    for dp, _, fs in os.walk(root):
        for f in fs:
            if os.path.splitext(f)[1].lower() in IMG_EXT:
                out.append(os.path.join(dp, f))
    return sorted(out)

def stem(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]

class PairedIRVISDataset(Dataset):
    def __init__(self, ir_root: str, vis_root: str, image_size: int = 128, max_pairs: Optional[int] = None):
        super().__init__()
        ir_paths = list_images(ir_root)
        vis_paths = list_images(vis_root)
        if len(ir_paths) == 0 or len(vis_paths) == 0:
            raise RuntimeError(f"Empty dataset, IR={len(ir_paths)}, VIS={len(vis_paths)}")

        ir_map, vis_map = {}, {}
        for p in ir_paths:
            ir_map.setdefault(stem(p), []).append(p)
        for p in vis_paths:
            vis_map.setdefault(stem(p), []).append(p)

        keys = sorted(set(ir_map.keys()) & set(vis_map.keys()))
        pairs = [(ir_map[k][0], vis_map[k][0]) for k in keys]
        if len(pairs) == 0:
            raise RuntimeError("No pairs found by filename stem. Ensure IR/VIS share the same stem.")
        if max_pairs is not None:
            pairs = pairs[:int(max_pairs)]
        self.pairs = pairs

        self.t_ir = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.t_vis = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),
        ])

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        ip, vp = self.pairs[idx]
        ir = Image.open(ip).convert("L")
        vis = Image.open(vp).convert("RGB")
        return self.t_ir(ir), self.t_vis(vis), stem(ip)

# diffusion utilities
def get_beta_schedule(beta_schedule: str, *, beta_start: float, beta_end: float, num_diffusion_timesteps: int) -> torch.Tensor:
    if beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "cosine":
        # cosine schedule (Nichol & Dhariwal)
        steps = num_diffusion_timesteps
        s = 0.008
        x = np.linspace(0, steps, steps+1, dtype=np.float64)
        alphas_cumprod = np.cos(((x/steps)+s)/(1+s)*math.pi*0.5)**2
        alphas_cumprod = alphas_cumprod/alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:]/alphas_cumprod[:-1])
        betas = np.clip(betas, 1e-4, 0.999)
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return torch.from_numpy(betas).float()

def extract(a: torch.Tensor, t: torch.Tensor, x_shape) -> torch.Tensor:
    b = t.shape[0]
    out = a.gather(0, t)
    return out.view(b, *([1] * (len(x_shape) - 1)))

def q_sample(x0: torch.Tensor, t: torch.Tensor, sqrt_alphas_cumprod: torch.Tensor, sqrt_one_minus_alphas_cumprod: torch.Tensor) -> torch.Tensor:
    eps = torch.randn_like(x0)
    return extract(sqrt_alphas_cumprod, t, x0.shape) * x0 + extract(sqrt_one_minus_alphas_cumprod, t, x0.shape) * eps

# augmentation (keep consistent with v22)
def jitter_ir_night(ir: torch.Tensor) -> torch.Tensor:
    x = (ir + 1.0) * 0.5
    g = torch.empty(ir.shape[0], 1, 1, 1, device=ir.device).uniform_(0.6, 1.8)
    x = x.clamp(0, 1) ** g
    c = torch.empty(ir.shape[0], 1, 1, 1, device=ir.device).uniform_(0.7, 1.3)
    m = x.mean(dim=(2,3), keepdim=True)
    x = (x - m) * c + m
    x = (x + torch.randn_like(x) * 0.02).clamp(0, 1)
    return x * 2.0 - 1.0

@torch.no_grad()
def _gaussian_kernel_1d(k: int, sigma: float, device, dtype):
    if k % 2 == 0:
        k += 1
    half = k // 2
    x = torch.arange(-half, half + 1, device=device, dtype=dtype)
    g = torch.exp(-(x ** 2) / (2 * (sigma ** 2 + 1e-12)))
    g = g / (g.sum() + 1e-12)
    return g

@torch.no_grad()
def gaussian_blur_1ch_fast(x01: torch.Tensor, k: int = 7, sigma: float = 1.5) -> torch.Tensor:
    """x01: [B,1,H,W] in [0,1]."""
    if k <= 1 or sigma <= 0:
        return x01
    device, dtype = x01.device, x01.dtype
    g = _gaussian_kernel_1d(int(k), float(sigma), device=device, dtype=dtype)
    g1 = g.view(1, 1, -1, 1)
    g2 = g.view(1, 1, 1, -1)
    x = torch.nn.functional.conv2d(x01, g1, padding=(g1.shape[2] // 2, 0), groups=1)
    x = torch.nn.functional.conv2d(x,  g2, padding=(0, g2.shape[3] // 2), groups=1)
    return x

@torch.no_grad()
def ir_aug_nightlike_v2(ir: torch.Tensor,
                        gamma_range=(0.55, 1.90),
                        contrast_range=(0.55, 1.25),
                        noise_std_range=(0.00, 0.035),
                        blur_prob=0.65,
                        blur_k=7,
                        blur_sigma_range=(0.6, 2.2),
                        local_drop_prob=0.70,
                        local_drop_strength=(0.45, 0.85),
                        cutout_prob=0.25,
                        cutout_frac=(0.10, 0.28)) -> torch.Tensor:

    assert ir.ndim == 4 and ir.shape[1] == 1
    x = (ir + 1.0) * 0.5
    x = x.clamp(0, 1)
    B, _, H, W = x.shape
    dev = x.device

    # gamma
    g = torch.empty(B, 1, 1, 1, device=dev).uniform_(*gamma_range)
    x = x ** g

    # contrast around mean
    c = torch.empty(B, 1, 1, 1, device=dev).uniform_(*contrast_range)
    m = x.mean(dim=(2, 3), keepdim=True)
    x = (x - m) * c + m

    # blur
    if blur_prob > 0:
        pb = torch.rand(B, 1, 1, 1, device=dev)
        do_blur = (pb < blur_prob).float()
        sig = torch.empty(B, 1, 1, 1, device=dev).uniform_(*blur_sigma_range)
        if do_blur.sum().item() > 0:
            xb = x.clone()
            for i in range(B):
                if do_blur[i].item() > 0.5:
                    xb[i:i + 1] = gaussian_blur_1ch_fast(x[i:i + 1], k=int(blur_k), sigma=float(sig[i].item()))
            x = xb

    # local attenuation (rear/bottom missing evidence)
    if local_drop_prob > 0:
        p = torch.rand(B, device=dev)
        for i in range(B):
            if p[i].item() < local_drop_prob:
                h0 = int(H * torch.empty(1, device=dev).uniform_(0.45, 0.75).item())
                w0 = int(W * torch.empty(1, device=dev).uniform_(0.45, 0.75).item())
                dh = int(H * torch.empty(1, device=dev).uniform_(0.18, 0.45).item())
                dw = int(W * torch.empty(1, device=dev).uniform_(0.18, 0.45).item())
                h1 = min(H, h0 + dh)
                w1 = min(W, w0 + dw)
                s = torch.empty(1, device=dev).uniform_(*local_drop_strength).item()
                x[i:i + 1, :, h0:h1, w0:w1] = x[i:i + 1, :, h0:h1, w0:w1] * s

    # cutout (hard missing patch)
    if cutout_prob > 0:
        p = torch.rand(B, device=dev)
        for i in range(B):
            if p[i].item() < cutout_prob:
                frac = torch.empty(1, device=dev).uniform_(*cutout_frac).item()
                ch = max(1, int(H * frac))
                cw = max(1, int(W * frac))
                top = int(torch.empty(1, device=dev).uniform_(0, max(1, H - ch)).item())
                left = int(torch.empty(1, device=dev).uniform_(0, max(1, W - cw)).item())
                fill = x[i:i + 1].mean().item()
                x[i:i + 1, :, top:top + ch, left:left + cw] = fill

    # noise
    ns = torch.empty(B, 1, 1, 1, device=dev).uniform_(*noise_std_range)
    x = (x + torch.randn_like(x) * ns).clamp(0, 1)

    return x * 2.0 - 1.0

def apply_ir_aug(ir: torch.Tensor, mode: str, args) -> torch.Tensor:
    mode = str(mode).lower()
    if mode == "none":
        return ir
    if mode == "jitter":
        return jitter_ir_night(ir)
    # default: nightlike_v2
    return ir_aug_nightlike_v2(
        ir,
        gamma_range=(args.ir_gamma_min, args.ir_gamma_max),
        contrast_range=(args.ir_contrast_min, args.ir_contrast_max),
        noise_std_range=(args.ir_noise_min, args.ir_noise_max),
        blur_prob=args.ir_blur_prob,
        blur_k=args.ir_blur_k_aug,
        blur_sigma_range=(args.ir_blur_sigma_min, args.ir_blur_sigma_max),
        local_drop_prob=args.ir_local_drop_prob,
        local_drop_strength=(args.ir_local_drop_min, args.ir_local_drop_max),
        cutout_prob=args.ir_cutout_prob,
        cutout_frac=(args.ir_cutout_frac_min, args.ir_cutout_frac_max),
    )

@torch.no_grad()
def duskify(vis: torch.Tensor,
            gamma_range=(1.2, 1.8),
            contrast_range=(0.65, 0.90),
            wb_r_range=(1.05, 1.20),
            wb_b_range=(0.80, 0.95),
            noise_std_range=(0.00, 0.02)) -> torch.Tensor:
    assert vis.ndim == 4 and vis.shape[1] == 3
    x = (vis + 1.0) * 0.5
    x = x.clamp(0, 1)
    B = x.shape[0]
    device = x.device
    gamma = torch.empty(B, 1, 1, 1, device=device).uniform_(*gamma_range)
    m = x.mean(dim=(2,3), keepdim=True)
    x = (x / (m + 1e-6)).clamp(0, 10) ** gamma
    x = x * m
    c = torch.empty(B, 1, 1, 1, device=device).uniform_(*contrast_range)
    m2 = x.mean(dim=(2,3), keepdim=True)
    x = (x - m2) * c + m2
    wb_r = torch.empty(B, 1, 1, 1, device=device).uniform_(*wb_r_range)
    wb_b = torch.empty(B, 1, 1, 1, device=device).uniform_(*wb_b_range)
    x = torch.cat([x[:,0:1]*wb_r, x[:,1:2], x[:,2:3]*wb_b], dim=1)
    noise_std = torch.empty(B, 1, 1, 1, device=device).uniform_(*noise_std_range)
    x = (x + torch.randn_like(x) * noise_std).clamp(0, 1)
    return x * 2.0 - 1.0

# timestep resampling
class TimestepResampler:
    def __init__(self, T: int, mode: str = "uniform", ema: float = 0.99, eps: float = 1e-8, device: torch.device | str = "cpu"):
        self.T = int(T)
        self.mode = str(mode)
        self.ema = float(ema)
        self.eps = float(eps)
        self.device = torch.device(device)
        self.m2 = torch.ones(self.T, device=self.device)

    @torch.no_grad()
    def sample(self, B: int, t_min: int, t_max: int) -> torch.Tensor:
        if self.mode == "uniform":
            return torch.randint(low=t_min, high=t_max + 1, size=(B,), device=self.device, dtype=torch.long)
        if self.mode == "loss_second_moment":
            w = (self.m2 + self.eps).sqrt()
            # mask outside [t_min, t_max]
            mask = torch.zeros_like(w)
            mask[t_min:t_max+1] = 1.0
            w = w * mask + self.eps
            w = w / w.sum()
            return torch.multinomial(w, num_samples=B, replacement=True).long()
        raise ValueError(f"Unknown t_sampler: {self.mode}")

    @torch.no_grad()
    def update(self, t: torch.Tensor, loss_per_sample: torch.Tensor):
        t = t.detach().view(-1).long()
        l2 = loss_per_sample.detach().view(-1).float().clamp(min=0) ** 2
        # vectorized update with scatter: EMA per index
        # We'll do per-sample loop; B is small (<=64), loop is fine and stable.
        for ti, li2 in zip(t.tolist(), l2.tolist()):
            self.m2[ti] = self.ema * self.m2[ti] + (1.0 - self.ema) * float(li2)

# late-boost schedule
def late_boost_mult(t: torch.Tensor, T: int, start: float, gain: float, k: float) -> torch.Tensor:
    """
    progress = 1 - t/(T-1), in [0,1]; larger means later (closer to x0).
    mult = 1 + gain * sigmoid((progress - start)/k)
    """
    if gain <= 0:
        return torch.ones_like(t, dtype=torch.float32)
    prog = 1.0 - (t.float() / max(T - 1, 1))
    z = (prog - float(start)) / max(float(k), 1e-6)
    return 1.0 + float(gain) * torch.sigmoid(z)

# logging
def _append_line(path: str, s: str):
    with open(path, "a", encoding="utf-8") as f:
        f.write(s + "\n")



def load_state_dict_forgiving(model: nn.Module, state: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    msd = model.state_dict()
    loaded = 0
    skipped = 0
    for k, v in state.items():
        if k in msd and isinstance(v, torch.Tensor) and msd[k].shape == v.shape:
            msd[k].copy_(v)
            loaded += 1
        else:
            skipped += 1
    model.load_state_dict(msd, strict=False)
    return loaded, skipped

# args
def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ir_root", type=str, required=True)
    p.add_argument("--vis_root", type=str, required=True)
    p.add_argument("--outdir", type=str, default="runs/pcp_train")
    p.add_argument("--image_size", type=int, default=128)
    p.add_argument("--max_pairs", type=int, default=None)

    # IR augmentation (bridge day-to-night domain gap)
    p.add_argument("--ir_aug_mode", type=str, default="nightlike_v2",
               choices=["nightlike_v2", "jitter", "none"],
               help="IR augmentation. nightlike_v2 recommended for robust training-from-scratch.")
    p.add_argument("--ir_gamma_min", type=float, default=0.55)
    p.add_argument("--ir_gamma_max", type=float, default=1.90)
    p.add_argument("--ir_contrast_min", type=float, default=0.55)
    p.add_argument("--ir_contrast_max", type=float, default=1.25)
    p.add_argument("--ir_noise_min", type=float, default=0.00)
    p.add_argument("--ir_noise_max", type=float, default=0.035)
    p.add_argument("--ir_blur_prob", type=float, default=0.65)
    p.add_argument("--ir_blur_k_aug", type=int, default=7)
    p.add_argument("--ir_blur_sigma_min", type=float, default=0.6)
    p.add_argument("--ir_blur_sigma_max", type=float, default=2.2)
    p.add_argument("--ir_local_drop_prob", type=float, default=0.70)
    p.add_argument("--ir_local_drop_min", type=float, default=0.45)
    p.add_argument("--ir_local_drop_max", type=float, default=0.85)
    p.add_argument("--ir_cutout_prob", type=float, default=0.25)
    p.add_argument("--ir_cutout_frac_min", type=float, default=0.10)
    p.add_argument("--ir_cutout_frac_max", type=float, default=0.28)

    # ROI robustness (forwarded to the PCP model)
    p.add_argument("--roi_q_hot_base", type=float, default=0.85)
    p.add_argument("--roi_q_edge_base", type=float, default=0.85)
    p.add_argument("--roi_alpha", type=float, default=0.6)
    p.add_argument("--roi_sharpness", type=float, default=10.0)
    p.add_argument("--roi_adapt", action="store_true", help="enable adaptive ROI quantiles based on IR contrast.")
    p.add_argument("--roi_q_min", type=float, default=0.75)
    p.add_argument("--roi_q_max", type=float, default=0.92)
    p.add_argument("--roi_contrast_ref", type=float, default=0.18)
    p.add_argument("--roi_adapt_strength", type=float, default=0.35)
    p.add_argument("--roi_dilate_k", type=int, default=9)
    p.add_argument("--roi_dilate_iters", type=int, default=1)

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    # auto two-phase schedule (single-run overnight)
    p.add_argument("--auto_schedule", action="store_true", help="enable automatic 2-phase training schedule (no manual intervention).")
    p.add_argument("--phase1_epochs", type=int, default=0, help="number of initial epochs for phase-1 (fast, gdc_on_dusk disabled). 0 disables auto schedule unless --auto_schedule is set and value>0.")
    p.add_argument("--lr_phase2", type=float, default=0.0, help="phase-2 learning rate. 0 means use lr*0.4.")
    p.add_argument("--w_day_phase2", type=float, default=-1.0, help="phase-2 w_day. -1 keeps w_day.")
    p.add_argument("--w_gdc_phase2", type=float, default=-1.0, help="phase-2 w_gdc. -1 keeps w_gdc.")
    p.add_argument("--w_gate_sup_phase2", type=float, default=-1.0, help="phase-2 w_gate_sup. -1 keeps w_gate_sup.")
    p.add_argument("--late_boost_gain_phase2", type=float, default=-1.0, help="phase-2 late_boost_gain. -1 keeps late_boost_gain.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--save_every", type=int, default=20)

    # diffusion time
    p.add_argument("--use_time", action="store_true")
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--t_min", type=int, default=0)
    p.add_argument("--t_max", type=int, default=999)
    p.add_argument("--beta_schedule", type=str, default="linear", choices=["linear","cosine"])
    p.add_argument("--beta_start", type=float, default=1e-4)
    p.add_argument("--beta_end", type=float, default=2e-2)

    # ReSample-style t sampling
    p.add_argument("--t_sampler", type=str, default="loss_second_moment", choices=["uniform","loss_second_moment"])
    p.add_argument("--t_ema", type=float, default=0.99)

    # model
    p.add_argument("--feat_dim", type=int, default=128)
    p.add_argument("--base", type=int, default=64)
    p.add_argument("--norm", type=str, default="bn", choices=["bn","gn","none"])
    p.add_argument("--ir_blur_k", type=int, default=5)
    p.add_argument("--ir_blur_sigma", type=float, default=1.0)

    # contrastive
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--k_roi", type=int, default=64)
    p.add_argument("--k_bg", type=int, default=64)
    p.add_argument("--bg_pool", type=int, default=512)

    # weights
    p.add_argument("--w_roi", type=float, default=1.0)
    p.add_argument("--w_bg", type=float, default=0.1)
    p.add_argument("--w_edge", type=float, default=0.05)
    # edge-focal (lane/ground preservation)
    p.add_argument("--edge_focal_pow", type=float, default=1.7,
                   help="Edge-Focal power on IR edge magnitude (>1 focuses thin lines).")
    p.add_argument("--edge_bg_only", action="store_true",
                   help="Apply edge-focal only on background (1-ROI).")
    p.add_argument("--edge_ground_boost", type=float, default=0.0,
                   help="Boost edge-focal weight on ground region (bottom part). 0 disables.")
    p.add_argument("--edge_ground_y0", type=float, default=0.55,
                   help="Ground boost starts from y>y0 (0~1).")
    p.add_argument("--w_contrast_cross", type=float, default=0.25)
    p.add_argument("--w_contrast_intra", type=float, default=0.25)
    p.add_argument("--w_ortho", type=float, default=0.05)

    # dayness
    p.add_argument("--w_day", type=float, default=0.15)
    p.add_argument("--day_head_hidden", type=int, default=256)
    p.add_argument("--day_lp_k", type=int, default=11)
    p.add_argument("--day_lp_sigma", type=float, default=2.0)

    # gated CDC
    p.add_argument("--w_gdc", type=float, default=0.08)
    p.add_argument("--gdc_mode", type=str, default="hinge_missing", choices=["hinge_missing","sym"])
    p.add_argument("--gdc_eps", type=float, default=1e-3)
    p.add_argument("--gdc_on_dusk", action="store_true")
    p.add_argument("--gdc_on_dusk_every", type=int, default=3, help="run dusk branch every N iters (>=1).")
    p.add_argument("--w_gate_prior", type=float, default=0.10)
    p.add_argument("--w_gate_tv", type=float, default=0.02)
    p.add_argument("--gate_prior_pow", type=float, default=0.75)

    # gate supervision (feat-missing)
    p.add_argument("--w_gfeat", type=float, default=0.02)
    p.add_argument("--sim_target", type=float, default=0.65)
    p.add_argument("--w_gate_sup", type=float, default=0.08)
    p.add_argument("--gate_sup_alpha", type=float, default=0.7)

    # late-boost (align with final steps)
    p.add_argument("--late_boost_start", type=float, default=0.60)
    p.add_argument("--late_boost_gain", type=float, default=2.0)
    p.add_argument("--late_boost_k", type=float, default=0.08)

    # init/resume
    p.add_argument("--init_ckpt", type=str, default="", help="initialize from a pretrained PCP checkpoint (model weights).")
    p.add_argument("--resume", type=str, default="", help="resume training from a saved ckpt (model+opt).")
    p.add_argument("--strict_init", action="store_true")
    p.add_argument("--amp", action="store_true")

    return p.parse_args()

def main():
    args = build_args()
    os.makedirs(args.outdir, exist_ok=True)
    log_path = os.path.join(args.outdir, "train.log")

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = PairedIRVISDataset(args.ir_root, args.vis_root, image_size=args.image_size, max_pairs=args.max_pairs)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                    pin_memory=True, drop_last=True)

    model = PosteriorCuePredictor(
        feat_dim=args.feat_dim, base=args.base, norm=args.norm,
        use_time=args.use_time, ir_blur_k=args.ir_blur_k_aug, ir_blur_sigma=args.ir_blur_sigma,
        day_head_hidden=args.day_head_hidden, day_use_time=True,
        roi_q_hot_base=args.roi_q_hot_base,
        roi_q_edge_base=args.roi_q_edge_base,
        roi_alpha=args.roi_alpha,
        roi_sharpness=args.roi_sharpness,
        roi_adapt=True if args.roi_adapt else True,
        roi_q_min=args.roi_q_min,
        roi_q_max=args.roi_q_max,
        roi_contrast_ref=args.roi_contrast_ref,
        roi_adapt_strength=args.roi_adapt_strength,
        roi_dilate_k=args.roi_dilate_k,
        roi_dilate_iters=args.roi_dilate_iters
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # init/resume
    start_epoch = 1
    best = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        state = ckpt.get("model", ckpt.get("state_dict", ckpt))
        model.load_state_dict(state, strict=False)
        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best = float(ckpt.get("mean_loss", best))
        _append_line(log_path, f"[resume] {args.resume} start_epoch={start_epoch} best={best:.6f}")
    elif args.init_ckpt:
        if os.path.isfile(args.init_ckpt):
            ckpt = torch.load(args.init_ckpt, map_location="cpu")
            state = ckpt.get("model", ckpt.get("state_dict", ckpt))
            strict = bool(getattr(args, "strict_init", False))
            if strict:
                msg = model.load_state_dict(state, strict=True)
                _append_line(log_path, f"[init_ckpt] {args.init_ckpt} strict=True missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
            else:
                loaded, skipped = load_state_dict_forgiving(model, state)
                _append_line(log_path, f"[init_ckpt] {args.init_ckpt} strict=False loaded={loaded} skipped={skipped}")
        else:
            _append_line(log_path, f"[init_ckpt] WARNING not found: {args.init_ckpt}")

    print(f"[PCP-train] model_py={__import__('pcp_model').__file__}")
    print(f"[PCP-train] forward_sig={inspect.signature(model.forward)}")
    _append_line(log_path, f"[start] outdir={args.outdir} device={device} pairs={len(ds)} batch={args.batch_size}")

    # diffusion q(x_t|x0)
    betas = get_beta_schedule(args.beta_schedule, beta_start=args.beta_start, beta_end=args.beta_end, num_diffusion_timesteps=args.T).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    scaler = torch.cuda.amp.GradScaler() if (args.amp and device.type == "cuda") else None
    bce = torch.nn.BCEWithLogitsLoss()

    t_resampler = TimestepResampler(args.T, mode=args.t_sampler, ema=args.t_ema, device=device)

    def log(s: str):
        print(s, flush=True)
        _append_line(log_path, s)

    def save_ckpt(name: str, epoch: int, mean_loss: float):
        path = os.path.join(args.outdir, name)
        torch.save({"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(), "mean_loss": mean_loss, "args": vars(args)}, path)
        log(f"[save] {path} (epoch={epoch}, loss={mean_loss:.6f})")

    global_step = 0

    #auto schedule state
    cur_lr = float(args.lr)

    def _set_lr(new_lr: float):
        nonlocal cur_lr
        if abs(float(new_lr) - float(cur_lr)) < 1e-12:
            return
        for pg in opt.param_groups:
            pg["lr"] = float(new_lr)
        cur_lr = float(new_lr)
        _append_line(log_path, f"[lr] set lr={cur_lr:g}")


    for epoch in range(start_epoch, args.epochs + 1):
        model.train()

        #auto 2-phase schedule (hands-off overnight)
        if args.auto_schedule and int(args.phase1_epochs) > 0:
            phase2 = (epoch > int(args.phase1_epochs))
        else:
            phase2 = False

        # lr schedule
        lr2 = float(args.lr_phase2) if float(args.lr_phase2) > 0 else float(args.lr) * 0.4
        _set_lr(lr2 if phase2 else float(args.lr))

        # effective weights (phase2 overrides if provided)
        w_day_eff = float(args.w_day_phase2) if (phase2 and float(args.w_day_phase2) >= 0) else float(args.w_day)
        w_gdc_eff = float(args.w_gdc_phase2) if (phase2 and float(args.w_gdc_phase2) >= 0) else float(args.w_gdc)
        w_gate_sup_eff = float(args.w_gate_sup_phase2) if (phase2 and float(args.w_gate_sup_phase2) >= 0) else float(args.w_gate_sup)
        late_gain_eff = float(args.late_boost_gain_phase2) if (phase2 and float(args.late_boost_gain_phase2) >= 0) else float(args.late_boost_gain)

        # gdc_on_dusk enabled only in phase2 (unless auto schedule disabled)
        # gdc_on_dusk: only active in phase2 when auto_schedule is enabled
        gdc_on_dusk_active = bool(args.gdc_on_dusk) and ( (not args.auto_schedule) or phase2 )
        gdc_on_dusk_every = max(1, int(args.gdc_on_dusk_every))

        if epoch == start_epoch or (args.auto_schedule and int(args.phase1_epochs) > 0 and (epoch == int(args.phase1_epochs) + 1)):
            _append_line(log_path, f"[phase] epoch={epoch} phase2={int(phase2)} lr={cur_lr:g} "
                                   f"w_day={w_day_eff:g} w_gdc={w_gdc_eff:g} w_gate_sup={w_gate_sup_eff:g} late_gain={late_gain_eff:g} dusk={int(gdc_on_dusk_active)}")

        losses_epoch = []
        t0 = time.time()

        for it, (ir, vis, _) in enumerate(dl, start=1):
            global_step += 1
            ir = ir.to(device, non_blocking=True)
            vis = vis.to(device, non_blocking=True)

            # timestep sample (ReSample)
            t = t_resampler.sample(ir.shape[0], args.t_min, args.t_max)
            t_in = t if args.use_time else None

            # augmentations / q-sample
            ir_a = apply_ir_aug(ir, args.ir_aug_mode, args)
            ir_b0 = apply_ir_aug(ir, args.ir_aug_mode, args)
            ir_b = q_sample(ir_b0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)

            vis_xt = q_sample(vis, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
            # dusk branch may be run sparsely
            use_dusk_this = bool(gdc_on_dusk_active and (gdc_on_dusk_every >= 1) and ((global_step % gdc_on_dusk_every) == 0)) and ((w_day_eff > 0.0) or (w_gdc_eff > 0.0))
            if use_dusk_this:
                vis_dusk_xt = q_sample(duskify(vis), t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
            else:
                vis_dusk_xt = None

            # lowpass for dayness supervision
            vis_lp = lowpass_rgb(vis_xt, k=args.day_lp_k, sigma=args.day_lp_sigma)
            dusk_lp = lowpass_rgb(vis_dusk_xt, k=args.day_lp_k, sigma=args.day_lp_sigma) if vis_dusk_xt is not None else None

            opt.zero_grad(set_to_none=True)
            use_amp = (scaler is not None)

            with (torch.cuda.amp.autocast(dtype=torch.float16) if use_amp else torch.enable_grad()):
                out = model(ir_a, vis_xt, t=t_in)

                #  struct losses
                l_roi8, l_bg8 = roi_l2_losses(out["s8_ir"], out["s8_vis"], out["roi8"])
                l_roi16, l_bg16 = roi_l2_losses(out["s16_ir"], out["s16_vis"], out["roi16"])
                l_align = args.w_roi * (l_roi8 + l_roi16) + args.w_bg * (l_bg8 + l_bg16)

                l_edge = edge_focal_loss_from_irvis(
                    ir_a, vis_xt,
                    roi_full=out.get('roi_full', None),
                    edge_focal_pow=args.edge_focal_pow,
                    edge_bg_only=args.edge_bg_only,
                    edge_ground_boost=args.edge_ground_boost,
                    edge_ground_y0=args.edge_ground_y0,
                ) * args.w_edge

                # contrastive
                if (args.w_contrast_cross > 0) or (args.w_contrast_intra > 0):
                    ir01 = ((ir_a + 1.0) * 0.5).clamp(0, 1)
                    ir01_8 = F.interpolate(ir01, size=out["roi8"].shape[-2:], mode="bilinear", align_corners=False)
                    ir01_16 = F.interpolate(ir01, size=out["roi16"].shape[-2:], mode="bilinear", align_corners=False)

                if args.w_contrast_cross > 0:
                    l_cross = (
                                        patch_infonce_cross_hardneg_v17(out["s8_ir"], out["s8_vis"], out["roi8"], ir01_down=ir01_8,
                                                                        temperature=args.temperature, k_roi=args.k_roi, k_bg=args.k_bg, bg_pool=args.bg_pool)
                                        + patch_infonce_cross_hardneg_v17(out["s16_ir"], out["s16_vis"], out["roi16"], ir01_down=ir01_16,
                                                                          temperature=args.temperature, k_roi=args.k_roi, k_bg=args.k_bg, bg_pool=args.bg_pool)
                    ) * args.w_contrast_cross
                else:
                    l_cross = vis_xt.new_tensor(0.0)

                if args.w_contrast_intra > 0:
                    ir_b_enc = model.encode_ir(ir_b, t=t_in)
                    l_intra = (
                                        patch_infonce_intra(out["s8_ir"], ir_b_enc["s8"], out["roi8"], temperature=args.temperature, k_roi=args.k_roi)
                                        + patch_infonce_intra(out["s16_ir"], ir_b_enc["s16"], out["roi16"], temperature=args.temperature, k_roi=args.k_roi)
                    ) * args.w_contrast_intra
                else:
                    l_intra = vis_xt.new_tensor(0.0)

                # ortho between struct and color features
                l_ortho = (orthogonal_loss(out["s8_vis"], out["c8_vis"]) + orthogonal_loss(out["s16_vis"], out["c16_vis"])) * args.w_ortho

                #dayness (late-boosted)
                # day=1 on vis_lp, dusk=0 on dusk_lp
                if w_day_eff > 0.0:
                    day_logits = model.encode_vis(vis_lp, t=t_in)['day_logit']
                    l_day = bce(day_logits, torch.ones_like(day_logits))
                    if dusk_lp is not None:
                        dusk_logits = model.encode_vis(dusk_lp, t=t_in)['day_logit']
                        l_day = 0.5 * (l_day + bce(dusk_logits, torch.zeros_like(dusk_logits)))
                    # late boost (focus on late stage / small t)
                    m_late = late_boost_mult(t, args.T, args.late_boost_start, late_gain_eff, args.late_boost_k)
                    l_day = (l_day * m_late.mean()).clamp(min=0) * w_day_eff
                else:
                    l_day = vis_xt.new_tensor(0.0)

                #gated CDC + gate losses (late-boosted)
                gate_active = ((w_gdc_eff > 0.0) or (float(args.w_gate_prior) > 0.0) or (float(args.w_gate_tv) > 0.0) or (w_gate_sup_eff > 0.0) or (float(args.w_gfeat) > 0.0))
                if gate_active:
                    # build edges
                    vis01 = ((vis_xt + 1.0) * 0.5).clamp(0, 1)
                    vis_y = (0.299 * vis01[:,0:1] + 0.587 * vis01[:,1:2] + 0.114 * vis01[:,2:3]).clamp(0, 1)
                    # simple sobel magnitude from v22 training
                    def sobel_mag_gray01(x01: torch.Tensor) -> torch.Tensor:
                        # wrapper: use cached kernels to reduce CUDA fragmentation
                        return _sobel_mag_gray01_cached(x01)
                    
                    e_vis = sobel_mag_gray01(vis_y)
                    e_ir = sobel_mag_gray01(ir01)
                    
                    if args.gdc_mode == "sym":
                        r = (e_vis - e_ir).abs()
                    else:
                        r = torch.relu(e_ir - e_vis)
                    
                    # charbonnier
                    r = torch.sqrt(r * r + args.gdc_eps * args.gdc_eps)
                    
                    gate = out["g_gate"]  # [B,1,H,W]
                    l_gdc = (r * gate).mean()
                    
                    if (w_gdc_eff > 0.0) and (vis_dusk_xt is not None):
                        gate_d = gate  # reuse current gate to avoid second forward
                        vis01_d = ((vis_dusk_xt + 1.0) * 0.5).clamp(0, 1)
                        vis_y_d = (0.299 * vis01_d[:,0:1] + 0.587 * vis01_d[:,1:2] + 0.114 * vis01_d[:,2:3]).clamp(0, 1)
                        e_vis_d = sobel_mag_gray01(vis_y_d)
                        if args.gdc_mode == "sym":
                            r_d = (e_vis_d - e_ir).abs()
                        else:
                            r_d = torch.relu(e_ir - e_vis_d)
                        r_d = torch.sqrt(r_d * r_d + args.gdc_eps * args.gdc_eps)
                        l_gdc = l_gdc + (r_d * gate_d).mean() * 0.5
                    
                    l_gdc = (l_gdc * m_late.mean()).clamp(min=0) * w_gdc_eff
                    
                    # gate prior (prevent collapse)
                    e_ir_n = e_ir / (e_ir.flatten(1).mean(dim=1).view(-1,1,1,1) + 1e-6)
                    prior = (e_ir_n.clamp(0, 1) ** args.gate_prior_pow) * out["roi_full"]
                    l_gate_prior = (gate - prior).abs().mean()
                    l_gate_prior = (l_gate_prior * m_late.mean()).clamp(min=0) * args.w_gate_prior
                    
                    # gate sup from sim16 feature-missing
                    sim16_01 = ((out["sim16"] + 1.0) * 0.5).clamp(0, 1)
                    feat_missing16 = torch.relu(float(args.sim_target) - sim16_01)  # [B,1,h16,w16]
                    feat_missing_full = F.interpolate(feat_missing16, size=gate.shape[-2:], mode="bilinear", align_corners=False)
                    hard_map = (prior + float(args.gate_sup_alpha) * feat_missing_full).clamp(0, 1).detach()
                    l_gate_sup = F.smooth_l1_loss(gate, hard_map)
                    l_gate_sup = (l_gate_sup * m_late.mean()).clamp(min=0) * w_gate_sup_eff
                    
                    # optional: feature-missing measurement loss weighted by gate16
                    if args.w_gfeat > 0:
                        l_gfeat = torch.sqrt(feat_missing16 * feat_missing16 + args.gdc_eps * args.gdc_eps)
                        gate16 = F.interpolate(gate, size=feat_missing16.shape[-2:], mode="bilinear", align_corners=False)
                        l_gfeat = (l_gfeat * gate16).mean()
                        l_gfeat = (l_gfeat * m_late.mean()).clamp(min=0) * float(args.w_gfeat)
                    else:
                        l_gfeat = gate.new_tensor(0.0)
                    
                    # TV smooth gate
                    tv = (gate[:, :, :, 1:] - gate[:, :, :, :-1]).abs().mean() + (gate[:, :, 1:, :] - gate[:, :, :-1, :]).abs().mean()
                    l_gate_tv = (tv * m_late.mean()).clamp(min=0) * args.w_gate_tv
                    
                else:
                    l_gdc = vis_xt.new_tensor(0.0)
                    l_gate_prior = vis_xt.new_tensor(0.0)
                    l_gate_tv = vis_xt.new_tensor(0.0)
                    l_gate_sup = vis_xt.new_tensor(0.0)
                    l_gfeat = vis_xt.new_tensor(0.0)

                loss = l_align + l_edge + l_cross + l_intra + l_ortho + l_day + l_gdc + l_gate_prior + l_gate_tv + l_gate_sup + l_gfeat

            # backward
            if scaler is None:
                loss.backward()
                opt.step()
            else:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

            losses_epoch.append(float(loss.detach().item()))

            # update resampler with per-sample proxy
            # Use a cheap proxy: mean(|r*gate|) per sample (CDC hardness) + l_day per sample.
            with torch.no_grad():
                per = (r * gate).mean(dim=(1,2,3)) + (F.softplus(day_logits).view(-1) * 0.05)
                t_resampler.update(t, per)

            if (it % 50 == 0) or (it == 1):
                log(
                    f"[ep {epoch:03d} it {it:04d}] "
                    f"loss={losses_epoch[-1]:.4f} "
                    f"align={float(l_align.detach().item()):.4f} edge={float(l_edge.detach().item()):.4f} "
                    f"cross={float(l_cross.detach().item()):.4f} intra={float(l_intra.detach().item()):.4f} "
                    f"ortho={float(l_ortho.detach().item()):.4f} day={float(l_day.detach().item()):.4f} "
                    f"gdc={float(l_gdc.detach().item()):.4f} gprior={float(l_gate_prior.detach().item()):.4f} gtv={float(l_gate_tv.detach().item()):.4f} "
                    f"gsup={float(l_gate_sup.detach().item()):.4f} gfeat={float(l_gfeat.detach().item()):.4f} "
                    f"dusk={int(use_dusk_this)} t_sampler={args.t_sampler}"
                )

        mean_loss = float(np.mean(losses_epoch))
        dt = time.time() - t0
        log(f"[epoch {epoch:03d}] mean_loss={mean_loss:.6f} time={dt:.1f}s")

        if mean_loss < best:
            best = mean_loss
            save_ckpt("pcp_best.pt", epoch, mean_loss)

        if (epoch % args.save_every) == 0:
            save_ckpt(f"pcp_ep{epoch:03d}.pt", epoch, mean_loss)

    save_ckpt("pcp_last.pt", args.epochs, best)
    log("Done.")

if __name__ == "__main__":
    main()
