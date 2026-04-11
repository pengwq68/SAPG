
import os
import math
import glob
import time
import logging
import argparse
import inspect
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from PIL import Image
from torchvision import transforms

from guided_diffusion.script_util import create_model



def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    if beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "cosine":
        s = 0.008
        steps = num_diffusion_timesteps + 1
        x = np.linspace(0, steps, steps, dtype=np.float64)
        alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = np.clip(betas, 1e-4, 0.999)
    else:
        raise NotImplementedError(beta_schedule)

    assert betas.shape == (num_diffusion_timesteps,)
    return torch.from_numpy(betas).float()


def extract(a, t, x_shape):
    """Extract values from 1D tensor a at indices t and reshape to [B,1,1,1]."""
    b = t.shape[0]
    out = a.gather(0, t)
    return out.view(b, *([1] * (len(x_shape) - 1)))



def create_logger(save_root: str):
    os.makedirs(save_root, exist_ok=True)
    log_file = os.path.join(save_root, "train_ir_cond_vis_diffusion.log")

    logger = logging.getLogger(f"train_ir_cond_vis_diffusion_{save_root}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # prevent duplicate handlers in Jupyter
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)

    logger.info("Log file: %s", log_file)
    return logger



def _strip_module_prefix(sd):
    return {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in sd.items()}


def _add_module_prefix(sd):
    return {("module." + k) if not k.startswith("module.") else k: v for k, v in sd.items()}


def load_state_dict_flexible(model, state_dict):
    """Load state dict even if 'module.' prefix mismatches."""
    msd = model.state_dict()
    has_module_in_model = any(k.startswith("module.") for k in msd.keys())
    has_module_in_ckpt = any(k.startswith("module.") for k in state_dict.keys())

    sd = state_dict
    if has_module_in_model and not has_module_in_ckpt:
        sd = _add_module_prefix(sd)
    elif (not has_module_in_model) and has_module_in_ckpt:
        sd = _strip_module_prefix(sd)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    return missing, unexpected


def save_checkpoint(path, step, model, ema_model, optimizer, betas_cpu, args_dict):
    ckpt = {
        "step": step,
        "model": model.state_dict(),
        "ema_model": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "betas": betas_cpu,
        "args": args_dict,
    }
    torch.save(ckpt, path)


def call_with_supported_kwargs(fn, **kwargs):
    sig = inspect.signature(fn)
    kk = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(**kk)


def expand_unet_in_channels_(model: torch.nn.Module, new_in_channels: int, logger=None):
    """
    If create_model doesn't take in_channels, we expand the first conv weight.
    - Find the earliest Conv2d expecting 3 in_channels and expand to new_in_channels
    - Copy RGB weights into first 3 channels; init extra channels with mean(RGB)
    """
    conv = None
    conv_name = None
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d) and m.in_channels == 3:
            conv = m
            conv_name = name
            break

    if conv is None:
        raise RuntimeError("Cannot find a Conv2d with in_channels=3 to expand. Your UNet might be different.")

    if conv.in_channels == new_in_channels:
        return

    w = conv.weight.data  # [out, in=3, k, k]
    out_ch, in_ch, kh, kw = w.shape
    if in_ch != 3:
        raise RuntimeError(f"Expected first conv in_channels=3, got {in_ch}")

    new_w = torch.zeros((out_ch, new_in_channels, kh, kw), device=w.device, dtype=w.dtype)
    new_w[:, :3] = w
    if new_in_channels > 3:
        mean_rgb = w.mean(dim=1, keepdim=True)  # [out,1,k,k]
        for c in range(3, new_in_channels):
            new_w[:, c:c+1] = mean_rgb

    conv.in_channels = new_in_channels
    conv.weight = torch.nn.Parameter(new_w)
    if logger:
        logger.info("Expanded first conv '%s' in_channels 3 -> %d", conv_name, new_in_channels)

def disable_all_checkpointing_(model, logger=None):
    n = 0
    for m in model.modules():
        if hasattr(m, "use_checkpoint"):
            try:
                if getattr(m, "use_checkpoint") is True:
                    setattr(m, "use_checkpoint", False)
                n += 1
            except Exception:
                pass
    if logger:
        logger.info("Disabled checkpointing in %d modules (those with attribute use_checkpoint).", n)


def list_images(root):
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(root, "**", e), recursive=True))
    return sorted(files)


def stem(p):
    return os.path.splitext(os.path.basename(p))[0]


class KAISTPairedIRVISDataset(data.Dataset):
    def __init__(self, ir_root, vis_root, image_size=128, max_pairs=None, pair_mode="stem", hflip=True):
        super().__init__()
        self.ir_root = ir_root
        self.vis_root = vis_root
        self.image_size = image_size
        self.pair_mode = pair_mode

        ir_paths = list_images(ir_root)
        vis_paths = list_images(vis_root)
        if len(ir_paths) == 0 or len(vis_paths) == 0:
            raise RuntimeError(f"Empty set. IR={len(ir_paths)} under {ir_root}, VIS={len(vis_paths)} under {vis_root}")

        if pair_mode != "stem":
            raise NotImplementedError("Only pair_mode=stem is implemented in this script.")

        ir_map = {}
        for p in ir_paths:
            ir_map.setdefault(stem(p), []).append(p)
        vis_map = {}
        for p in vis_paths:
            vis_map.setdefault(stem(p), []).append(p)

        keys = sorted(set(ir_map.keys()) & set(vis_map.keys()))
        pairs = []
        for k in keys:
            # if duplicates, just take the first (but keep deterministic)
            pairs.append((ir_map[k][0], vis_map[k][0]))

        if len(pairs) == 0:
            raise RuntimeError(
                "No paired files found by stem matching.\n"
                f"IR root: {ir_root}\nVIS root: {vis_root}\n"
                "Try to make sure paired IR/VIS share the same filename stem."
            )

        if max_pairs is not None:
            pairs = pairs[:int(max_pairs)]

        self.pairs = pairs

        # transforms (keep consistent)
        # IR: grayscale -> tensor [1,H,W] -> normalize to [-1,1]
        self.ir_tf = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        # VIS: RGB -> tensor [3,H,W] -> normalize to [-1,1]
        self.vis_tf = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        self.hflip = hflip

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ir_path, vis_path = self.pairs[idx]

        ir = Image.open(ir_path).convert("L")
        vis = Image.open(vis_path).convert("RGB")

        # paired augmentation: same random flip
        if self.hflip and (torch.rand(1).item() < 0.5):
            ir = ir.transpose(Image.FLIP_LEFT_RIGHT)
            vis = vis.transpose(Image.FLIP_LEFT_RIGHT)

        ir_t = self.ir_tf(ir)    # [1,H,W] in [-1,1]
        vis_t = self.vis_tf(vis) # [3,H,W] in [-1,1]
        return ir_t, vis_t, os.path.basename(vis_path)


def build_args():
    p = argparse.ArgumentParser()

    # paired data
    p.add_argument("--ir_root", type=str, required=True, help="IR images root (paired with VIS by filename stem)")
    p.add_argument("--vis_root", type=str, required=True, help="VIS images root (paired with IR by filename stem)")
    p.add_argument("--image_size", type=int, default=128)
    p.add_argument("--max_pairs", type=int, default=None)

    # model (same as your current)
    p.add_argument("--model_channels", type=int, default=96)
    p.add_argument("--num_res_blocks", type=int, default=2)
    p.add_argument("--channel_mult", type=str, default="1,2,2,4")
    p.add_argument("--attn_res", type=str, default="32,16,8")
    p.add_argument("--data_parallel", action="store_true")

    # memory
    p.add_argument("--no_checkpoint", action="store_true",
                   help="Disable gradient checkpointing (default: checkpointing enabled if supported by model)")

    # diffusion
    p.add_argument("--num_timesteps", type=int, default=1000)
    p.add_argument("--beta_schedule", type=str, default="linear")
    p.add_argument("--beta_start", type=float, default=1e-4)
    p.add_argument("--beta_end", type=float, default=0.02)

    # train
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--total_steps", type=int, default=200000)
    p.add_argument("--log_interval", type=int, default=100)
    p.add_argument("--save_interval", type=int, default=1000)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=1234)

    # conditional tricks
    p.add_argument("--cond_drop_prob", type=float, default=0.1,
                   help="Classifier-free training: drop IR condition to zeros with this prob (0=off)")

    # fp16 / perf
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--tf32", action="store_true")

    # resume / save
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--save_root", type=str, default="runs/ir_cond_vis_diffusion")

    return p.parse_args()



def main():
    args = build_args()

    # perf flags
    if args.tf32:
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # save dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_root, timestamp)
    logger = create_logger(save_dir)
    logger.info("Args: %s", args)
    logger.info("Use device: %s, CUDA available: %s, GPU count: %d",
                device, torch.cuda.is_available(), torch.cuda.device_count())

    # data
    dataset = KAISTPairedIRVISDataset(
        ir_root=args.ir_root,
        vis_root=args.vis_root,
        image_size=args.image_size,
        max_pairs=args.max_pairs,
        pair_mode="stem",
        hflip=True
    )
    logger.info("Found %d paired samples", len(dataset))

    pin_memory = (device.type == "cuda")
    dl_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    if args.workers > 0:
        dl_kwargs["persistent_workers"] = True
        dl_kwargs["prefetch_factor"] = 4
    loader = data.DataLoader(dataset, **dl_kwargs)

    # diffusion params
    betas = get_beta_schedule(
        args.beta_schedule,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        num_diffusion_timesteps=args.num_timesteps,
    ).to(device)

    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    model = call_with_supported_kwargs(
        create_model,
        image_size=args.image_size,
        num_class=1000,
        num_channels=args.model_channels,
        model_channels=args.model_channels,  # some repos use this name
        num_res_blocks=args.num_res_blocks,
        learn_sigma=False,
        class_cond=False,
        attention_resolutions=args.attn_res,
        attn_res=args.attn_res,
        num_heads=4,
        num_head_channels=64,
        num_heads_upsample=4,
        use_scale_shift_norm=True,
        dropout=0.0,
        channel_mult=args.channel_mult,
        resblock_updown=True,
        use_checkpoint=(not args.no_checkpoint),
        use_fp16=False,
        use_new_attention_order=False,
        in_channels=4,
        out_channels=3,
    ).to(device)


    try:
        expand_unet_in_channels_(model, 4, logger=logger)
    except Exception as e:
        logger.info("First-conv expansion skipped/failed (may be OK if model already 4ch): %s", str(e))

    if args.no_checkpoint:
        disable_all_checkpointing_(model, logger=logger)
    model.train()

    # optional DataParallel
    if args.data_parallel and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        logger.info("Enabled DataParallel with %d GPUs.", torch.cuda.device_count())

    # EMA model
    ema_model = call_with_supported_kwargs(
        create_model,
        image_size=args.image_size,
        num_class=1000,
        num_channels=args.model_channels,
        model_channels=args.model_channels,
        num_res_blocks=args.num_res_blocks,
        learn_sigma=False,
        class_cond=False,
        attention_resolutions=args.attn_res,
        attn_res=args.attn_res,
        num_heads=4,
        num_head_channels=64,
        num_heads_upsample=4,
        use_scale_shift_norm=True,
        dropout=0.0,
        channel_mult=args.channel_mult,
        resblock_updown=True,
        use_checkpoint=(not args.no_checkpoint),
        use_fp16=False,
        use_new_attention_order=False,
        in_channels=4,
        out_channels=3,
    ).to(device)

    try:
        expand_unet_in_channels_(ema_model, 4, logger=logger)
    except Exception:
        pass

    if args.no_checkpoint:
        disable_all_checkpointing_(ema_model, logger=logger)
    if args.data_parallel and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        ema_model = torch.nn.DataParallel(ema_model)

    # init EMA weights
    miss, unexp = load_state_dict_flexible(ema_model, model.state_dict())
    if miss or unexp:
        logger.info("EMA init non-strict load (ok). missing=%d unexpected=%d", len(miss), len(unexp))

    ema_decay = 0.9999

    @torch.no_grad()
    def ema_update():
        for p_ema, p in zip(ema_model.parameters(), model.parameters()):
            p_ema.data.mul_(ema_decay).add_(p.data, alpha=1.0 - ema_decay)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # AMP
    try:
        from torch.amp import autocast, GradScaler
        scaler = GradScaler("cuda", enabled=(device.type == "cuda" and args.fp16))
        use_new_amp = True
    except Exception:
        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler(enabled=(device.type == "cuda" and args.fp16))
        use_new_amp = False

    def autocast_ctx():
        if device.type != "cuda" or (not args.fp16):
            class Dummy:
                def __enter__(self): pass
                def __exit__(self, exc_type, exc, tb): pass
            return Dummy()
        if use_new_amp:
            return autocast(device_type="cuda", dtype=torch.float16)
        return autocast()

    # resume
    global_step = 0
    if args.resume is not None and os.path.exists(args.resume):
        logger.info("Resume training from ckpt: %s", args.resume)
        ckpt = torch.load(args.resume, map_location="cpu")

        miss, unexp = load_state_dict_flexible(model, ckpt["model"] if "model" in ckpt else ckpt)
        logger.info("Loaded model. missing=%d unexpected=%d", len(miss), len(unexp))

        if "ema_model" in ckpt:
            miss, unexp = load_state_dict_flexible(ema_model, ckpt["ema_model"])
            logger.info("Loaded ema_model. missing=%d unexpected=%d", len(miss), len(unexp))
        else:
            miss, unexp = load_state_dict_flexible(ema_model, model.state_dict())
            logger.info("No ema_model in ckpt. EMA re-inited. missing=%d unexpected=%d", len(miss), len(unexp))

        if "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
                logger.info("Loaded optimizer state.")
            except Exception as e:
                logger.warning("Failed to load optimizer state (ok to ignore): %s", str(e))

        global_step = int(ckpt.get("step", 0))
        logger.info("Resumed global_step = %d", global_step)

    # train loop
    start_time = time.time()
    start_step_for_timing = global_step
    last_log_time = start_time
    end_time = time.time()

    loss_ema = None
    loss_ema_momentum = 0.98

    logger.info("Start training...")

    it = iter(loader)
    while global_step < args.total_steps:
        try:
            ir, vis, _ = next(it)
        except StopIteration:
            it = iter(loader)
            ir, vis, _ = next(it)

        data_time = time.time() - end_time
        iter_start = time.time()

        global_step += 1

        ir = ir.to(device, non_blocking=True)   # [B,1,H,W] in [-1,1]
        vis = vis.to(device, non_blocking=True) # [B,3,H,W] in [-1,1]
        b = vis.size(0)

        t = torch.randint(0, args.num_timesteps, (b,), device=device, dtype=torch.long)
        noise = torch.randn_like(vis)

        sqrt_ac = extract(sqrt_alphas_cumprod, t, vis.shape)
        sqrt_om = extract(sqrt_one_minus_alphas_cumprod, t, vis.shape)
        x_t = sqrt_ac * vis + sqrt_om * noise

        # classifier-free condition drop (training)
        if args.cond_drop_prob > 0.0:
            keep = (torch.rand((b, 1, 1, 1), device=device) > args.cond_drop_prob).float()
            ir_cond = ir * keep
        else:
            ir_cond = ir

        x_in = torch.cat([x_t, ir_cond], dim=1)  # [B,4,H,W]

        optimizer.zero_grad(set_to_none=True)

        with autocast_ctx():
            eps_pred = model(x_in, t)
            loss = F.mse_loss(eps_pred.float(), noise.float())

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        ema_update()

        iter_time = time.time() - iter_start
        end_time = time.time()

        if loss_ema is None:
            loss_ema = loss.item()
        else:
            loss_ema = loss_ema * loss_ema_momentum + loss.item() * (1 - loss_ema_momentum)

        if global_step % args.log_interval == 0:
            now = time.time()
            elapsed = now - start_time
            steps_done = max(1, global_step - start_step_for_timing)
            it_per_sec = steps_done / max(1e-6, elapsed)
            img_per_sec = it_per_sec * args.batch_size

            remaining = max(0, args.total_steps - global_step)
            eta_sec = remaining / max(1e-6, it_per_sec)

            logger.info(
                "step %d/%d  loss %.6f (ema %.6f)  data %.3fs  iter %.3fs  %.3f it/s  %.3f img/s  ETA %s",
                global_step, args.total_steps,
                loss.item(), loss_ema,
                data_time, iter_time,
                it_per_sec, img_per_sec,
                time.strftime("%H:%M:%S", time.gmtime(eta_sec))
            )
            last_log_time = now

        if global_step % args.save_interval == 0 or global_step == args.total_steps:
            ckpt_path = os.path.join(save_dir, f"ir_cond_vis_diffusion_step_{global_step}.pt")
            save_checkpoint(
                ckpt_path,
                global_step,
                model,
                ema_model,
                optimizer,
                betas.detach().cpu(),
                vars(args),
            )
            logger.info("Saved checkpoint: %s", ckpt_path)

    final_path = os.path.join(save_dir, "ir_cond_vis_diffusion_final.pt")
    save_checkpoint(
        final_path,
        global_step,
        model,
        ema_model,
        optimizer,
        betas.detach().cpu(),
        vars(args),
    )
    logger.info("Finished training. Final checkpoint: %s", final_path)


if __name__ == "__main__":
    main()
