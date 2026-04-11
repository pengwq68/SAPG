"""
Various utilities for neural networks.
"""

import math

import torch as th
import torch.nn as nn

try:
    from torch.amp import autocast
except Exception:
    from torch.cuda.amp import autocast

# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
class SiLU(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.

    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


# class CheckpointFunction(th.autograd.Function):
#     @staticmethod
#     def forward(ctx, run_function, length, *args):
#         ctx.run_function = run_function
#         ctx.input_tensors = list(args[:length])
#         ctx.input_params = list(args[length:])
#         with th.no_grad():
#             output_tensors = ctx.run_function(*ctx.input_tensors)
#         return output_tensors

#     @staticmethod
#     def backward(ctx, *output_grads):
#         ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
#         with th.enable_grad():
#             # Fixes a bug where the first op in run_function modifies the
#             # Tensor storage in place, which is not allowed for detach()'d
#             # Tensors.
#             shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
#             output_tensors = ctx.run_function(*shallow_copies)
#         input_grads = th.autograd.grad(
#             output_tensors,
#             ctx.input_tensors + ctx.input_params,
#             output_grads,
#             allow_unused=True,
#         )
#         del ctx.input_tensors
#         del ctx.input_params
#         del output_tensors
#         return (None, None) + input_grads
# class CheckpointFunction(th.autograd.Function):
#     @staticmethod
#     def forward(ctx, run_function, length, *args):
#         ctx.run_function = run_function
#         ctx.input_tensors = list(args[:length])
#         ctx.input_params = list(args[length:])
#         ctx.save_for_backward(*args)

#         # 记录 forward 时 autocast 状态（AMP）
#         try:
#             ctx.autocast_enabled = th.is_autocast_enabled()
#         except Exception:
#             ctx.autocast_enabled = False
#         try:
#             ctx.autocast_dtype = th.get_autocast_gpu_dtype()
#         except Exception:
#             ctx.autocast_dtype = th.float16

#         with th.no_grad():
#             output_tensors = ctx.run_function(*ctx.input_tensors)
#         return output_tensors

#     @staticmethod
#     def backward(ctx, *output_grads):
#         # 重新构建需要梯度的输入（input_tensors 一定要）
#         ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]

#         with th.enable_grad():
#             output_tensors = ctx.run_function(*ctx.input_tensors)

#         # output_tensors 可能是 Tensor 或 tuple/list
#         if isinstance(output_tensors, th.Tensor):
#             outputs = (output_tensors,)
#         else:
#             outputs = tuple(output_tensors)

#         # 要求导的候选：输入 + 参数
#         candidates = list(ctx.input_tensors) + list(ctx.input_params)

#         # 只把 requires_grad=True 的张量送进 autograd.grad，避免报错
#         req_mask = [getattr(t, "requires_grad", False) for t in candidates]
#         req_tensors = [t for t, m in zip(candidates, req_mask) if m]

#         if len(req_tensors) == 0:
#             grads_req = []
#         else:
#             grads_req = th.autograd.grad(
#                 outputs=outputs,
#                 inputs=req_tensors,
#                 grad_outputs=output_grads,
#                 allow_unused=True,
#                 retain_graph=False,
#                 create_graph=False,
#             )

#         # 把梯度填回到 candidates 的位置，不需要的填 None
#         grads_full = []
#         it = iter(grads_req)
#         for m in req_mask:
#             grads_full.append(next(it) if m else None)

#         # 对应 apply 的前两个参数 (run_function, length) 返回 None
#         return (None, None) + tuple(grads_full)

class CheckpointFunction(th.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])

        # 记录 autocast 状态（exists in most torch versions; fallback safe）
        try:
            ctx.autocast_enabled = th.is_autocast_enabled()
        except Exception:
            ctx.autocast_enabled = False

        # 记录 dtype（不同 torch 版本 API 不一致，尽量兜底）
        dtype = th.float16
        try:
            dtype = th.get_autocast_gpu_dtype()
        except Exception:
            try:
                # torch>=2.0 sometimes provides get_autocast_dtype("cuda")
                dtype = th.get_autocast_dtype("cuda")
            except Exception:
                dtype = th.float16
        ctx.autocast_dtype = dtype

        with th.no_grad():
            outputs = ctx.run_function(*ctx.input_tensors)
        return outputs

    @staticmethod
    def backward(ctx, *output_grads):
        # 让输入变成需要梯度的叶子
        inputs = [x.detach().requires_grad_(True) for x in ctx.input_tensors]

        with th.enable_grad():
            if th.cuda.is_available():
                # 关键：重算 forward 时恢复 autocast（兼容两个 autocast 签名）
                try:
                    # torch.amp.autocast 版本：需要 device_type
                    with autocast(device_type="cuda",
                                  enabled=ctx.autocast_enabled,
                                  dtype=ctx.autocast_dtype):
                        outputs = ctx.run_function(*inputs)
                except TypeError:
                    # torch.cuda.amp.autocast 版本：没有 device_type 参数
                    with autocast(enabled=ctx.autocast_enabled,
                                  dtype=ctx.autocast_dtype):
                        outputs = ctx.run_function(*inputs)
            else:
                outputs = ctx.run_function(*inputs)

        if isinstance(outputs, th.Tensor):
            outputs = (outputs,)

        grads = th.autograd.grad(
            outputs,
            inputs + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        return (None, None) + grads



