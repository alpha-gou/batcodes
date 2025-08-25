from typing import Tuple

import torch
import triton
import triton.language as tl
from triton import Config


@triton.jit
def act_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs).to(tl.float32)
    s = tl.max(tl.abs(x)) / 448.0
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y)
    tl.store(s_ptr + pid, s)


def act_quant(
    x: torch.Tensor, block_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.is_contiguous()
    assert x.size(-1) % block_size == 0
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)
    act_quant_kernel[grid](x, y, s, BLOCK_SIZE=block_size)
    return y, s


@triton.jit
def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.load(s_ptr + pid_m * n + pid_n)
    y = x * s  # 直接 *s，所以误差就在 fp 从 fp32 存储为 fp8_e4m3 阶段
    tl.store(y_ptr + offs, y, mask=mask)


def weight_dequant(
    x: torch.Tensor, s: torch.Tensor, block_size: int = 128
) -> torch.Tensor:
    assert x.is_contiguous() and s.is_contiguous()
    assert x.dim() == 2 and s.dim() == 2
    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.get_default_dtype())
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE"]),
        triton.cdiv(N, meta["BLOCK_SIZE"]),
    )
    weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
    return y


# finfo = torch.finfo(torch.float8_e4m3fn)
# finfo(resolution=1, min=-448, max=448, eps=0.125, smallest_normal=0.015625, tiny=0.015625, dtype=float8_e4m3fn)
# jit func 不能有默认值 不然会编译报错
@triton.jit
def weight_quant_kernel(x_ptr, y_ptr, s_ptr, M, N, scale, BLOCK_SIZE: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    # 对称量化
    # 相较 max_val 的放缩: y = 448 * (x / absmax) -> 将 absmax 映射到 448 占满区间
    s = tl.max(tl.abs(x)) / scale
    y = x / s
    # 如果 / 15, FP 可能会有很大的数 overflow float8_e4m3fn 边界, 比如
    y = y.to(y_ptr.dtype.element_ty)  # 误差发生的地方
    tl.store(y_ptr + offs, y, mask=mask)  # y is quanted_x float8_e4m3fn
    tl.store(s_ptr + pid_m * n + pid_n, s)  # scale


# quant to block float8
def weight_quant(
    x: torch.Tensor, block_size: int = 128, scale: float = 448.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.is_contiguous()
    assert x.dim() == 2
    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    # store scales
    sM, sN = (
        torch.tensor(1.0 * M / block_size).ceil().int(),
        torch.tensor(1.0 * N / block_size).ceil().int(),
    )
    s = x.new_empty(sM, sN, dtype=torch.float32)  # scale fp32 存，没影响
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE"]),  # cell div 向上取整
        triton.cdiv(N, meta["BLOCK_SIZE"]),
    )
    weight_quant_kernel[grid](x, y, s, M, N, scale, BLOCK_SIZE=block_size)
    return y, s
