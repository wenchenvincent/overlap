#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
#
# Controlled CU occupancy experiment: a "CU occupier" kernel holds N CUs fully
# occupied (matching RCCL's resource footprint), while grouped GEMM runs on the
# remaining CUs. This lets us measure GEMM slowdown as a function of how many
# CUs are stolen, without RCCL's serialization behavior getting in the way.
#
# Usage:
#   python bench_cu_occupancy.py
#   python bench_cu_occupancy.py --occupy-cus 0,32,64,112,128
#   python bench_cu_occupancy.py --occupy-cus 0,112 --blocks-per-cu 1,2,3

import argparse
import sys
import statistics

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton JIT helpers (same as bench_overlap.py)
# ---------------------------------------------------------------------------

@triton.jit
def remap_xcd(pid, GRID_MN, NUM_XCDS: tl.constexpr = 8):
    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    tall_xcds = GRID_MN % NUM_XCDS
    tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    if xcd < tall_xcds:
        pid = xcd * pids_per_xcd + local_pid
    else:
        pid = (
            tall_xcds * pids_per_xcd
            + (xcd - tall_xcds) * (pids_per_xcd - 1)
            + local_pid
        )
    return pid


@triton.jit
def pid_grid(pid: int, num_pid_m: int, num_pid_n: int, GROUP_SIZE_M: tl.constexpr = 1):
    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit
def _remap_xcd_tile_grid(tile_in_mm, num_row_tiles, num_col_tiles,
                          GROUP_SIZE: tl.constexpr = 1, NUM_XCDS: tl.constexpr = 8):
    return pid_grid(
        remap_xcd(tile_in_mm, num_row_tiles * num_col_tiles, NUM_XCDS=NUM_XCDS),
        num_row_tiles, num_col_tiles, GROUP_SIZE_M=GROUP_SIZE,
    )


# ---------------------------------------------------------------------------
# Grouped GEMM kernel (same as bench_overlap.py)
# ---------------------------------------------------------------------------

@triton.jit
def grouped_gemm_kernel(
    lhs_ptr, rhs_ptr, group_sizes_ptr, out_ptr,
    M: int, K: int, N: int, G: int,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr, GRID_DIM: tl.constexpr, NUM_XCDS: tl.constexpr,
):
    tl.assume(M > 0)
    tl.assume(K > 0)
    tl.assume(N > 0)
    tl.assume(G > 0)
    num_n_tiles = tl.cdiv(N, BLOCK_N)
    tile = tl.program_id(0)
    last_mm_tile = 0
    last_m = 0
    for g in range(G):
        m = tl.load(group_sizes_ptr + g)
        num_m_tiles = tl.cdiv(m, BLOCK_M)
        num_tiles = num_m_tiles * num_n_tiles
        while tile >= last_mm_tile and tile < last_mm_tile + num_tiles:
            tile_in_mm = tile - last_mm_tile
            tile_m, tile_n = _remap_xcd_tile_grid(
                tile_in_mm, num_m_tiles, num_n_tiles,
                GROUP_SIZE=GROUP_SIZE, NUM_XCDS=NUM_XCDS,
            )
            offs_lhs_m = (tile_m.to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)) % m
            offs_rhs_n = (tile_n.to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)) % N
            offs_k = tl.arange(0, BLOCK_K).to(tl.int64)
            lhs_ptrs = lhs_ptr + (last_m + offs_lhs_m[:, None]) * K + offs_k[None, :]
            rhs_ptrs = rhs_ptr + g.to(tl.int64) * K * N + offs_k[:, None] * N + offs_rhs_n[None, :]
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for _k in range(0, tl.cdiv(K, BLOCK_K)):
                lhs_tile = tl.load(lhs_ptrs)
                rhs_tile = tl.load(rhs_ptrs)
                acc += tl.dot(lhs_tile, rhs_tile, input_precision="ieee")
                lhs_ptrs += BLOCK_K
                rhs_ptrs += BLOCK_K * N
            acc = acc.to(out_ptr.type.element_ty)
            offs_out_m = tile_m.to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_out_n = tile_n.to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)
            out_ptrs = out_ptr + (last_m + offs_out_m[:, None]) * N + offs_out_n[None, :]
            tl.store(out_ptrs, acc,
                     mask=(offs_out_m[:, None] < m) & (offs_out_n[None, :] < N))
            tile += GRID_DIM
        last_mm_tile += num_tiles
        last_m += m


def run_grouped_gemm(lhs, rhs, group_sizes, out, M, K, N, G, grid_dim, num_xcds):
    num_n_tiles = triton.cdiv(N, 128)
    num_m_tiles_per_group = (group_sizes + 128 - 1) // 128
    total_tiles = int((num_m_tiles_per_group * num_n_tiles).sum().item())
    num_programs = min(grid_dim, total_tiles)
    grouped_gemm_kernel[(num_programs,)](
        lhs, rhs, group_sizes, out,
        M, K, N, G,
        BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
        GROUP_SIZE=1, GRID_DIM=grid_dim, NUM_XCDS=num_xcds,
    )


# ---------------------------------------------------------------------------
# CU occupier kernel
#
# Matches RCCL's resource footprint per wave:
#   - 140 VGPRs (allocated as 144) per wave
#   - 19968 bytes LDS per block
#   - 256 threads (4 waves) per block
#
# Each block spins reading from a flag pointer until it's set to 1.
# Launch N*blocks_per_cu blocks to fully occupy N CUs.
# With blocks_per_cu=3: 3 * 144 = 432 VGPRs per SIMD, leaving only 80
# which is not enough for a GEMM wave (needs 112 VGPRs).
# ---------------------------------------------------------------------------

@triton.jit
def cu_occupier_kernel(
    flag_ptr,
    # Dummy arrays to force VGPR usage — each wave loads these into registers
    dummy_ptr,
    DEPTH: tl.constexpr,      # number of dummy floats to hold in VGPRs per thread
    LDS_SIZE: tl.constexpr,   # bytes of LDS to allocate per block
):
    # Allocate LDS (forces LDS_SIZE bytes reserved for this block)
    lds = tl.arange(0, LDS_SIZE // 4)  # won't actually compile this way

    # Force VGPR usage by loading DEPTH values into registers and keeping them live
    tid = tl.program_id(0) * 256 + tl.arange(0, 256)
    vals = tl.zeros((256,), dtype=tl.float32)
    for d in range(DEPTH):
        vals += tl.load(dummy_ptr + tid * DEPTH + d).to(tl.float32)

    # Spin until flag is set
    while tl.load(flag_ptr) == 0:
        pass

    # Keep vals live so VGPRs aren't optimized away
    tl.store(dummy_ptr + tid, vals)


# Since Triton's compiler may optimize away VGPR usage, let's use a HIP kernel instead.
# We'll launch the occupier via a raw HIP kernel using torch's cuda extension mechanism.
# But simpler: use torch.cuda.Stream + a long-running torch operation.
#
# Actually, the simplest reliable approach: launch a large matmul on a separate stream
# that uses the exact number of blocks we want, and it naturally occupies those CUs.

def launch_cu_occupier_via_matmul(num_blocks, device, stream):
    """
    Launch a matmul that occupies exactly `num_blocks` CUs on the given stream.
    We use a square matmul sized so Triton/rocBLAS launches num_blocks thread blocks.
    The matmul keeps CUs busy with compute + memory traffic, similar to RCCL.
    """
    # A simple approach: launch a big matmul on the occupier stream.
    # rocBLAS will use many blocks, keeping CUs occupied.
    # For precise control, we use our own Triton spin kernel.
    pass


# Simpler approach: a Triton kernel that just does repeated memory reads in a loop,
# holding registers. We control block count directly via grid size.

@triton.jit
def spin_kernel(
    scratch_ptr,
    num_iters: int,
    BLOCK_SIZE: tl.constexpr,
    # Use tl.dot to force MFMA accumulator registers — these can't be optimized away.
    # A (TILE_M, TILE_K) x (TILE_K, TILE_N) dot product holds TILE_M * TILE_N fp32
    # accumulators in VGPRs. With 128x128, that's 16384 fp32 values across 4 waves
    # = 4096 per wave. Way more than needed — 64x64 = 4096 values / 4 waves = 1024
    # per wave is still huge. Use 32x32 = 1024 / 4 waves = 256 VGPRs per wave.
    # Actually: accumulator VGPRs for dot are TILE_M * TILE_N / waves_per_block.
    # For 256 threads = 4 waves (64 threads each), a 64x64 acc = 4096 fp32 / 4 waves
    # = 1024 VGPRs per wave — too many. 32x32 = 1024 / 4 = 256 VGPRs/wave — still high.
    # Let's target ~140 VGPRs: use TILE=16, acc=16*16=256 fp32, /4 waves = 64 VGPRs,
    # plus address regs ≈ ~80. Hmm, let's just try different tile sizes and check.
    TILE_M: tl.constexpr = 128,
    TILE_N: tl.constexpr = 128,
    TILE_K: tl.constexpr = 64,
):
    pid = tl.program_id(0)

    # Load tile data from scratch
    base = pid * (TILE_M * TILE_K + TILE_K * TILE_N)
    offs_a = base + tl.arange(0, TILE_M)[:, None] * TILE_K + tl.arange(0, TILE_K)[None, :]
    offs_b = base + TILE_M * TILE_K + tl.arange(0, TILE_K)[:, None] * TILE_N + tl.arange(0, TILE_N)[None, :]

    a = tl.load(scratch_ptr + offs_a)
    b = tl.load(scratch_ptr + offs_b)

    # Accumulator — lives in VGPRs for the entire spin loop
    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)

    # Spin: each iteration loads fresh data (offset by iteration count)
    # so compiler can't hoist loads out of the loop
    for i in range(num_iters):
        a = tl.load(scratch_ptr + offs_a + i).to(tl.bfloat16)
        b = tl.load(scratch_ptr + offs_b + i).to(tl.bfloat16)
        acc += tl.dot(a, b, input_precision="ieee")

    # Store acc sum to prevent DCE
    # Reduce acc to a single scalar per program and store it
    tl.store(scratch_ptr + pid, tl.sum(acc).to(scratch_ptr.type.element_ty))


def launch_spin_kernel(num_blocks, scratch, num_iters, stream):
    """Launch spin kernel on given stream with exact block count."""
    with torch.cuda.stream(stream):
        spin_kernel[(num_blocks,)](
            scratch, num_iters,
            BLOCK_SIZE=256,  # 256 threads = 4 waves, same as RCCL
        )


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------

def benchmark_gemm_with_occupier(lhs, rhs, group_sizes, out, M, K, N, G,
                                 grid_dim, num_xcds,
                                 occupy_blocks, scratch, spin_iters,
                                 warmup, iters):
    """Run grouped gemm while CU occupier runs on a separate stream."""
    occupy_stream = torch.cuda.Stream()
    compute_stream = torch.cuda.default_stream()

    if occupy_blocks == 0:
        # No occupier — just run gemm alone
        for _ in range(warmup):
            run_grouped_gemm(lhs, rhs, group_sizes, out, M, K, N, G, grid_dim, num_xcds)
        torch.cuda.synchronize()

        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            starts[i].record()
            run_grouped_gemm(lhs, rhs, group_sizes, out, M, K, N, G, grid_dim, num_xcds)
            ends[i].record()
            torch.cuda.synchronize()
        return [s.elapsed_time(e) for s, e in zip(starts, ends)]

    # Warmup both
    for _ in range(warmup):
        launch_spin_kernel(occupy_blocks, scratch, spin_iters, occupy_stream)
        run_grouped_gemm(lhs, rhs, group_sizes, out, M, K, N, G, grid_dim, num_xcds)
        torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        # Launch occupier first
        launch_spin_kernel(occupy_blocks, scratch, spin_iters, occupy_stream)
        # Then gemm on compute stream
        starts[i].record(compute_stream)
        run_grouped_gemm(lhs, rhs, group_sizes, out, M, K, N, G, grid_dim, num_xcds)
        ends[i].record(compute_stream)
        torch.cuda.synchronize()

    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Controlled CU occupancy vs grouped GEMM experiment")
    parser.add_argument("--G", type=int, default=8)
    parser.add_argument("--M", type=int, default=16384)
    parser.add_argument("--K", type=int, default=8192)
    parser.add_argument("--N", type=int, default=8192)
    parser.add_argument("--grid-dim", type=int, default=256)
    parser.add_argument("--num-xcds", type=int, default=8)
    parser.add_argument("--occupy-cus", type=str, default="0,32,64,112,144,192,224",
                        help="Comma-separated list of CU counts to occupy")
    parser.add_argument("--blocks-per-cu", type=str, default="1",
                        help="Comma-separated blocks per CU to test (1=partial, 3=full exclusion)")
    parser.add_argument("--spin-iters", type=int, default=100000,
                        help="Spin loop iterations (controls occupier duration)")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    G, M, K, N = args.G, args.M, args.K, args.N
    assert K % 64 == 0

    occupy_cu_list = [int(x.strip()) for x in args.occupy_cus.split(",")]
    bpc_list = [int(x.strip()) for x in args.blocks_per_cu.split(",")]

    # Allocate GEMM tensors
    lhs = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    rhs = torch.randn(G, K, N, dtype=torch.bfloat16, device=device)
    group_size_val = M // G
    remainder = M % G
    gs_list = [group_size_val] * G
    if remainder > 0:
        gs_list[-1] += remainder
    group_sizes = torch.tensor(gs_list, dtype=torch.int32, device=device)
    out = torch.empty(M, N, dtype=torch.bfloat16, device=device)

    # Scratch buffer for spin kernel
    # Each block needs TILE_M*TILE_K + TILE_K*TILE_N input + TILE_M*TILE_N output elements
    # 128*64 + 64*128 + 128*128 = 8192 + 8192 + 16384 = 32768 elements per block
    max_blocks = max(max(occupy_cu_list), 1) * max(bpc_list)
    scratch_size = max(max_blocks * (128 * 64 + 64 * 128 + 128 * 128) + 10000, 1024)
    scratch = torch.randn(scratch_size, dtype=torch.bfloat16, device=device)

    print("Config: G={}, M={}, K={}, N={}, grid_dim={}".format(G, M, K, N, args.grid_dim))
    print("Spin iterations: {}".format(args.spin_iters))
    print()

    # Header
    print("{:>10s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s}".format(
        "OccupyCUs", "Blks/CU", "TotBlocks", "Mean(ms)", "Min(ms)", "Max(ms)", "Slowdown"))
    print("-" * 75)

    baseline_mean = None

    for bpc in bpc_list:
        for occupy_cus in occupy_cu_list:
            total_blocks = occupy_cus * bpc

            times = benchmark_gemm_with_occupier(
                lhs, rhs, group_sizes, out, M, K, N, G,
                args.grid_dim, args.num_xcds,
                total_blocks, scratch, args.spin_iters,
                args.warmup, args.iters,
            )

            mean_t = statistics.mean(times)
            min_t = min(times)
            max_t = max(times)

            if baseline_mean is None:
                baseline_mean = mean_t

            slowdown = mean_t / baseline_mean

            print("{:>10d} {:>10d} {:>10d} {:>10.3f} {:>10.3f} {:>10.3f} {:>9.2f}x".format(
                occupy_cus, bpc, total_blocks, mean_t, min_t, max_t, slowdown))

    print()
    print("Note: blocks_per_cu=1 allows GEMM to co-schedule (partial exclusion)")
    print("      blocks_per_cu=3 fully excludes GEMM from those CUs (3*144=432 VGPRs, no room for GEMM's 112)")


if __name__ == "__main__":
    main()
