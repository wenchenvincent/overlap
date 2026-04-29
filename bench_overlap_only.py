#!/usr/bin/env python3
"""
Minimal: only the overlap phase, looped many times so PMC counter windows
land in the middle of real concurrent execution. No gemm-only / sequential phases.
"""
import argparse, os, sys, time
import torch
import torch.distributed as dist


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--G", type=int, default=32)
    p.add_argument("--M", type=int, default=267424)
    p.add_argument("--K", type=int, default=1280)
    p.add_argument("--N", type=int, default=2560)
    p.add_argument("--ag-size-mb", type=int, default=512)
    p.add_argument("--grid-dims", type=int, default=228)
    p.add_argument("--num-xcds", type=int, default=8)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--seconds", type=float, default=30.0,
                   help="Run overlap iters for this many seconds")
    args = p.parse_args()

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    G, M, K, N = args.G, args.M, args.K, args.N
    grid_dim = args.grid_dims

    from primus_turbo.pytorch.ops import grouped_gemm as pt_grouped_gemm

    lhs = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    rhs = torch.randn(G, N, K, dtype=torch.bfloat16, device=device)
    gs_list = [M // G] * G
    if M % G:
        gs_list[-1] += M % G
    group_lens = torch.tensor(gs_list, dtype=torch.int64, device=device)
    out = torch.empty(M, N, dtype=torch.bfloat16, device=device)

    ag_numel = args.ag_size_mb * 1024 * 1024 // 2
    ag_in = torch.randn(ag_numel, dtype=torch.bfloat16, device=device)
    ag_out = torch.empty(world_size * ag_numel, dtype=torch.bfloat16, device=device)
    comm_stream = torch.cuda.Stream(device=device)

    def gemm():
        return pt_grouped_gemm(lhs, rhs, group_lens, trans_b=True, num_cu=grid_dim)

    # Warmup
    for _ in range(args.warmup):
        with torch.cuda.stream(comm_stream):
            dist.all_gather_into_tensor(ag_out, ag_in, async_op=True)
        gemm()
        torch.cuda.synchronize()
    dist.barrier()

    if rank == 0:
        print(f"[rank0] starting {args.seconds}s overlap loop, grid_dim={grid_dim}, "
              f"NCCL_MAX_NCHANNELS={os.environ.get('NCCL_MAX_NCHANNELS','default')}",
              file=sys.stderr, flush=True)

    compute_stream = torch.cuda.default_stream()
    deadline = time.time() + args.seconds
    n = 0
    while time.time() < deadline:
        with torch.cuda.stream(comm_stream):
            dist.all_gather_into_tensor(ag_out, ag_in, async_op=True)
        gemm()
        compute_stream.wait_stream(comm_stream)
        torch.cuda.synchronize()
        n += 1

    if rank == 0:
        print(f"[rank0] completed {n} overlap iters", file=sys.stderr, flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
