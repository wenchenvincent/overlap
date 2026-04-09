#!/usr/bin/env python3
# Compare CU resource usage across different RCCL collectives.
# Usage: torchrun --nproc_per_node=8 bench_collectives.py

import os
import torch
import torch.distributed as dist

def setup():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_world_size(), dist.get_rank()

def main():
    local_rank, world_size, rank = setup()
    device = torch.device(f"cuda:{local_rank}")

    size_mb = 512
    numel = size_mb * 1024 * 1024 // 2  # bf16

    # Buffers for each collective
    ag_input = torch.randn(numel, dtype=torch.bfloat16, device=device)
    ag_output = torch.empty(world_size * numel, dtype=torch.bfloat16, device=device)

    rs_input = torch.randn(world_size * numel, dtype=torch.bfloat16, device=device)
    rs_output = torch.empty(numel, dtype=torch.bfloat16, device=device)

    # All-reduce on same total size as AG/RS so data volume is comparable.
    # AG: each rank sends numel, total output = world_size * numel
    # RS: each rank sends world_size * numel, total output = numel
    # AR: each rank reduces world_size * numel in-place (2x comms: RS + AG)
    ar_tensor = torch.randn(world_size * numel, dtype=torch.bfloat16, device=device)

    # Warmup all collectives
    for _ in range(3):
        dist.all_gather_into_tensor(ag_output, ag_input)
        dist.reduce_scatter_tensor(rs_output, rs_input)
        dist.all_reduce(ar_tensor)
    torch.cuda.synchronize()

    # Now run each one individually for profiling
    if rank == 0:
        print("Running collectives for profiling...", flush=True)

    # All-gather
    torch.cuda.synchronize()
    dist.barrier()
    dist.all_gather_into_tensor(ag_output, ag_input)
    torch.cuda.synchronize()

    # Reduce-scatter
    dist.barrier()
    dist.reduce_scatter_tensor(rs_output, rs_input)
    torch.cuda.synchronize()

    # All-reduce
    dist.barrier()
    dist.all_reduce(ar_tensor)
    torch.cuda.synchronize()

    if rank == 0:
        print("Done.", flush=True)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
