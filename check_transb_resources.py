"""Profile trans_b=False vs trans_b=True CK grouped gemm kernel resources."""
import torch
from primus_turbo.pytorch.ops import grouped_gemm

device = "cuda:0"
torch.cuda.set_device(device)

G, M, K, N = 8, 2048, 8192, 8192
group_lens = torch.full((G,), M, dtype=torch.int64, device=device)
x = torch.randn(G * M, K, device=device, dtype=torch.bfloat16)

w_no_trans = torch.randn(G, K, N, device=device, dtype=torch.bfloat16)
w_trans = torch.randn(G, N, K, device=device, dtype=torch.bfloat16)

# Warmup
for _ in range(3):
    grouped_gemm(x, w_no_trans, group_lens, trans_b=False)
    grouped_gemm(x, w_trans, group_lens, trans_b=True)
torch.cuda.synchronize()

# Run each once for profiling
grouped_gemm(x, w_no_trans, group_lens, trans_b=False)
torch.cuda.synchronize()

grouped_gemm(x, w_trans, group_lens, trans_b=True)
torch.cuda.synchronize()

print("Done")
