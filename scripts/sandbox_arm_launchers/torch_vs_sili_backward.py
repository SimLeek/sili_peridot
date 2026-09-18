import time

import torch

torch.manual_seed(0)

LAYER_SHAPES = {
    "input_proj (36->288)": (36, 288),
    "q/k/v/o_proj (288->288)": (288, 288),
}

N_CALLS = 4500  # matches sili's profiled _bwd call count (900 steps x 5 layers)


def bench(n_in, n_out, device, n_calls=N_CALLS):
    layer = torch.nn.Linear(n_in, n_out, bias=False).to(device)
    opt = torch.optim.RMSprop(layer.parameters(), lr=1e-3)
    x = torch.randn(1, n_in, device=device)
    # warmup
    for _ in range(20):
        opt.zero_grad()
        y = layer(x)
        loss = y.sum()
        loss.backward()
        opt.step()
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_calls):
        opt.zero_grad()
        y = layer(x)
        loss = y.sum()
        loss.backward()
        opt.step()
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    total = t1 - t0
    return total, total / n_calls


print(f"torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")
print(f"N_CALLS per shape: {N_CALLS}\n")

devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
for device in devices:
    print(f"=== device: {device} ===")
    for name, (n_in, n_out) in LAYER_SHAPES.items():
        total, per_call = bench(n_in, n_out, device)
        print(
            f"  {name}: {total:.3f}s total / {N_CALLS} calls = {per_call * 1000:.4f} ms/call ({1 / per_call:.1f} calls/sec)"
        )
    print()

print("Reference: sili's measured _bwd cost (profiled tonight, fp32, r_target_min=0.5, NUM_CPUS=4):")
print("  4500 calls in 83.85s tottime = 18.6 ms/call (53.7 calls/sec)")
