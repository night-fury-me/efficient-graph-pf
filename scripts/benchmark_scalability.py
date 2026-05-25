"""Scalability benchmark: dense vs matrix-free on Cora (N=2708).

Compares timing and accuracy at subgraph sizes N=50,100,200,500,1000
and the full graph N=2708.
"""

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F_func

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from iem.examples.ignn_cora import IGNN, _download_cora, _load_cora
from iem.adversarial import (
    extract_ego_subgraph,
    _compute_structural_jacobian,
    structural_sensitivity_matrix,
    constrained_sensitivity_matrix,
    optimal_structural_attack,
)
from iem.scalable import ScalableSensitivity, scalable_adversarial_analysis
from iem.certify import spectral_radius


def train_ignn(data, device, epochs=200):
    model = IGNN(data["n_features"], hidden=64, n_classes=data["n_classes"]).to(device)
    X = data["X"].to(device)
    A_hat = data["A_hat"].to(device)
    y = data["y"].to(device)

    optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    best_val, best_state = 0.0, None

    for ep in range(1, epochs + 1):
        model.train()
        logits, _, _ = model(X, A_hat)
        loss = F_func.cross_entropy(logits[data["train_mask"]], y[data["train_mask"]])
        optim.zero_grad()
        loss.backward()
        optim.step()

        if ep % 10 == 0:
            model.eval()
            with torch.no_grad():
                logits, _, _ = model(X, A_hat)
                pred = logits.argmax(dim=1)
                val_acc = float((pred[data["val_mask"]] == y[data["val_mask"]]).float().mean())
            if val_acc > best_val:
                best_val = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits, Z_star, ctx = model(X, A_hat)
        pred = logits.argmax(dim=1)
        test_acc = float((pred[data["test_mask"]] == y[data["test_mask"]]).float().mean())
    return model, Z_star, ctx, test_acc


def get_subgraph(model, Z_star, ctx, A_hat, max_nodes):
    idx = extract_ego_subgraph(A_hat, max_nodes=max_nodes)
    A_sub = A_hat[idx][:, idx]
    X_proj_sub = ctx["X_proj"][idx]
    Z_star_sub = Z_star[idx]
    ctx_sub = {"A_hat": A_sub, "X_proj": X_proj_sub}
    return Z_star_sub, ctx_sub, idx


def benchmark_dense(model, z_star, ctx, label):
    F = model.operator
    N = ctx["A_hat"].shape[0]
    D = z_star.numel()
    n_edges = int((ctx["A_hat"].abs() > 1e-10).sum().item() // 2)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()

    try:
        J_z, J_A, _ = _compute_structural_jacobian(F, z_star, ctx)
        t_jac = time.perf_counter() - t0

        t1 = time.perf_counter()
        S = structural_sensitivity_matrix(F, z_star, ctx, J_z=J_z, J_A=J_A)
        t_solve = time.perf_counter() - t1

        t2 = time.perf_counter()
        S_c, edge_list = constrained_sensitivity_matrix(S, ctx["A_hat"])
        sigma_1 = float(torch.linalg.svdvals(S_c)[0]) if S_c.shape[1] > 0 else 0.0
        t_svd = time.perf_counter() - t2

        total = time.perf_counter() - t0
        mem = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0

        print(f"  {label:>20s} | N={N:5d} D={D:6d} |E|={n_edges:5d} | "
              f"Jac={t_jac:6.1f}s Solve={t_solve:6.1f}s SVD={t_svd:5.1f}s | "
              f"Total={total:7.1f}s | sigma_1={sigma_1:.4f} | Mem={mem:.0f}MB")
        return {"sigma_1": sigma_1, "total": total, "success": True}

    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        total = time.perf_counter() - t0
        print(f"  {label:>20s} | N={N:5d} D={D:6d} |E|={n_edges:5d} | "
              f"OOM after {total:.1f}s: {str(e)[:60]}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"sigma_1": None, "total": total, "success": False}


def benchmark_matfree(model, z_star, ctx, label, k=6, neumann_terms=20):
    F = model.operator
    N = ctx["A_hat"].shape[0]
    D = z_star.numel()
    n_edges = int((ctx["A_hat"].abs() > 1e-10).sum().item() // 2)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    try:
        op = ScalableSensitivity(F, z_star, ctx, neumann_terms=neumann_terms)

        t1 = time.perf_counter()
        U, sigma, Vh = op.top_k_svd(k=min(k, n_edges), n_oversamples=10, n_power_iter=5)
        t_svd = time.perf_counter() - t1
        sigma_1 = float(sigma[0])

        t2 = time.perf_counter()
        top_vulns = op.edge_vulnerability(top_k=10)
        t_vuln = time.perf_counter() - t2

        total = time.perf_counter() - t0
        mem = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0

        print(f"  {label:>20s} | N={N:5d} D={D:6d} |E|={n_edges:5d} | "
              f"SVD={t_svd:6.1f}s Vuln={t_vuln:6.1f}s | "
              f"Total={total:7.1f}s | sigma_1={sigma_1:.4f} | Mem={mem:.0f}MB")
        return {"sigma_1": sigma_1, "total": total, "success": True,
                "top_vulns": top_vulns[:5]}

    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        total = time.perf_counter() - t0
        print(f"  {label:>20s} | N={N:5d} D={D:6d} |E|={n_edges:5d} | "
              f"FAIL after {total:.1f}s: {str(e)[:60]}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"sigma_1": None, "total": total, "success": False}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    data_dir = Path("datasets/cora")
    print("\n=== Loading Cora ===")
    _download_cora(data_dir)
    data = _load_cora(data_dir)
    print(f"  N={data['N']}, features={data['n_features']}, classes={data['n_classes']}")

    print("\n=== Training IGNN (200 epochs, early stopping) ===")
    model, Z_star, ctx, test_acc = train_ignn(data, device, epochs=200)
    A_hat = data["A_hat"].to(device)
    print(f"  Test accuracy: {test_acc:.3f}")
    residual = (model.operator(Z_star, ctx) - Z_star).norm().item()
    print(f"  Fixed-point residual: {residual:.2e}")

    rho = spectral_radius(
        lambda z: model.operator(z.reshape(Z_star.shape), ctx).reshape(-1),
        Z_star, method="power",
    )
    print(f"  Spectral radius rho: {rho:.4f}")

    # ------------------------------------------------------------------
    subgraph_sizes = [50, 100, 200, 500, 1000]
    full_N = data["N"]

    print("\n" + "=" * 110)
    print("SCALABILITY BENCHMARK: Dense vs Matrix-Free")
    print("=" * 110)

    print(f"\n{'Method':>20s} | {'Graph':>25s} | {'Timing':>30s} | {'Total':>8s} | {'sigma_1':>8s} | {'Mem':>6s}")
    print("-" * 110)

    results = {}

    # --- Dense pipeline at various subgraph sizes ---
    print("\n--- Dense Pipeline ---")
    for sz in subgraph_sizes:
        z_sub, ctx_sub, idx = get_subgraph(model, Z_star, ctx, A_hat, sz)
        label = f"Dense N={sz}"
        r = benchmark_dense(model, z_sub, ctx_sub, label)
        results[f"dense_{sz}"] = r
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Matrix-free pipeline at various subgraph sizes ---
    print("\n--- Matrix-Free Pipeline ---")
    for sz in subgraph_sizes:
        z_sub, ctx_sub, idx = get_subgraph(model, Z_star, ctx, A_hat, sz)
        label = f"MatFree N={sz}"
        r = benchmark_matfree(model, z_sub, ctx_sub, label)
        results[f"matfree_{sz}"] = r
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Matrix-free on FULL GRAPH ---
    print("\n--- Matrix-Free: FULL GRAPH (N=2708) ---")
    label = f"MatFree N={full_N}"
    r = benchmark_matfree(model, Z_star, ctx, label, k=6, neumann_terms=15)
    results[f"matfree_{full_N}"] = r

    # ------------------------------------------------------------------
    print("\n" + "=" * 110)
    print("SUMMARY")
    print("=" * 110)

    print(f"\n{'Config':>25s} | {'sigma_1':>10s} | {'Time (s)':>10s} | {'Status':>8s}")
    print("-" * 65)
    for key in sorted(results.keys(), key=lambda k: (k.split("_")[0], int(k.split("_")[1]))):
        r = results[key]
        s1 = f"{r['sigma_1']:.4f}" if r["sigma_1"] is not None else "OOM"
        t = f"{r['total']:.1f}"
        st = "OK" if r["success"] else "FAIL"
        print(f"  {key:>23s} | {s1:>10s} | {t:>10s} | {st:>8s}")

    # Accuracy comparison at overlapping sizes
    print("\n--- Dense vs Matrix-Free Accuracy ---")
    for sz in subgraph_sizes:
        dk = f"dense_{sz}"
        mk = f"matfree_{sz}"
        if results[dk]["success"] and results[mk]["success"]:
            d_s1 = results[dk]["sigma_1"]
            m_s1 = results[mk]["sigma_1"]
            rel_err = abs(d_s1 - m_s1) / (d_s1 + 1e-10)
            speedup = results[dk]["total"] / max(results[mk]["total"], 0.01)
            print(f"  N={sz:4d}: dense_sigma1={d_s1:.4f}  mf_sigma1={m_s1:.4f}  "
                  f"rel_err={rel_err:.4f}  speedup={speedup:.1f}x")

    if results.get(f"matfree_{full_N}", {}).get("success"):
        print(f"\n  FULL GRAPH (N={full_N}): sigma_1={results[f'matfree_{full_N}']['sigma_1']:.4f}  "
              f"time={results[f'matfree_{full_N}']['total']:.1f}s")
        print(f"  Dense pipeline CANNOT run at N={full_N} (D={full_N*64}={full_N*64:,})")

    print("\n=== Benchmark complete ===")


if __name__ == "__main__":
    main()
