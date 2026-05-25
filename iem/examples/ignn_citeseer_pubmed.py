"""IEM + N-1 validation on Citeseer and Pubmed (Planetoid format).

Same loader as Cora, just different dataset names. Adds two more citation
domains (CS-focused + biomedical) to the IEM validation matrix.

Citeseer: 3,327 nodes, 4,732 edges, 6 classes, 3,703 features
Pubmed:  19,717 nodes, 44,338 edges, 3 classes, 500 features

Usage:
    .venv/bin/python -m iem.examples.ignn_citeseer_pubmed
"""

from __future__ import annotations

import pickle
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from iem import IEMiner
from iem.ift import compute_jacobian
from iem.examples.ignn_cora import IGNN

PLANETOID_URL = "https://github.com/kimiyoung/planetoid/raw/master/data/"


def _download_planetoid(name: str, data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    files = [
        f"ind.{name}.x", f"ind.{name}.tx", f"ind.{name}.allx",
        f"ind.{name}.y", f"ind.{name}.ty", f"ind.{name}.ally",
        f"ind.{name}.graph", f"ind.{name}.test.index",
    ]
    for fname in files:
        dst = data_dir / fname
        if not dst.exists():
            print(f"    downloading {fname}...", flush=True)
            urllib.request.urlretrieve(PLANETOID_URL + fname, str(dst))


def _load_planetoid(name: str, data_dir: Path) -> dict:
    """Load Planetoid dataset (cora/citeseer/pubmed) into dense tensors."""
    _download_planetoid(name, data_dir)

    def _pkl(fname):
        with open(data_dir / fname, "rb") as f:
            return pickle.load(f, encoding="latin1")

    x = _pkl(f"ind.{name}.x")
    tx = _pkl(f"ind.{name}.tx")
    allx = _pkl(f"ind.{name}.allx")
    y = _pkl(f"ind.{name}.y")
    ty = _pkl(f"ind.{name}.ty")
    ally = _pkl(f"ind.{name}.ally")
    graph = _pkl(f"ind.{name}.graph")

    test_idx = []
    with open(data_dir / f"ind.{name}.test.index") as f:
        for line in f:
            test_idx.append(int(line.strip()))
    test_idx = np.array(test_idx)
    test_idx_sorted = np.sort(test_idx)

    features = sp.vstack([allx, tx]).tolil()
    # Citeseer has isolated test nodes with indices beyond allx+tx row count.
    # Extend the matrix to cover all indices.
    max_idx = max(test_idx.max(), features.shape[0] - 1)
    if max_idx >= features.shape[0]:
        features.resize((max_idx + 1, features.shape[1]))
    features[test_idx] = features[test_idx_sorted]
    X = torch.tensor(features.toarray(), dtype=torch.float32)

    labels = np.vstack([ally, ty])
    if max_idx >= labels.shape[0]:
        labels = np.vstack([labels, np.zeros((max_idx + 1 - labels.shape[0], labels.shape[1]))])
    labels[test_idx] = labels[test_idx_sorted]
    y_tensor = torch.tensor(labels.argmax(axis=1), dtype=torch.long)

    N = X.shape[0]
    adj = sp.lil_matrix((N, N), dtype=np.float32)
    for src, dsts in graph.items():
        for dst in dsts:
            adj[src, dst] = 1.0
            adj[dst, src] = 1.0
    adj = adj + sp.eye(N)
    deg = np.array(adj.sum(axis=1)).flatten()
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
    D_inv_sqrt = sp.diags(deg_inv_sqrt)
    A_hat = D_inv_sqrt @ adj @ D_inv_sqrt
    A_hat_dense = torch.tensor(A_hat.toarray(), dtype=torch.float32)

    # Standard Planetoid split
    n_train_per_class = 20
    n_val = 500
    train_mask = torch.zeros(N, dtype=torch.bool)
    train_mask[:n_train_per_class * int(y_tensor.max() + 1)] = True
    val_mask = torch.zeros(N, dtype=torch.bool)
    val_end = n_train_per_class * int(y_tensor.max() + 1) + n_val
    val_mask[n_train_per_class * int(y_tensor.max() + 1):val_end] = True
    test_mask = torch.zeros(N, dtype=torch.bool)
    test_mask[test_idx] = True

    return {
        "X": X, "A_hat": A_hat_dense, "y": y_tensor, "N": N,
        "n_features": X.shape[1], "n_classes": int(y_tensor.max()) + 1,
        "train_mask": train_mask, "val_mask": val_mask, "test_mask": test_mask,
    }


def run_full_iem_n1(name: str, data: dict, device):
    """Train IGNN + IEM Shapley + N-1 edge ranking on one dataset."""
    import torch.nn.functional as F_func

    print(f"\n{'='*60}", flush=True)
    print(f"=== {name}: N={data['N']}, feat={data['n_features']}, classes={data['n_classes']} ===", flush=True)

    X = data["X"].to(device)
    A_hat = data["A_hat"].to(device)
    y = data["y"].to(device)

    model = IGNN(data["n_features"], hidden=64, n_classes=data["n_classes"]).to(device)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for ep in range(1, 51):
        model.train()
        logits, _, _ = model(X, A_hat)
        loss = F_func.cross_entropy(logits[data["train_mask"]], y[data["train_mask"]])
        optim.zero_grad()
        loss.backward()
        optim.step()
        if ep % 25 == 0:
            model.eval()
            with torch.no_grad():
                logits, _, _ = model(X, A_hat)
                pred = logits.argmax(dim=1)
                val_acc = float((pred[data["val_mask"]] == y[data["val_mask"]]).float().mean())
                test_acc = float((pred[data["test_mask"]] == y[data["test_mask"]]).float().mean())
            print(f"  ep {ep} | loss {loss.item():.4f} | val {val_acc:.3f} | test {test_acc:.3f}", flush=True)

    model.eval()
    with torch.no_grad():
        logits, Z_star, ctx = model(X, A_hat)
        pred = logits.argmax(dim=1)
        test_acc = float((pred[data["test_mask"]] == y[data["test_mask"]]).float().mean())
    residual = (model.operator(Z_star, ctx) - Z_star).norm().item()
    print(f"  Final: test_acc={test_acc:.3f}, residual={residual:.2e}", flush=True)

    # Subgraph for IEM
    deg = A_hat.sum(dim=1)
    center = int(deg.argmax().item())
    neighbors = (A_hat[center] > 0).nonzero(as_tuple=True)[0]
    idx = neighbors[:50]
    S = len(idx)
    A_sub = A_hat[idx][:, idx]
    X_proj_sub = ctx["X_proj"][idx]
    Z_sub = Z_star[idx]
    ctx_sub = {"A_hat": A_sub, "X_proj": X_proj_sub}

    edges = [(i, j) for i in range(S) for j in range(i+1, S) if A_sub[i, j].abs() > 1e-6]
    print(f"  Subgraph: {S} nodes, {len(edges)} edges", flush=True)

    # Contractivity + Shapley
    miner = IEMiner(lambda z, c=ctx_sub: model.operator(z, c), Z_sub, ctx_sub, method="direct")
    rho = miner.rho
    print(f"  rho={rho:.4f}, contractive={rho < 1}", flush=True)

    phi = miner.node_shapley("X_proj")
    n_nz = int((phi > 1e-6).sum().item())
    print(f"  Shapley: {n_nz}/{S} nonzero", flush=True)

    cert = miner.certified_bound(phi, epsilon=0.1)
    print(f"  Certified: max_bound={cert['max_bound']:.2e}", flush=True)

    # N-1 edge ranking
    if len(edges) < 3:
        print(f"  Too few edges for N-1", flush=True)
        return {"name": name, "acc": test_acc, "rho": rho, "n_nz": n_nz,
                "bound": cert["max_bound"], "tau": None}

    D = Z_sub.numel()
    def F_z(z):
        return model.operator(z.reshape(Z_sub.shape), ctx_sub).reshape(-1)
    J = compute_jacobian(F_z, Z_sub)
    I_mat = torch.eye(D, device=device, dtype=J.dtype)
    A_sys = I_mat - J
    try:
        A_inv = torch.linalg.inv(A_sys)
    except:
        lam = max(rho - 0.99, 0.01)
        A_inv = torch.linalg.inv((1 + lam) * I_mat - J)

    # Brute-force
    t0 = time.time()
    bf_scores = torch.zeros(len(edges), device=device)
    with torch.no_grad():
        for i_e, (i, j) in enumerate(edges):
            A_pert = A_sub.clone()
            A_pert[i, j] = 0.0
            A_pert[j, i] = 0.0
            ctx_p = {**ctx_sub, "A_hat": A_pert}
            Z = Z_sub.clone()
            for _ in range(50):
                Z = model.operator(Z, ctx_p)
            bf_scores[i_e] = (Z - Z_sub).norm().item()
    bf_time = time.time() - t0

    # IEM
    t0 = time.time()
    iem_scores = torch.zeros(len(edges), device=device)
    eps = 1e-4
    with torch.no_grad():
        f_base = model.operator(Z_sub, ctx_sub).reshape(-1)
        for i_e, (i, j) in enumerate(edges):
            A_pert = A_sub.clone()
            A_pert[i, j] += eps
            A_pert[j, i] += eps
            f_pert = model.operator(Z_sub, {**ctx_sub, "A_hat": A_pert}).reshape(-1)
            dz = A_inv @ ((f_pert - f_base) / eps)
            iem_scores[i_e] = dz.norm().item()
    iem_time = time.time() - t0

    tau, p = kendalltau(bf_scores.cpu().numpy(), iem_scores.cpu().numpy())
    k = min(5, len(edges))
    bf_top = set(bf_scores.argsort(descending=True)[:k].tolist())
    iem_top = set(iem_scores.argsort(descending=True)[:k].tolist())
    agree = len(bf_top & iem_top) / k

    print(f"  N-1: τ={tau:+.3f} (p={p:.2e}), top-{k}={agree:.0%}, "
          f"BF={bf_time:.1f}s IEM={iem_time:.1f}s", flush=True)

    return {"name": name, "acc": test_acc, "rho": rho, "n_nz": n_nz,
            "bound": cert["max_bound"], "tau": tau, "top5": agree}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets = [
        ("Citeseer (CS Citations)", _load_planetoid("citeseer", Path("datasets/citeseer"))),
        ("Pubmed (Biomedical)", _load_planetoid("pubmed", Path("datasets/pubmed"))),
    ]

    results = []
    for name, data in datasets:
        r = run_full_iem_n1(name, data, device)
        results.append(r)

    print(f"\n{'='*60}", flush=True)
    print("=== CITESEER + PUBMED SUMMARY ===", flush=True)
    for r in results:
        tau_str = f"{r['tau']:+.3f}" if r.get('tau') is not None else "N/A"
        top5_str = f"{r.get('top5', 0):.0%}" if r.get('top5') is not None else "N/A"
        print(f"  {r['name']:<30} acc={r['acc']:.3f} ρ={r['rho']:.3f} "
              f"shap={r['n_nz']}/50 bound={r['bound']:.2e} τ={tau_str} top5={top5_str}", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
