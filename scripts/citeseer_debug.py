"""Diagnostic script for Citeseer IGNN accuracy investigation."""
from __future__ import annotations
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F_func
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from iem.examples.ignn_cora import IGNN

PLANETOID_URL = "https://github.com/kimiyoung/planetoid/raw/master/data/"


def _download_planetoid(name, data_dir):
    import urllib.request
    data_dir.mkdir(parents=True, exist_ok=True)
    files = [
        f"ind.{name}.x", f"ind.{name}.tx", f"ind.{name}.allx",
        f"ind.{name}.y", f"ind.{name}.ty", f"ind.{name}.ally",
        f"ind.{name}.graph", f"ind.{name}.test.index",
    ]
    for fname in files:
        dst = data_dir / fname
        if not dst.exists():
            print(f"  downloading {fname}...")
            urllib.request.urlretrieve(PLANETOID_URL + fname, str(dst))


def _load_planetoid_original(name, data_dir):
    """Original loader from the codebase."""
    _download_planetoid(name, data_dir)

    def _pkl(fname):
        with open(data_dir / fname, "rb") as f:
            return pickle.load(f, encoding="latin1")

    x = _pkl(f"ind.{name}.x")
    tx = _pkl(f"ind.{name}.tx")
    allx = _pkl(f"ind.{name}.allx")
    y_raw = _pkl(f"ind.{name}.y")
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
    max_idx = max(test_idx.max(), features.shape[0] - 1)
    if max_idx >= features.shape[0]:
        features.resize((max_idx + 1, features.shape[1]))
    features[test_idx] = features[test_idx_sorted]
    X = torch.tensor(features.toarray(), dtype=torch.float32)

    labels = np.vstack([ally, ty])
    if max_idx >= labels.shape[0]:
        n_ext = max_idx + 1 - labels.shape[0]
        labels = np.vstack([labels, np.zeros((n_ext, labels.shape[1]))])
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

    n_train_per_class = 20
    n_val = 500
    n_classes = int(y_tensor.max()) + 1
    train_mask = torch.zeros(N, dtype=torch.bool)
    train_mask[:n_train_per_class * n_classes] = True
    val_mask = torch.zeros(N, dtype=torch.bool)
    val_start = n_train_per_class * n_classes
    val_mask[val_start:val_start + n_val] = True
    test_mask = torch.zeros(N, dtype=torch.bool)
    test_mask[test_idx] = True

    return {
        "X": X, "A_hat": A_hat_dense, "y": y_tensor, "N": N,
        "n_features": X.shape[1], "n_classes": n_classes,
        "train_mask": train_mask, "val_mask": val_mask, "test_mask": test_mask,
        "test_idx": test_idx, "adj_raw": adj, "labels_raw": labels,
    }


class SimpleGCN(nn.Module):
    def __init__(self, n_features, hidden, n_classes, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(n_features, hidden)
        self.fc2 = nn.Linear(hidden, n_classes)
        self.dropout = dropout

    def forward(self, X, A_hat):
        H = F_func.relu(A_hat @ self.fc1(X))
        H = F_func.dropout(H, p=self.dropout, training=self.training)
        return A_hat @ self.fc2(H)


def train_model(model, X, A_hat, y, train_mask, val_mask, test_mask,
                epochs=200, lr=0.01, wd=5e-4, label=""):
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_val = 0.0
    best_test = 0.0
    for ep in range(1, epochs + 1):
        model.train()
        if isinstance(model, IGNN):
            logits, _, _ = model(X, A_hat)
        else:
            logits = model(X, A_hat)
        loss = F_func.cross_entropy(logits[train_mask], y[train_mask])
        optim.zero_grad()
        loss.backward()
        optim.step()

        if ep % 50 == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                if isinstance(model, IGNN):
                    logits, _, _ = model(X, A_hat)
                else:
                    logits = model(X, A_hat)
                pred = logits.argmax(dim=1)
                val_acc = (pred[val_mask] == y[val_mask]).float().mean().item()
                test_acc = (pred[test_mask] == y[test_mask]).float().mean().item()
                if val_acc > best_val:
                    best_val = val_acc
                    best_test = test_acc
                print(f"  [{label}] ep {ep:3d} | loss {loss.item():.4f} | val {val_acc:.3f} | test {test_acc:.3f}")
    return best_val, best_test


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_dir = Path("datasets/citeseer")
    print("=" * 60)
    print("LOADING CITESEER")
    data = _load_planetoid_original("citeseer", data_dir)

    X = data["X"]
    y = data["y"]
    A_hat = data["A_hat"]
    train_mask = data["train_mask"]
    val_mask = data["val_mask"]
    test_mask = data["test_mask"]
    N = data["N"]

    # ── DIAGNOSTIC 1: Data shape sanity ──
    print("=" * 60)
    print("DIAGNOSTIC 1: Data shapes")
    print(f"  N = {N}")
    print(f"  X.shape = {X.shape}")
    print(f"  y.shape = {y.shape}")
    print(f"  A_hat.shape = {A_hat.shape}")
    print(f"  n_classes = {data['n_classes']}")
    print(f"  train: {train_mask.sum().item()}, val: {val_mask.sum().item()}, test: {test_mask.sum().item()}")
    print(f"  test_idx range: [{data['test_idx'].min()}, {data['test_idx'].max()}]")

    # ── DIAGNOSTIC 2: Zero-feature nodes ──
    print("=" * 60)
    print("DIAGNOSTIC 2: Zero-feature nodes")
    row_norms = X.norm(dim=1)
    zero_feat = (row_norms == 0)
    print(f"  Total zero-feature nodes: {zero_feat.sum().item()} / {N}")
    print(f"  Zero-feat in train: {(zero_feat & train_mask).sum().item()}")
    print(f"  Zero-feat in val:   {(zero_feat & val_mask).sum().item()}")
    print(f"  Zero-feat in test:  {(zero_feat & test_mask).sum().item()}")
    if zero_feat.sum() > 0:
        zf_labels = y[zero_feat]
        print(f"  Zero-feat label dist: {torch.bincount(zf_labels, minlength=data['n_classes']).tolist()}")

    # ── DIAGNOSTIC 3: Label distribution ──
    print("=" * 60)
    print("DIAGNOSTIC 3: Label distribution")
    for split_name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        counts = torch.bincount(y[mask], minlength=data['n_classes'])
        print(f"  {split_name:5s}: {counts.tolist()} (total={counts.sum().item()})")

    # ── DIAGNOSTIC 4: Extended nodes with fake label=0 ──
    print("=" * 60)
    print("DIAGNOSTIC 4: Extended nodes with fake label=0")
    labels_raw = data["labels_raw"]
    zero_label_rows = (labels_raw.sum(axis=1) == 0)
    n_zero_label = zero_label_rows.sum()
    print(f"  Nodes with all-zero label row (fake class 0): {n_zero_label}")
    zero_label_mask = torch.tensor(zero_label_rows, dtype=torch.bool)
    print(f"  Zero-label in train: {(zero_label_mask & train_mask).sum().item()}")
    print(f"  Zero-label in val:   {(zero_label_mask & val_mask).sum().item()}")
    print(f"  Zero-label in test:  {(zero_label_mask & test_mask).sum().item()}")
    if (zero_label_mask & test_mask).sum() > 0:
        print("  *** BUG: Zero-label nodes in test set assigned class 0 by argmax ***")
        print("  *** This corrupts test accuracy! ***")

    # ── DIAGNOSTIC 5: Isolated nodes ──
    print("=" * 60)
    print("DIAGNOSTIC 5: Isolated nodes")
    adj_raw = data["adj_raw"]
    raw_deg = np.array(adj_raw.sum(axis=1)).flatten()
    isolated = (raw_deg <= 1.0)  # only self-loop
    isolated_t = torch.tensor(isolated, dtype=torch.bool)
    print(f"  Isolated nodes (deg<=1 incl self-loop): {isolated.sum()}")
    print(f"  Isolated in test: {(isolated_t & test_mask).sum().item()}")

    # ── DIAGNOSTIC 6: Feature statistics ──
    print("=" * 60)
    print("DIAGNOSTIC 6: Feature statistics")
    print(f"  X mean: {X.mean().item():.6f}, std: {X.std().item():.6f}")
    print(f"  X row-norm: mean={row_norms.mean():.4f}, std={row_norms.std():.4f}, min={row_norms.min():.4f}, max={row_norms.max():.4f}")
    X_norm = X.clone()
    rnorm = X_norm.norm(dim=1, keepdim=True)
    rnorm[rnorm == 0] = 1.0
    X_norm = X_norm / rnorm

    # ── EXPERIMENTS ──
    X_d = X.to(device)
    A_d = A_hat.to(device)
    y_d = y.to(device)
    X_norm_d = X_norm.to(device)

    print("=" * 60)
    print("EXPERIMENT 1: IGNN original (50 epochs, as in codebase)")
    model_ignn = IGNN(data["n_features"], hidden=64, n_classes=data["n_classes"]).to(device)
    val1, test1 = train_model(model_ignn, X_d, A_d, y_d, train_mask, val_mask, test_mask,
                              epochs=50, label="IGNN-orig")
    # Spectral norm of W
    model_ignn.eval()
    with torch.no_grad():
        W_weight = model_ignn.W.weight
        sv = torch.linalg.svdvals(W_weight)
        print(f"  W spectral norm (sigma_max): {sv[0].item():.6f}")
        print(f"  W sigma_min: {sv[-1].item():.6f}")

    print("=" * 60)
    print("EXPERIMENT 2: Simple 2-layer GCN (200 epochs)")
    model_gcn = SimpleGCN(data["n_features"], 64, data["n_classes"]).to(device)
    val2, test2 = train_model(model_gcn, X_d, A_d, y_d, train_mask, val_mask, test_mask,
                              epochs=200, label="GCN-orig")

    print("=" * 60)
    print("EXPERIMENT 3: IGNN with row-normalized features (200 epochs)")
    model_ignn_norm = IGNN(data["n_features"], hidden=64, n_classes=data["n_classes"]).to(device)
    val3, test3 = train_model(model_ignn_norm, X_norm_d, A_d, y_d, train_mask, val_mask, test_mask,
                              epochs=200, label="IGNN-rownorm")

    print("=" * 60)
    print("EXPERIMENT 4: GCN with row-normalized features (200 epochs)")
    model_gcn_norm = SimpleGCN(data["n_features"], 64, data["n_classes"]).to(device)
    val4, test4 = train_model(model_gcn_norm, X_norm_d, A_d, y_d, train_mask, val_mask, test_mask,
                              epochs=200, label="GCN-rownorm")

    print("=" * 60)
    print("EXPERIMENT 5: IGNN WITHOUT spectral norm + row-norm (200 epochs)")
    model_ignn_nosn = IGNN(data["n_features"], hidden=64, n_classes=data["n_classes"],
                           spectral_norm=False).to(device)
    val5, test5 = train_model(model_ignn_nosn, X_norm_d, A_d, y_d, train_mask, val_mask, test_mask,
                              epochs=200, label="IGNN-noSN")

    # ── EXPERIMENT 6: Clean test accuracy (exclude fake-label nodes) ──
    print("=" * 60)
    print("EXPERIMENT 6: Test accuracy excluding zero-label (fake) nodes")
    clean_test_mask = test_mask & ~zero_label_mask
    print(f"  Original test size: {test_mask.sum().item()}")
    print(f"  Clean test size:    {clean_test_mask.sum().item()}")
    print(f"  Removed: {(test_mask & zero_label_mask).sum().item()} fake nodes")

    for name, mdl, x_in in [
        ("IGNN-orig", model_ignn, X_d),
        ("GCN-orig", model_gcn, X_d),
        ("IGNN-rownorm", model_ignn_norm, X_norm_d),
        ("GCN-rownorm", model_gcn_norm, X_norm_d),
        ("IGNN-noSN", model_ignn_nosn, X_norm_d),
    ]:
        mdl.eval()
        with torch.no_grad():
            if isinstance(mdl, IGNN):
                logits, _, _ = mdl(x_in, A_d)
            else:
                logits = mdl(x_in, A_d)
            pred = logits.argmax(dim=1)
            dirty_acc = (pred[test_mask] == y_d[test_mask]).float().mean().item()
            clean_acc = (pred[clean_test_mask] == y_d[clean_test_mask]).float().mean().item()
        print(f"  {name:15s} | dirty={dirty_acc:.3f} | clean={clean_acc:.3f} | delta={clean_acc - dirty_acc:+.3f}")

    # ── EXPERIMENT 7: IGNN longer training ──
    print("=" * 60)
    print("EXPERIMENT 7: IGNN row-norm, 500 epochs, lr=0.005")
    model_ignn7 = IGNN(data["n_features"], hidden=64, n_classes=data["n_classes"]).to(device)
    val7, test7 = train_model(model_ignn7, X_norm_d, A_d, y_d, train_mask, val_mask, test_mask,
                              epochs=500, lr=0.005, label="IGNN-500ep")

    # ── SUMMARY ──
    print("=" * 60)
    print("SUMMARY TABLE")
    print(f"  {'Model':<25} {'Val':>6} {'Test':>6}")
    print(f"  {'-'*40}")
    for name, v, t in [
        ("IGNN-orig-50ep", val1, test1),
        ("GCN-orig-200ep", val2, test2),
        ("IGNN-rownorm-200ep", val3, test3),
        ("GCN-rownorm-200ep", val4, test4),
        ("IGNN-noSN-rownorm-200ep", val5, test5),
        ("IGNN-rownorm-500ep", val7, test7),
    ]:
        print(f"  {name:<25} {v:>6.3f} {t:>6.3f}")


if __name__ == "__main__":
    main()
