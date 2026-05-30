"""Deep dive into Citeseer data loader bug: val-test accuracy gap."""
from __future__ import annotations
import pickle
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F_func

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from iem.examples.ignn_cora import IGNN

PLANETOID_URL = "https://github.com/kimiyoung/planetoid/raw/master/data/"


def _pkl(data_dir, fname):
    with open(data_dir / fname, "rb") as f:
        return pickle.load(f, encoding="latin1")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path("datasets/citeseer")

    # Load raw components
    allx = _pkl(data_dir, "ind.citeseer.allx")
    tx = _pkl(data_dir, "ind.citeseer.tx")
    ally = _pkl(data_dir, "ind.citeseer.ally")
    ty = _pkl(data_dir, "ind.citeseer.ty")
    graph = _pkl(data_dir, "ind.citeseer.graph")

    test_idx = []
    with open(data_dir / "ind.citeseer.test.index") as f:
        for line in f:
            test_idx.append(int(line.strip()))
    test_idx = np.array(test_idx)
    test_idx_sorted = np.sort(test_idx)

    print("=== RAW DATA INSPECTION ===")
    print(f"allx shape: {allx.shape}")
    print(f"tx shape:   {tx.shape}")
    print(f"ally shape: {ally.shape}")
    print(f"ty shape:   {ty.shape}")
    print(f"test_idx: len={len(test_idx)}, min={test_idx.min()}, max={test_idx.max()}")
    print(f"test_idx_sorted: min={test_idx_sorted.min()}, max={test_idx_sorted.max()}")
    print(f"allx+tx rows: {allx.shape[0] + tx.shape[0]}")
    print(f"Expected contiguous test range: [{allx.shape[0]}, {allx.shape[0] + tx.shape[0] - 1}]")

    # KEY INSIGHT: Citeseer has non-contiguous test indices
    # Some indices in test_idx are BEYOND allx.shape[0] + tx.shape[0] - 1
    contiguous_range = set(range(allx.shape[0], allx.shape[0] + tx.shape[0]))
    test_set = set(test_idx.tolist())
    missing_from_contiguous = contiguous_range - test_set
    extra_beyond = test_set - contiguous_range
    print(f"\nContiguous range [{allx.shape[0]}, {allx.shape[0]+tx.shape[0]-1}]: {len(contiguous_range)} indices")
    print(f"Actual test indices in that range: {len(test_set & contiguous_range)}")
    print(f"Missing indices (in contiguous but not in test_idx): {len(missing_from_contiguous)}")
    if missing_from_contiguous:
        missing_sorted = sorted(missing_from_contiguous)
        print(f"  Missing indices: {missing_sorted}")
    print(f"Extra indices (in test_idx but beyond contiguous): {len(extra_beyond)}")
    if extra_beyond:
        print(f"  Extra indices: {sorted(extra_beyond)}")

    # Now trace through the loader logic step by step
    print("\n=== TRACING THE REORDER BUG ===")
    features = sp.vstack([allx, tx]).tolil()
    print(f"features after vstack: {features.shape}")
    max_idx = max(test_idx.max(), features.shape[0] - 1)
    print(f"max_idx: {max_idx}")
    if max_idx >= features.shape[0]:
        n_extend = max_idx + 1 - features.shape[0]
        print(f"Extending features by {n_extend} rows (zero-padded)")
        features.resize((max_idx + 1, features.shape[1]))

    # The reorder step: features[test_idx] = features[test_idx_sorted]
    # This is supposed to undo the shuffling in test indices.
    # But for Citeseer, test_idx has GAPS -- some indices in the
    # contiguous range [allx_size, allx_size+tx_size) are NOT in test_idx.
    # These "missing" indices get actual tx data but are NEVER reordered.
    # The 15 "extra" indices beyond the contiguous range start as zeros
    # and get assigned features from the sorted positions.

    # Let's check: are the missing indices in train/val/test?
    print("\n=== MISSING INDEX ANALYSIS ===")
    n_classes = 6
    n_train = 20 * n_classes  # 120
    n_val = 500
    N = max_idx + 1

    train_indices = set(range(n_train))
    val_indices = set(range(n_train, n_train + n_val))

    missing_in_train = missing_from_contiguous & train_indices
    missing_in_val = missing_from_contiguous & val_indices
    missing_in_test = missing_from_contiguous & test_set
    missing_in_none = missing_from_contiguous - train_indices - val_indices - test_set
    print(f"Missing indices in train: {len(missing_in_train)}")
    print(f"Missing indices in val:   {len(missing_in_val)}")
    print(f"Missing indices in test:  {len(missing_in_test)}")
    print(f"Missing indices unlabeled: {len(missing_in_none)}")

    # Check what labels the REORDERED data has for test nodes
    labels_raw = np.vstack([ally, ty])
    print(f"\nlabels_raw shape: {labels_raw.shape}")
    if max_idx >= labels_raw.shape[0]:
        n_ext = max_idx + 1 - labels_raw.shape[0]
        labels_raw = np.vstack([labels_raw, np.zeros((n_ext, labels_raw.shape[1]))])
        print(f"Extended labels to: {labels_raw.shape}")

    # BEFORE reorder: what class does each test index have?
    labels_before = labels_raw.copy()
    test_labels_before = labels_before[test_idx]
    zero_rows_before = (test_labels_before.sum(axis=1) == 0)
    print(f"\nBefore reorder: test nodes with zero-label rows: {zero_rows_before.sum()}")

    # AFTER reorder
    labels_raw[test_idx] = labels_raw[test_idx_sorted]
    test_labels_after = labels_raw[test_idx]
    zero_rows_after = (test_labels_after.sum(axis=1) == 0)
    print(f"After reorder:  test nodes with zero-label rows: {zero_rows_after.sum()}")

    # The ty labels -- are they already in the right order?
    print(f"\nty label distribution: {ty.sum(axis=0).tolist()}")
    print(f"ally label distribution: {ally.sum(axis=0).tolist()}")

    # CRITICAL CHECK: Compare with PyG's loader
    # In the standard Planetoid loader (Kipf's GCN repo, PyG, DGL),
    # the test indices should map to ty labels. Let's verify:
    print("\n=== CORRECT LABEL ASSIGNMENT CHECK ===")
    # The correct approach: ty[i] is the label for test_idx_sorted[i]
    # After reorder: labels[test_idx[i]] should equal ty[sort_perm[i]]
    # where sort_perm maps sorted positions back to original positions
    # Actually the standard approach is:
    # 1. labels = vstack(ally, ty) -- ty rows are at [allx_size, allx_size+tx_size)
    # 2. labels[test_idx] = labels[test_idx_sorted]
    # This means: the label at position test_idx[i] gets the label that WAS at test_idx_sorted[i]
    # test_idx_sorted[i] is in range [allx_size, allx_size+tx_size) for Cora (all contiguous)
    # For Citeseer, test_idx_sorted has GAPS, so some positions are zero-filled

    # Let's see which test_idx_sorted values are beyond the original labels
    orig_label_size = ally.shape[0] + ty.shape[0]
    beyond = test_idx_sorted >= orig_label_size
    print(f"test_idx_sorted values beyond original label matrix: {beyond.sum()}")
    print(f"These indices: {test_idx_sorted[beyond].tolist()}")

    # These nodes had zero labels in the extended matrix, then got mapped back
    # to arbitrary test_idx positions. This is the corruption source.

    # VERIFY: What does PyG do differently?
    # PyG handles this by only evaluating test nodes that have valid labels.
    # Or by filling in label -1 for unknown nodes and masking them out.

    print("\n=== FIX: USE -1 FOR UNKNOWN LABELS ===")
    # Reload clean
    labels_clean = np.vstack([ally, ty])
    if max_idx >= labels_clean.shape[0]:
        n_ext = max_idx + 1 - labels_clean.shape[0]
        # Instead of zeros, fill with -1 indicator
        filler = np.full((n_ext, labels_clean.shape[1]), -1.0)
        labels_clean = np.vstack([labels_clean, filler])
    labels_clean[test_idx] = labels_clean[test_idx_sorted]

    # Now identify which test nodes got -1 filler
    filler_test = (labels_clean[test_idx].min(axis=1) < 0)
    print(f"Test nodes that got filler labels: {filler_test.sum()}")

    # For valid test nodes, compute labels normally
    valid_test = ~filler_test
    print(f"Valid test nodes: {valid_test.sum()} / {len(test_idx)}")

    # Now train IGNN and GCN on FIXED data and compare
    print("\n=== FIXED DATA: IGNN + GCN EXPERIMENTS ===")

    # Rebuild proper labels
    labels_fixed = np.vstack([_pkl(data_dir, "ind.citeseer.ally"),
                              _pkl(data_dir, "ind.citeseer.ty")])
    if max_idx >= labels_fixed.shape[0]:
        n_ext = max_idx + 1 - labels_fixed.shape[0]
        labels_fixed = np.vstack([labels_fixed, np.zeros((n_ext, labels_fixed.shape[1]))])
    labels_fixed[test_idx] = labels_fixed[test_idx_sorted]
    y_fixed = torch.tensor(labels_fixed.argmax(axis=1), dtype=torch.long)

    # Build features (with row normalization)
    features2 = sp.vstack([_pkl(data_dir, "ind.citeseer.allx"),
                           _pkl(data_dir, "ind.citeseer.tx")]).tolil()
    if max_idx >= features2.shape[0]:
        features2.resize((max_idx + 1, features2.shape[1]))
    features2[test_idx] = features2[test_idx_sorted]
    X = torch.tensor(features2.toarray(), dtype=torch.float32)
    # Row normalize
    rnorm = X.norm(dim=1, keepdim=True)
    rnorm[rnorm == 0] = 1.0
    X = X / rnorm

    # Adjacency
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
    A_hat = torch.tensor(A_hat.toarray(), dtype=torch.float32)

    # Masks
    train_mask = torch.zeros(N, dtype=torch.bool)
    train_mask[:120] = True
    val_mask = torch.zeros(N, dtype=torch.bool)
    val_mask[120:620] = True
    # FIXED test mask: exclude nodes with zero-label (unknown class)
    zero_label = (labels_fixed.sum(axis=1) == 0)
    test_mask_dirty = torch.zeros(N, dtype=torch.bool)
    test_mask_dirty[test_idx] = True
    test_mask_clean = test_mask_dirty & ~torch.tensor(zero_label, dtype=torch.bool)

    print(f"Train: {train_mask.sum().item()}, Val: {val_mask.sum().item()}")
    print(f"Test (dirty): {test_mask_dirty.sum().item()}, Test (clean): {test_mask_clean.sum().item()}")

    # Label distribution in clean test
    clean_counts = torch.bincount(y_fixed[test_mask_clean], minlength=6)
    print(f"Clean test label dist: {clean_counts.tolist()}")

    X_d = X.to(device)
    A_d = A_hat.to(device)
    y_d = y_fixed.to(device)

    # Train IGNN (200 epochs, row-norm features, clean test mask)
    print("\n--- IGNN (fixed, 200ep) ---")
    model = IGNN(X.shape[1], hidden=64, n_classes=6).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for ep in range(1, 201):
        model.train()
        logits, _, _ = model(X_d, A_d)
        loss = F_func.cross_entropy(logits[train_mask], y_d[train_mask])
        optim.zero_grad()
        loss.backward()
        optim.step()
        if ep % 50 == 0:
            model.eval()
            with torch.no_grad():
                logits, _, _ = model(X_d, A_d)
                pred = logits.argmax(dim=1)
                val_acc = (pred[val_mask] == y_d[val_mask]).float().mean().item()
                test_dirty = (pred[test_mask_dirty] == y_d[test_mask_dirty]).float().mean().item()
                test_clean = (pred[test_mask_clean] == y_d[test_mask_clean]).float().mean().item()
            print(f"  ep {ep:3d} | loss {loss.item():.4f} | val {val_acc:.3f} | test_dirty {test_dirty:.3f} | test_clean {test_clean:.3f}")

    # Train simple GCN
    class SimpleGCN(nn.Module):
        def __init__(self, nf, h, nc):
            super().__init__()
            self.fc1 = nn.Linear(nf, h)
            self.fc2 = nn.Linear(h, nc)
        def forward(self, X, A):
            H = F_func.relu(A @ self.fc1(X))
            H = F_func.dropout(H, p=0.5, training=self.training)
            return A @ self.fc2(H)

    print("\n--- GCN (fixed, 200ep) ---")
    gcn = SimpleGCN(X.shape[1], 64, 6).to(device)
    optim = torch.optim.Adam(gcn.parameters(), lr=0.01, weight_decay=5e-4)
    for ep in range(1, 201):
        gcn.train()
        logits = gcn(X_d, A_d)
        loss = F_func.cross_entropy(logits[train_mask], y_d[train_mask])
        optim.zero_grad()
        loss.backward()
        optim.step()
        if ep % 50 == 0:
            gcn.eval()
            with torch.no_grad():
                logits = gcn(X_d, A_d)
                pred = logits.argmax(dim=1)
                val_acc = (pred[val_mask] == y_d[val_mask]).float().mean().item()
                test_dirty = (pred[test_mask_dirty] == y_d[test_mask_dirty]).float().mean().item()
                test_clean = (pred[test_mask_clean] == y_d[test_mask_clean]).float().mean().item()
            print(f"  ep {ep:3d} | loss {loss.item():.4f} | val {val_acc:.3f} | test_dirty {test_dirty:.3f} | test_clean {test_clean:.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
