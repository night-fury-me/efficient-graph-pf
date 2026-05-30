# Revision R6 — safety-critical motivation, bulletproof route (IN PROGRESS)

**Concern (R3 W4):** abstract/intro invoke drug-interaction screening and fraud detection as safety-critical motivation, but the evaluated graphs are citation / co-purchase / IEEE power — neither drug nor fraud is tested.

**Chosen route: bulletproof (add a dataset).** Add a real **fraud-detection** node-classification graph to the continuous-to-discrete transfer experiment, so the "fraudulent accounts via perturbed edges" motivation is actually evaluated. (Drug interaction = molecular *graph* classification, which does not fit the node-classification + N-1-edge-removal + dense-`A_hat` pipeline, so drug stays a motivating example.)

**Dataset:** CARE-GNN **Amazon review-fraud** graph (`Amazon.mat`): 11,944 user nodes, 25 features, 2 classes (11,123 benign / 821 fraud), prebuilt symmetric `homo` adjacency. Public (GitHub), `scipy.io.loadmat`-able, **no PyG/DGL needed** (they aren't installed). Downloaded to `datasets/amazon_fraud/`. Smaller than the existing Pubmed (19,717) which already runs → no size cap.

**Implementation (verified):**
- New loader `iem/examples/ignn_amazon_fraud.py::_load_amazon_fraud` — byte-for-byte same recipe as `_load_amazon` (`adj+adjᵀ` → `+I` → `D^{-1/2}·D^{-1/2}`, confirmed against ignn_amazon.py L67–73), RandomState(42) 60/20/20, identical return-dict contract, full graph, alignment assert.
- Wired into `load_all_datasets()` + `DATASET_NAMES` in `exp_tau_all_datasets.py` + `DATASETS`/`DATASET_LABELS` (6th column) in `make_fig_tau_heatmap.py`.
- Load-test passed: N=11944, 25 feat, 2 classes, |E|=4.4M, A_hat symmetric, normalization spot-check OK, masks disjoint+cover.

**Run:** `.venv/bin/python -u scripts/exp_tau_all_datasets.py` (background `bw55pacv2`, monitor `bvlylfc21`), GPU ~45–60 min. Re-runs all 6 (first 5 reproduce deterministically; Amazon Fraud is new). Adds `tau`/`tau_weighted`/`tau_weight_only` for the fraud cells.

**Post-run plan:**
1. Analyse the 7 Amazon-Fraud cells (tau / tauW / tauWo) — does the weighted transfer hold on a fraud graph? Report honestly whatever it is.
2. Regenerate `fig_tau_heatmap.pdf` (now 6 columns).
3. Update transfer numbers (X/42 cells) + the `\Cref{fig:tau_heatmap}` prose.
4. Abstract/intro: state the framework is evaluated on a **fraud-detection** graph (closing R3 W4 for fraud); keep drug-interaction as a labeled motivating example.

**Caveat:** class imbalance ~6.9% fraud (irrelevant to the τ rank metric; matters only if accuracy is reported). GAT-2 on the full fraud graph may OOM (like Pubmed/WikiCS) → recorded as OOM.

## EXECUTED ✓ (2026-05-30; 10pp, 0 overfull, 0 undefined refs)

**Run complete** — 390 runs / 6 datasets. **Amazon Fraud cells (6 archs; GAT-2 OOM'd):** tauW **+0.981**, tauWo **+0.337** → **v_k lift +0.645** (max **+0.90** on IGNN, where weight-only is +0.088). Overall **39/39 cells positive** (weighted), median **+0.99**. Fraud is the **strongest control evidence**: on its dense, near-uniform-weight graph the edge weight is nearly useless and `v_k` carries the ranking — the sharpest rebuttal to "it's just the edge weight."

**Edits:** figure regenerated (6 columns, GAT-OOM footnote updated); abstract (39/39, Δτ up to +0.65 on a fraud graph), intro contributions (10 datasets / 5 domains / 390 runs), setup (added Amazon Fraud + `\cite{dou2020enhancing}`, 39 cells), transfer paragraph + caption (39/39, fraud control, GAT exception), Prop-transfer scoping retained. Bib entry `dou2020enhancing` added.

**Refit to 10pp:** trimmed redundant conclusion limitations (numbers retained in body) + setup baseline hyperparams + robust-backbone per-dataset numbers. No result removed.

**Drug-interaction:** stays a labeled motivating example (molecular = graph-classification, doesn't fit the node-classification + N-1 pipeline). Fraud now closes R3 W4 on the fraud side.

