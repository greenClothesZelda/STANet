# STANet Model Audit Report (PDF Baseline)

Baseline document: `docs/model.pdf`  
Scope: naming/interface alignment only. No architecture or loss rewrite was applied in this change set.

## Section-by-section Audit

| PDF section | Status | Notes | Related code |
|---|---|---|---|
| 1.1 Static region features | Partial | POI decomposition with learnable non-negative weights is implemented. Inputs are additionally normalized in dataset preprocessing. | `dataset_frame/sta_dataset.py`, `models/module/regeon_encoder.py` |
| 1.2 Temporal context embedding | Match | Day/hour/holiday embeddings are summed as described. | `models/module/temporal_encoder.py` |
| 1.3 Dynamic demand features | Partial | Uses lag values, valid mask, sparse mask, and clipped recency. Explicit scalar sparsity count `c_{r,t}` is not separately modeled. | `models/module/local_lag_encoder.py` |
| 2. Initial region-time embedding | Match | Concatenation and linear projection over static/temporal/dynamic embeddings is implemented. | `models/sta_net.py` |
| 3. Temporal state update | Match | GRU update followed by gated fusion between GRU output and input embedding is implemented. | `models/module/temporal_state_module.py` |
| 4. Snapshot global attention | Mismatch | Global attention is implemented, but OD-based soft bias term `g_OD(r,j,t)` is not injected. | `models/module/snapshot_global_attn.py` |
| 5. Temporal aggregation | Partial | Temporal attention-style weighted aggregation is implemented; internal sequence GRU is used before scoring. | `models/module/temporal_aggregation.py` |
| 6. Sparse-aware output heads | Partial | `p_event` and `y_hat_pos` heads exist. Final output uses `PTReLU(p_event) * y_hat_pos`, not exactly `p_event * y_hat_pos`. | `models/sta_net.py` |
| 7. Loss function | Mismatch | Event BCE + magnitude term exists, but formula differs from PDF (weighted blend and different magnitude term definition). | `models/train_model.py` |

## Explicitly Deferred Items

1. OD bias implementation inside snapshot attention.
2. Output head exact PDF form (`y_hat = p_event * y_hat_pos` without `PTReLU` transform).
3. Loss function rewrite to exact PDF objective and regularization term.
4. Dynamic descriptor rewrite to include explicit `c_{r,t}` term if strict PDF parity is required.

## Naming alignment delivered in this change set

See `docs/pdf_naming_map.md` for the canonical-to-legacy map and backward-compatible aliases.
