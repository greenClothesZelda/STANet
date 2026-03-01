# STANet PDF Naming Map

This document maps symbols in `docs/model.pdf` to canonical code names used in this repository.
The PDF is the source of truth for naming.

## Feature Symbols

| PDF symbol | Meaning | Canonical code name | Legacy alias |
|---|---|---|---|
| `u_r^{stat}` | Static region embedding | `StaticRegionEncoder` output | `RegeonEncoder` output |
| `u_t^{time}` | Temporal context embedding | `TemporalContextEncoder` output | `TemporalEncoder` output |
| `u_{r,t}^{dyn}` | Dynamic demand embedding | `DynamicDemandEncoder` output | `LocalLagEncoder` output |
| `y_{r,t}^{(l)}` | Local lag demand window | `demand_features.y_lag` | `demand_features.demand_series` |
| `m_{r,t,k}` | Missing/valid indicator | `demand_features.m_lag` | `demand_features.valid_mask` |
| `Delta t_{r,t}^{last}` | Global recency descriptor | `demand_features.delta_t_last` | `demand_features.deactivation_period` |
| `dow(t)` | Day of week index | `temporal_features.dow` | `temporal_features.day_of_week` |
| `hod(t)` | Hour of day index | `temporal_features.hod` | `temporal_features.hour_of_day` |
| `holiday(t)` | Holiday indicator | `temporal_features.holiday` | `temporal_features.is_holiday` |

## Module Symbols

| PDF concept | Canonical code module | Legacy alias |
|---|---|---|
| Static region encoder `f_stat` | `StaticRegionEncoder` | `RegeonEncoder` |
| Temporal context encoder | `TemporalContextEncoder` | `TemporalEncoder` |
| Dynamic demand encoder `f_dyn` | `DynamicDemandEncoder` | `LocalLagEncoder` |
| Snapshot global attention | `SnapshotGlobalAttention` | `SnapshotGlobalAttn` |
| Temporal state update (GRU + gate) | `TemporalStateUpdater` | `TemporalStateModule` |
| Temporal window aggregation | `TemporalWindowAggregator` | `TemporalAggregationModule` |

## Output Symbols

| PDF symbol | Meaning | Canonical key | Legacy alias |
|---|---|---|---|
| `p_{r,t+1}` | Event probability | `p_event` | `event_prob` |
| `hat{y}_{r,t+1}^{+}` | Conditional magnitude | `y_hat_pos` | `magnitude` |
| `hat{y}_{r,t+1}` | Final prediction | `y_hat` | `prediction` |
