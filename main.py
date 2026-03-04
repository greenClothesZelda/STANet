import logging
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
import torch
from torch.utils.data import Subset
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

from dataset_frame.sta_dataset import STADataset, stad_collate_fn
from models import STANet, STANetForTrainer
from runners import test_loop, visualize_label_distribution_comparison
from models.attn import get_attn_module

log = logging.getLogger(__name__)
results = []


def set_seed(seed):
    import numpy as np
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class EarlyStoppingWithMinEpochs(EarlyStoppingCallback):
    def __init__(self, min_epochs=5, early_stopping_patience=3, early_stopping_threshold=0.0):
        super().__init__(early_stopping_patience=early_stopping_patience,
                         early_stopping_threshold=early_stopping_threshold)
        self.min_epochs = min_epochs

    def on_evaluate(self, args, state, control, **kwargs):
        if state.epoch is not None and state.epoch < self.min_epochs:
            log.info(
                f"Skipping early stopping check at epoch {state.epoch} (min_epochs={self.min_epochs})")
            return control
        return super().on_evaluate(args, state, control, **kwargs)


def _model_block(model_cfg, primary_key, legacy_key=None, default=None):
    if primary_key in model_cfg:
        return dict(model_cfg[primary_key])
    if legacy_key is not None and legacy_key in model_cfg:
        log.warning(f"Using legacy model config key '{legacy_key}'. Prefer '{primary_key}'.")
        return dict(model_cfg[legacy_key])
    if default is not None:
        return dict(default)
    raise KeyError(f"Missing required model config block: '{primary_key}'")


def build_model(config, dataset, device):
    x_geo = torch.cat(
        [dataset.coordinates, dataset.areas.unsqueeze(1).log1p()], dim=-1).to(device)
    poi_conf = _model_block(config.model, "POIEncoder")
    static_region_conf = _model_block(config.model, "StaticRegionEncoder", "RegeonEncoder")
    dynamic_demand_conf = _model_block(config.model, "DynamicDemandEncoder", "LocalLagEncoder")
    temporal_context_conf = _model_block(config.model, "TemporalContextEncoder", "TemporalEncoder")
    temporal_state_conf = _model_block(config.model, "TemporalStateUpdater", "TemporalStateModule")
    snapshot_attn_conf = _model_block(
        config.model, "SnapshotGlobalAttention", "SnapshotGlobalAttn")
    temporal_window_conf = _model_block(
        config.model, "TemporalWindowAggregator", "TemporalAggregationModule", default={})

    # Get attention module from config
    attn_name = config.model.attention.name
    attn_module = get_attn_module(attn_name)
    attn_configs = dict(config.model.attention.get('configs', {}))

    # override POI category count with data-driven value
    poi_conf['num_poi_categories'] = dataset.poi.shape[1] if hasattr(
        dataset, 'poi') else poi_conf.get('num_poi_categories', 0)
    temporal_context_conf['weather_dim'] = getattr(
        dataset, 'weather_dim', temporal_context_conf.get('weather_dim', 0)
    )
    poi_conf['temporal_dim'] = temporal_context_conf.get(
        'embedding_dim', poi_conf.get('temporal_dim', 16))
    stanet = STANet(
        embedding_dim=config.model.embedding_dim,
        POIEncoder_configs={
            **poi_conf,
            'x_poi': dataset.poi.to(device),
        },
        StaticRegionEncoder_configs={
            **static_region_conf,
            'land_composition': dataset.composition.to(device),
            'x_geo': x_geo,
        },
        TemporalContextEncoder_configs={**temporal_context_conf},
        DynamicDemandEncoder_configs={
            **dynamic_demand_conf,
            'lag_window': config.dataset.time_step,
        },
        SnapshotGlobalAttention_configs={**snapshot_attn_conf},
        TemporalWindowAggregator_configs={**temporal_window_conf},
        TemporalStateUpdater_configs={**temporal_state_conf},
        PTReLU_configs=dict(config.model.PTReLU),
        attn_module=attn_module,
        attn_configs=attn_configs,
    ).to(device)

    return STANetForTrainer(stanet, **config.loss)


def split_dataset_sequential(dataset, train_ratio):
    total_len = len(dataset)
    train_len = int(total_len * train_ratio)
    train_indices = list(range(0, train_len))
    val_indices = list(range(train_len, total_len))
    log.info(
        f"total={len(dataset)} Dataset split: train size={len(train_indices)}, val size={len(val_indices)}")
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def run(config):
    set_seed(config.seed)
    device = torch.device(config.device)

    orig_cwd = Path(get_original_cwd())
    data_root = orig_cwd / "data" / "raw"

    dataset = STADataset(
        file_name=config.dataset.file_name,
        time_step=config.dataset.time_step,
        root=data_root,
        use_weather=config.dataset.get("use_weather", False),
        weather_file=config.dataset.get("weather_file", "meteorological_data.csv"),
        weather_encoding=config.dataset.get("weather_encoding", "cp949"),
        weather_target_columns=config.dataset.get(
            "weather_target_columns",
            ["강수량(mm)", "기온(°C)", "습도(%)", "적설(cm)"],
        ),
    )
    train_ds, val_ds = split_dataset_sequential(
        dataset, config.dataset.train_ratio)

    output_dir = HydraConfig.get().runtime.output_dir
    visualize_label_distribution_comparison(train_ds, val_ds, output_dir)

    model = build_model(config, dataset, device)

    training_kwargs = {**dict(config.train)}
    if 'eval_strategy' in training_kwargs:
        training_kwargs['eval_strategy'] = training_kwargs.pop(
            'eval_strategy')
    training_kwargs['output_dir'] = output_dir
    training_kwargs['seed'] = config.seed
    training_kwargs['report_to'] = []
    training_kwargs['log_level'] = 'info'
    training_args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=stad_collate_fn,
        callbacks=[EarlyStoppingWithMinEpochs(
            **dict(config.callbacks.early_stopping))],
    )

    trainer.train()
    metrics = trainer.evaluate()
    log.info(metrics)

    result = test_loop(trainer, val_ds, output_dir)
    results.append(result)


if __name__ == "__main__":
    run()
    log.info(f"All runs finished. Results: {results}")
