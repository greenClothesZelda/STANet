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
from runners.test import test_loop

log = logging.getLogger(__name__)


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


def build_model(config, dataset, device):
    num_nodes = dataset.num_nodes
    x_geo = torch.cat(
        [dataset.coordinates, dataset.areas.unsqueeze(1).log1p()], dim=-1).to(device)
    poi_conf = dict(config.model.POIEncoder)
    reg_conf = dict(config.model.RegeonEncoder)
    dyn_conf = dict(config.model.DynamicEncoder)
    temp_conf = dict(config.model.TemporalEncoder)
    tstate_conf = dict(config.model.TemporalStateModule)
    attn_conf = dict(config.model.SnapshotGlobalAttn)

    stanet = STANet(
        embedding_dim=config.model.embedding_dim,
        POIEncoder_configs={
            **poi_conf,
            'x_poi': torch.zeros((num_nodes, poi_conf['num_poi_categories']), device=device),
        },
        RegeonEncoder_configs={
            **reg_conf,
            'land_composition': dataset.composition.to(device),
            'x_geo': x_geo,
        },
        TemporalEncoder_configs={**temp_conf},
        DynamicEncoder_configs={
            **dyn_conf,
            'time_step': config.dataset.time_step,
        },
        SnapshotGlobalAttn_configs={**attn_conf},
        TemporalAggregationModule_configs={},
        TemporalStateModule_configs={**tstate_conf},
    ).to(device)

    return STANetForTrainer(stanet, lambda_mag=config.loss.lambda_mag)


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
    )
    train_ds, val_ds = split_dataset_sequential(
        dataset, config.dataset.train_ratio)

    model = build_model(config, dataset, device)

    output_dir = HydraConfig.get().runtime.output_dir
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

    test_loop(trainer, val_ds, output_dir)


if __name__ == "__main__":
    run()
