
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

log = logging.getLogger(__name__)


@torch.no_grad()
def test_loop(trainer, test_dataset, output_dir):
    log.info("Start Testing with Trainer...")
    output = trainer.predict(test_dataset)

    preds = output.predictions
    if isinstance(preds, tuple):
        preds = preds[0]
    labels = output.label_ids

    preds = preds.reshape(labels.shape)

    num_nodes = getattr(test_dataset, 'num_nodes', None)
    if num_nodes is None and hasattr(test_dataset, 'dataset'):
        num_nodes = getattr(test_dataset.dataset, 'num_nodes', None)
    if num_nodes is None:
        num_nodes = preds.shape[1]

    dist = np.abs(preds - labels)
    mae = float(np.mean(dist))
    mape = float(np.mean(dist / (labels + 1)) * 100)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    result_df = pd.DataFrame({
        'predictions': preds.flatten(),
        'labels': labels.flatten()
    })
    result_csv_path = Path(output_dir) / 'test_results.csv'
    result_df.to_csv(result_csv_path, index=False)

    visualize_predictions(result_csv_path, num_nodes, output_dir)

    log.info(f"Test Finished. MAE: {mae:.4f}, MAPE: {mape:.4f}")
    return {'MAE': mae, 'MAPE': mape}


def visualize_predictions(csv_path, num_nodes, output_dir):
    df = pd.read_csv(csv_path)
    pred = np.array(df['predictions'].values)
    labels = np.array(df['labels'].values)

    pred = pred.reshape(-1, num_nodes)
    labels = labels.reshape(-1, num_nodes)

    diff = np.abs(labels - pred)
    demand_sum = np.sum(labels, axis=1) / num_nodes
    mean = np.mean(diff, axis=1)

    plt.figure(figsize=(24, 16))
    plt.plot(demand_sum, label='Average Demand', color='black')
    plt.plot(mean, label='Mean of Absolute Error', color='red')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title('Demand and Prediction Error Over Time')
    plt.legend()
    plt.savefig(Path(output_dir) / "demand_error_analysis.png")
    plt.close()

    sum_demands = np.sum(labels, axis=0)
    max_demand_node = np.argmax(sum_demands)
    min_demand_node = np.argmin(sum_demands)
    mid_demand_node = np.argsort(sum_demands)[num_nodes // 2]

    visualize_sample(pred[:, max_demand_node], labels[:,
                     max_demand_node], output_dir, name='max_demand_node')
    visualize_sample(pred[:, min_demand_node], labels[:,
                     min_demand_node], output_dir, name='min_demand_node')
    visualize_sample(pred[:, mid_demand_node], labels[:,
                     mid_demand_node], output_dir, name='mid_demand_node')


def visualize_sample(pred, labels, output_dir, name):
    pred = np.array(pred)
    labels = np.array(labels)

    plt.figure(figsize=(24, 16))
    plt.plot(labels, label='Labels', color='black')
    plt.plot(pred, label='Predictions', color='red')
    plt.xlabel('Time Step')
    plt.ylabel('Demand')
    plt.title(f'Predictions vs Labels for Sample Node: {name}')
    plt.legend()
    plt.savefig(Path(output_dir) / f"predictions_{name}.png")
    plt.close()
