
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
        if len(preds) < 4:
            raise ValueError(f"Expected at least 4 prediction tensors, got {len(preds)}")
        logits, event_prob, magnitude, prediction = preds[:4]
    else:
        prediction = preds
        logits = prediction
        event_prob = np.zeros_like(prediction)
        magnitude = prediction

    labels = output.label_ids

    pred_values = prediction.reshape(labels.shape)

    num_nodes = getattr(test_dataset, 'num_nodes', None)
    if num_nodes is None and hasattr(test_dataset, 'dataset'):
        num_nodes = getattr(test_dataset.dataset, 'num_nodes', None)
    if num_nodes is None:
        num_nodes = pred_values.shape[1]

    dist = np.abs(pred_values - labels)
    mae = float(np.mean(dist))
    mape = float(np.mean(dist / (labels + 1)) * 100)
    zero_base_mape = float(np.mean(labels / (labels + 1)) * 100)
    nonzero_mask = labels > 0
    if np.any(nonzero_mask):
        mape_nonzero = float(
            np.mean(np.abs(pred_values[nonzero_mask] - labels[nonzero_mask]) / labels[nonzero_mask]) * 100
        )
    else:
        mape_nonzero = float("nan")

    rounded_pred = np.rint(np.clip(pred_values, a_min=0.0, a_max=None))
    zero_mask = labels == 0
    if np.any(zero_mask):
        zero_recall = float(np.mean(rounded_pred[zero_mask] == 0))
    else:
        zero_recall = float("nan")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    result_df = pd.DataFrame({
        'predictions': pred_values.flatten(),
        'labels': labels.flatten(),
        'event_prob': event_prob.flatten(),
        'magnitude': magnitude.flatten(),
    })
    result_csv_path = Path(output_dir) / 'test_results.csv'
    result_df.to_csv(result_csv_path, index=False)

    visualize_predictions(result_csv_path, num_nodes, output_dir)
    visualize_strip_chart(pred_values, labels, output_dir)

    log.info(
        f"Test Finished. MAE: {mae:.4f}, MAPE: {mape:.4f}, "
        f"MAPE_nonzero: {mape_nonzero:.4f}, ZeroRecall: {zero_recall:.4f}"
    )

    event_target = (labels > 0)
    event_acc = {}
    for thresh in [0.3, 0.5, 0.7, 0.9]:
        pred_event = (event_prob > thresh)
        correct = (pred_event == event_target).astype(float)
        acc = correct.mean()
        event_acc[f'acc@{thresh}'] = acc
    log.info(f"Event Acc: {event_acc}")
    return {
        'MAE': mae,
        'MAPE': mape,
        'MAPE_nonzero': mape_nonzero,
        'Zero_Recall': zero_recall,
        'Zero_Base_MAPE': zero_base_mape,
    }  # 'event_acc': event_acc}


def visualize_predictions(csv_path, num_nodes, output_dir):
    df = pd.read_csv(csv_path)
    pred = np.array(df['predictions'].values)
    labels = np.array(df['labels'].values)

    pred = pred.reshape(-1, num_nodes)
    labels = labels.reshape(-1, num_nodes)

    diff = np.abs(labels - pred)
    demand_sum = np.sum(labels, axis=1) / num_nodes
    mean = np.mean(diff, axis=1)

    plt.figure(figsize=(24, 5))
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

    plt.figure(figsize=(24, 5))
    plt.plot(labels, label='Labels', color='black')
    plt.plot(pred, label='Predictions', color='red')
    plt.xlabel('Time Step')
    plt.ylabel('Demand')
    plt.title(f'Predictions vs Labels for Sample Node: {name}')
    plt.legend()
    plt.savefig(Path(output_dir) / f"predictions_{name}.png")
    plt.close()


def visualize_strip_chart(preds, labels, output_dir):
    """
    Visualizes the distribution of predicted values for each ground truth label value.
    Creates a strip chart (scatter plot with jitter).
    """
    preds = preds.flatten()
    labels = labels.flatten()

    plt.figure(figsize=(12, 8))

    # Add random jitter to labels for better visualization of density
    jitter = np.random.uniform(-0.2, 0.2, size=labels.shape)

    plt.scatter(labels + jitter, preds, alpha=0.1, s=2, color='blue')

    plt.xlabel('Actual Demand (Label)')
    plt.ylabel('Predicted Demand')
    plt.title('Prediction Distribution per Label (Strip Chart)')
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.savefig(Path(output_dir) / "label_prediction_strip_chart.png")
    plt.close()
