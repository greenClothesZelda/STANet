import logging
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

log = logging.getLogger(__name__)


def _collect_labels(dataset, name):
    log.info(f"Scanning {name} for labels...")
    label_counts = Counter()

    # Iterate through dataset to collect labels
    # Using tqdm for progress tracking as this might take time
    for i in tqdm(range(len(dataset)), desc=f"Scanning {name}"):
        item = dataset[i]

        # Attempt to extract labels based on common formats
        lbl = None
        if isinstance(item, dict):
            if 'labels' in item:
                lbl = item['labels']
            elif 'label' in item:
                lbl = item['label']
        elif isinstance(item, (tuple, list)):
            lbl = item[-1]  # Assume last item is label

        if lbl is not None:
            if isinstance(lbl, torch.Tensor):
                lbl = lbl.numpy()
            # Flatten in case of sequence labels and update counter
            label_counts.update(np.array(lbl).flatten().astype(int))
    return label_counts


def visualize_label_distribution_comparison(train_dataset, test_dataset, output_dir):
    """
    Visualizes the label distribution of train and test sets on a single graph.
    Uses relative frequency (ratio) on the y-axis.
    """
    train_counts = _collect_labels(train_dataset, "Train Set")
    test_counts = _collect_labels(test_dataset, "Test Set")

    if not train_counts or not test_counts:
        log.warning("No labels found. Skipping visualization.")
        return

    def get_ratios(counts):
        total = sum(counts.values())
        return {k: v / total for k, v in counts.items()} if total > 0 else {}

    train_ratios = get_ratios(train_counts)
    test_ratios = get_ratios(test_counts)

    all_labels = sorted(set(train_ratios.keys()) | set(test_ratios.keys()))
    train_vals = [train_ratios.get(l, 0) for l in all_labels]
    test_vals = [test_ratios.get(l, 0) for l in all_labels]

    plt.figure(figsize=(12, 6))

    width = 0.35
    indices = np.array(all_labels)

    plt.bar(indices - width/2, train_vals, width=width,
            label='Train', color='blue', alpha=0.7)
    plt.bar(indices + width/2, test_vals, width=width,
            label='Test', color='red', alpha=0.7)

    plt.xlabel('Label Value (Demand)')
    plt.ylabel('Relative Frequency')
    plt.title('Label Distribution Comparison (Train vs Test)')
    plt.legend()
    plt.yscale('log')  # Log scale is often useful for sparse demand data
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    save_path = Path(output_dir) / "label_distribution_comparison.png"
    plt.savefig(save_path)
    plt.close()
    log.info(f"Saved label distribution comparison to {save_path}")
