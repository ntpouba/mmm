"""
Volume-space equivalent of plot_fc_networks.py: groups/plots the volume-pipeline
Schaefer-400 FC matrices by network membership. Reuses the exact same
network-grouping logic (same atlas order file, same LH-1-200/RH-201-400
convention -- verified to match the volume atlas's own label ordering).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from confounds import VARIANT_TAGS  # noqa: E402

SUBJECTS = ["sub-03", "sub-04", "sub-05"]
RUNS = ["run-01", "run-02"]

NPARCEL = 400
FC_DIR = ROOT / "scratch/fc_matrices_volume"
OUT_DIR = ROOT / "scratch/fc_matrices_bynetwork_volume"
ORDER_TXT = ROOT / "standard/Schaefer2018_400Parcels_17Networks_order.txt"

NETWORK_ORDER = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default", "TempPar"]


def load_network_labels(order_txt=ORDER_TXT):
    labels = []
    for line in order_txt.read_text().splitlines():
        _, name = line.split("\t")[:2]
        match = next((net for net in NETWORK_ORDER if net in name), None)
        if match is None:
            raise ValueError(f"Could not map parcel name to a network group: {name!r}")
        labels.append(match)
    if len(labels) != NPARCEL:
        raise ValueError(f"Expected {NPARCEL} parcel labels, got {len(labels)}")
    return labels


def network_sort_order(labels):
    return sorted(range(len(labels)), key=lambda i: NETWORK_ORDER.index(labels[i]))


def network_block_centers(sorted_labels):
    boundaries, centers = [], []
    start = 0
    for i in range(1, len(sorted_labels) + 1):
        if i == len(sorted_labels) or sorted_labels[i] != sorted_labels[start]:
            centers.append((start + i - 1) / 2)
            if i != len(sorted_labels):
                boundaries.append(i - 0.5)
            start = i
    return boundaries, centers


def plot_fc_by_network(fc, labels, title, out_path):
    order = network_sort_order(labels)
    fc_sorted = fc[np.ix_(order, order)]
    sorted_labels = [labels[i] for i in order]
    boundaries, centers = network_block_centers(sorted_labels)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(fc_sorted, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    ax.set_title(title)
    for b in boundaries:
        ax.axvline(b, color="k", linewidth=0.6)
        ax.axhline(b, color="k", linewidth=0.6)
    ax.set_xticks(centers)
    ax.set_xticklabels(NETWORK_ORDER, rotation=45, ha="right")
    ax.set_yticks(centers)
    ax.set_yticklabels(NETWORK_ORDER)
    fig.colorbar(im, ax=ax, label="Pearson r")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    labels = load_network_labels()

    for subject in SUBJECTS:
        for run in RUNS:
            run_out_dir = OUT_DIR / subject / run
            run_out_dir.mkdir(parents=True, exist_ok=True)

            for variant_tag in VARIANT_TAGS:
                fc_path = FC_DIR / f"{subject}_{run}_{variant_tag}_fc400.npy"
                if not fc_path.exists():
                    print(f"[{subject}/{run}/{variant_tag}] missing {fc_path}, skipping (run compute_fc_matrix_volume.py first)")
                    continue
                fc = np.load(fc_path)

                out = run_out_dir / f"{variant_tag}_fc400_bynetwork.png"
                plot_fc_by_network(fc, labels, f"{subject} {run} FC ({variant_tag}, by network, volume)", out)
                print(f"[{subject}/{run}/{variant_tag}] saved {out}")

    print("\ndone")


if __name__ == "__main__":
    main()
