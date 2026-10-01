"""
Group-and-plot Schaefer-400 FC matrices by network membership, mirroring the
layout in "Filmfest FC matrices.pdf" (parcel-level heatmap with network grid
lines).

This is purely a plotting add-on: it reuses the FC matrices already computed
by compute_fc_matrix.py and does not touch that script or its plots. Run
compute_fc_matrix.py first if scratch/fc_matrices/*_fc400.npy is missing.

Network group for each of the 400 parcels is derived from the parcel names in
Schaefer2018_400Parcels_17Networks_order.txt (the same 17-network atlas used
by extract_parcel_timecourses.py), by matching the standard Yeo 7-network
name substrings (e.g. "ContA_Cingm" -> "Cont", "VisCent_ExStr" -> "Vis") plus
"TempPar", the one 17-network label that doesn't nest under any of the 7 by
name -- kept here as its own 8th group rather than folded into another.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from confounds import VARIANT_TAGS
from layout import fc_bynetwork_png, fc_matrix, iter_runs

NPARCEL = 400
ORDER_TXT = Path("standard/Schaefer2018_400Parcels_17Networks_order.txt")

# Canonical Yeo-7 order (matches the reference figure's axis order), plus
# TempPar as its own 8th group after Default -- it's a real, distinct 17th
# network in this atlas and doesn't literally belong to any of the 7 by name.
NETWORK_ORDER = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default", "TempPar"]


def load_network_labels(order_txt=ORDER_TXT):
    """Return a length-400 list of network group names (NETWORK_ORDER), in the
    same row order as the combined parcel matrices (LH parcels 1-200, then RH
    parcels 1-200)."""
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
    """Row indices that group parcels by NETWORK_ORDER, stable within each network
    (so LH and RH parcels of the same network end up in one contiguous block,
    LH first then RH, matching their original relative order)."""
    return sorted(range(len(labels)), key=lambda i: NETWORK_ORDER.index(labels[i]))


def network_block_centers(sorted_labels):
    """Boundary positions (for axvline/axhline) between consecutive network
    blocks, plus each block's center (for tick placement), in a
    network-sorted label list."""
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

    for r in iter_runs():
        for variant_tag in VARIANT_TAGS:
            fc_path = fc_matrix(r, variant_tag)
            if not fc_path.exists():
                print(f"[{r}/{variant_tag}] missing {fc_path}, skipping (run compute_fc_matrix.py first)")
                continue
            fc = np.load(fc_path)

            out = fc_bynetwork_png(r, variant_tag)
            out.parent.mkdir(parents=True, exist_ok=True)
            plot_fc_by_network(fc, labels, f"{r} FC ({variant_tag}, by network)", out)
            print(f"[{r}/{variant_tag}] saved {out}")

    print("\ndone")


if __name__ == "__main__":
    main()
