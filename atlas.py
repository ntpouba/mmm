from __future__ import annotations

from layout import ROOT

STANDARD_DIR = ROOT / "standard"
ORDER_TXT = STANDARD_DIR / "Schaefer2018_400Parcels_17Networks_order.txt"

NPARCEL = 400

# Canonical Yeo-7 order (matches the reference figure's axis order), plus
# TempPar as its own 8th group after Default -- it's a real, distinct 17th
# network in this atlas and doesn't literally belong to any of the 7 by name.
NETWORK_ORDER = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default", "TempPar"]


def load_network_labels() -> list[str]:
    """Return a length-400 list of network group names (NETWORK_ORDER), in the
    same row order as the combined parcel matrices (LH parcels 1-200, then RH
    parcels 1-200)."""
    labels = []
    for line in ORDER_TXT.read_text().splitlines():
        _, name = line.split("\t")[:2]
        match = next((net for net in NETWORK_ORDER if net in name), None)
        if match is None:
            raise ValueError(f"Could not map parcel name to a network group: {name!r}")
        labels.append(match)
    if len(labels) != NPARCEL:
        raise ValueError(f"Expected {NPARCEL} parcel labels, got {len(labels)}")
    return labels


def network_sort_order(labels: list[str]) -> list[int]:
    """Row indices that group parcels by NETWORK_ORDER, stable within each network
    (so LH and RH parcels of the same network end up in one contiguous block,
    LH first then RH, matching their original relative order)."""
    return sorted(range(len(labels)), key=lambda i: NETWORK_ORDER.index(labels[i]))


def network_block_centers(sorted_labels: list[str]) -> tuple[list[float], list[float]]:
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
