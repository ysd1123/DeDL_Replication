#!/usr/bin/env python3
"""Plot training loss curve from a log snippet.

Example:
  python plot_loss_curve.py \
    --file ReplicationCodes/output_dedl_synth_1/out.log \
    --start-line 473 --end-line 2000 \
    --out pdl_train_mse.png

This script extracts lines like:
  [PDL] Epoch 999/1000, train MSE=0.1604
and plots train MSE vs epoch.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt


EPOCH_MSE_RE = re.compile(
    r"^\[(?P<tag>[^\]]+)\]\s+Epoch\s+(?P<epoch>\d+)\s*/\s*(?P<total>\d+)\s*,\s*train\s+MSE\s*=\s*(?P<mse>[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*$"
)


def parse_epoch_mse(
    lines: List[str], tag_filter: Optional[str] = None
) -> Tuple[str, List[int], List[float]]:
    epochs: List[int] = []
    losses: List[float] = []
    tags_seen: List[str] = []

    for line in lines:
        m = EPOCH_MSE_RE.match(line.strip())
        if not m:
            continue

        tag = m.group("tag").strip()
        if tag_filter is not None and tag != tag_filter:
            continue

        tags_seen.append(tag)
        epochs.append(int(m.group("epoch")))
        losses.append(float(m.group("mse")))

    if not epochs:
        raise RuntimeError(
            "No matching '[<TAG>] Epoch ... train MSE=...' lines found in the given range. "
            "Double-check the line range, log format, or use --tag to filter."
        )

    unique_tags = sorted(set(tags_seen))
    if tag_filter is not None:
        return tag_filter, epochs, losses

    if len(unique_tags) != 1:
        raise RuntimeError(
            "Multiple different tags were found in the selected range: "
            + ", ".join(f"[{t}]" for t in unique_tags)
            + ".\nPlease narrow the line range or specify one with --tag '<TAG>'."
        )

    return unique_tags[0], epochs, losses


def _configure_matplotlib() -> None:
    # Academic, clean defaults (English labels + Times New Roman).
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "stix",
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.linewidth": 1.0,
            "grid.linewidth": 0.6,
        }
    )


def plot_curve(tag: str, epochs: List[int], losses: List[float], out_path: Path | None) -> None:
    _configure_matplotlib()

    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    ax.plot(epochs, losses, linewidth=2.0, color="black", label="Train MSE")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss (MSE)")
    ax.set_title(f"{tag} Training Loss vs Epoch")

    ax.grid(True, which="major", linestyle="--", alpha=0.35)

    # Clean spines (still keep a classic academic look).
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(frameon=False, loc="best")
    fig.tight_layout()

    if out_path is None:
        plt.show()
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)

    plt.close(fig)


def _read_line_range(file_path: Path, start_line: int, end_line: int) -> List[str]:
    if start_line < 1 or end_line < 1:
        raise ValueError("start-line and end-line must be >= 1 (1-based indexing).")
    if end_line < start_line:
        raise ValueError("end-line must be >= start-line.")

    # Read all lines once; logs are typically manageable.
    lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()

    start_idx = start_line - 1
    end_idx_exclusive = min(end_line, len(lines))
    if start_idx >= len(lines):
        raise ValueError(
            f"start-line={start_line} exceeds file length ({len(lines)} lines)."
        )

    return lines[start_idx:end_idx_exclusive]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Parse [PDL] Epoch ... train MSE=... lines from a text file "
            "within a 1-based line range and plot loss vs epoch (Times New Roman)."
        )
    )
    parser.add_argument("--file", required=True, help="Path to the text/log file.")
    parser.add_argument(
        "--start-line", type=int, required=True, help="1-based start line (inclusive)."
    )
    parser.add_argument(
        "--end-line", type=int, required=True, help="1-based end line (inclusive)."
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output image path (e.g., pdl_loss.png). If omitted, shows an interactive window.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Optional: only parse lines whose bracket tag equals this value (e.g., 'PDL' or 'DeDL/SDL').",
    )

    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    snippet = _read_line_range(file_path, args.start_line, args.end_line)
    tag, epochs, losses = parse_epoch_mse(snippet, tag_filter=args.tag)

    # If epochs repeat or are unsorted, sort by epoch.
    pairs = sorted(zip(epochs, losses), key=lambda t: t[0])
    epochs_sorted = [p[0] for p in pairs]
    losses_sorted = [p[1] for p in pairs]

    out_path = Path(args.out) if args.out else None
    plot_curve(tag, epochs_sorted, losses_sorted, out_path)

    print(
        f"Parsed {len(epochs_sorted)} points. "
        f"Epoch range: {epochs_sorted[0]}..{epochs_sorted[-1]}. "
        f"Loss range: {min(losses_sorted):.6g}..{max(losses_sorted):.6g}."
    )
    if out_path is not None:
        print(f"Saved plot to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
