"""Experiments: nodes expanded and time per move, plain minimax vs alpha-beta."""

import csv
import itertools
import statistics
import sys
import tempfile
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ai_minimax import search
from game import Board

FIELDS = ["algo", "depth", "position", "nodes", "ms", "move"]


def openings() -> list[str]:
    """All 49 two-move openings, "00" .. "66"."""
    return [f"{a}{b}" for a, b in itertools.product(range(7), repeat=2)]


def run_nodes(positions: list[str], plain_depths, ab_depths, out_csv: Path) -> list[dict]:
    rows = []
    jobs = [("minimax", d, False) for d in plain_depths] + [("alphabeta", d, True) for d in ab_depths]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for algo, depth, ab in jobs:
            for pos in positions:
                t0 = time.perf_counter()
                r = search(Board.from_moves(pos), depth, alphabeta=ab)
                ms = (time.perf_counter() - t0) * 1000
                row = {"algo": algo, "depth": depth, "position": pos,
                       "nodes": r.nodes, "ms": round(ms, 2), "move": r.move}
                writer.writerow(row)
                f.flush()
                rows.append(row)
                print(f"{algo:9} depth {depth} pos {pos or '-':>2}: {r.nodes:>9} nodes {ms:>10.1f} ms", flush=True)
    return rows


def plot(csv_path: Path, out_dir: Path) -> None:
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    for metric, fname, ylabel in (("nodes", "nodes.png", "Nodes expanded (mean)"),
                                  ("ms", "time.png", "Time per move, ms (mean)")):
        fig, ax = plt.subplots(figsize=(6, 4))
        for algo in ("minimax", "alphabeta"):
            depths = sorted({int(r["depth"]) for r in rows if r["algo"] == algo})
            means = [statistics.mean(float(r[metric]) for r in rows
                                     if r["algo"] == algo and int(r["depth"]) == d) for d in depths]
            ax.plot(depths, means, marker="o", label="alpha-beta" if algo == "alphabeta" else "minimax")
        ax.set_yscale("log")
        ax.set_xlabel("Search depth")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=150)
        plt.close(fig)


def main():
    out = Path("results")
    run_nodes(openings()[::5], range(1, 7), range(1, 9), out / "nodes.csv")
    plot(out / "nodes.csv", out)
    print(f"wrote {out / 'nodes.csv'}, {out / 'nodes.png'}, {out / 'time.png'}")


def _self_check():
    ops = openings()
    assert len(ops) == 49 and ops[0] == "00" and ops[-1] == "66"
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "nodes.csv"
        rows = run_nodes(["", "33"], [1, 2], [1, 2], out)
        assert len(rows) == 8
        plain2 = [r for r in rows if r["algo"] == "minimax" and r["depth"] == 2 and r["position"] == ""]
        assert plain2[0]["nodes"] == 57
        assert out.read_text().splitlines()[0] == "algo,depth,position,nodes,ms,move"
        plot(out, Path(tmp))
        assert (Path(tmp) / "nodes.png").exists() and (Path(tmp) / "time.png").exists()
    print("experiments.py OK")


if __name__ == "__main__":
    if "--check" in sys.argv:
        _self_check()
    else:
        main()
