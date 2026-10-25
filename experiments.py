"""Experiments: search cost (nodes/time), paired 98-game matches, minimax vs MCTS."""

import csv
import itertools
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ai_minimax import search, iterative_deepening, h1, h2
from ai_mcts import mcts
from game import Board, RED

RESULTS = Path(__file__).parent / "results"

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


# --- agents: callables Board -> column ---------------------------------------

def minimax_agent(depth: int, heuristic=h1):
    return lambda board: search(board, depth, heuristic=heuristic).move


def id_agent(time_limit: float, stats: list | None = None):
    def agent(board):
        r = iterative_deepening(board, time_limit)
        if stats is not None:
            stats.append(r.depth)
        return r.move
    return agent


def mcts_agent(time_limit: float, seed: int, stats: list | None = None):
    rng = random.Random(seed)
    def agent(board):
        r = mcts(board, time_limit=time_limit, rng=rng)
        if stats is not None:
            stats.append(r.iterations)
        return r.move
    return agent


def random_agent(seed: int):
    rng = random.Random(seed)
    return lambda board: rng.choice(board.valid_moves())


# --- matches ------------------------------------------------------------------

MATCH_FIELDS = ["match", "opening", "red", "yellow", "result"]


def play_game(red, yellow, opening: str) -> int:
    """Play from the opening to the end; returns 1 (red), 2 (yellow) or 0 (draw)."""
    board = Board.from_moves(opening)
    while board.result() is None:
        agent = red if board.turn == RED else yellow
        board.drop(agent(board))
    return board.result()


def match(name: str, a, b, openings_list: list[str], writer=None) -> dict:
    """Paired games: for each opening, A plays red once and yellow once. Win 1, draw 0.5."""
    s = {"a_points": 0.0, "b_points": 0.0, "a_wins": 0, "b_wins": 0, "draws": 0}
    for op in openings_list:
        for red_name in ("A", "B"):
            red, yellow = (a, b) if red_name == "A" else (b, a)
            res = play_game(red, yellow, op)
            if res == 0:
                s["draws"] += 1
                s["a_points"] += 0.5
                s["b_points"] += 0.5
            elif (res == RED) == (red_name == "A"):
                s["a_wins"] += 1
                s["a_points"] += 1
            else:
                s["b_wins"] += 1
                s["b_points"] += 1
            if writer is not None:
                writer.writerow({"match": name, "opening": op, "red": red_name,
                                 "yellow": "B" if red_name == "A" else "A", "result": res})
    return s


def run_matches(out_csv: Path) -> list[dict]:
    pairs = [
        ("d2 vs random", minimax_agent(2), random_agent(0)),
        ("d4 vs d2", minimax_agent(4), minimax_agent(2)),
        ("d6 vs d4", minimax_agent(6), minimax_agent(4)),
        ("d6 vs d2", minimax_agent(6), minimax_agent(2)),
        ("H2 vs H1 (d4)", minimax_agent(4, h2), minimax_agent(4, h1)),
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summaries = []
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MATCH_FIELDS)
        writer.writeheader()
        for name, a, b in pairs:
            t0 = time.perf_counter()
            s = match(name, a, b, openings(), writer)
            f.flush()
            pct = 100 * s["a_points"] / (s["a_points"] + s["b_points"])
            print(f"{name:14} A {s['a_wins']:3} / B {s['b_wins']:3} / draw {s['draws']:3}"
                  f"  A score {pct:5.1f}%  ({time.perf_counter() - t0:.0f} s)", flush=True)
            summaries.append({"match": name, **s})
    return summaries


def plot_matches(csv_path: Path, out_dir: Path) -> None:
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    names = list(dict.fromkeys(r["match"] for r in rows))
    pcts = []
    for n in names:
        games = [r for r in rows if r["match"] == n]
        pts = sum(0.5 if r["result"] == "0" else
                  1.0 if (r["result"] == str(RED)) == (r["red"] == "A") else 0.0 for r in games)
        pcts.append(100 * pts / len(games))
    fig, ax = plt.subplots(figsize=(7, 0.6 * len(names) + 1.5))
    bars = ax.barh(names[::-1], pcts[::-1])
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.axvline(50, color="grey", linestyle="--", linewidth=1)
    ax.set_xlim(0, 110)
    ax.set_xlabel("Score of A (win 1, draw 0.5), % of games")
    ax.set_title(f"Paired games: 49 openings x 2 sides")
    fig.tight_layout()
    fig.savefig(out_dir / "winrate.png", dpi=150)
    plt.close(fig)


# --- minimax (iterative deepening) vs MCTS at equal time per move ---------------

BUDGET_FIELDS = ["budget", "minimax_points", "mcts_points", "draws", "games",
                 "mean_id_depth", "mean_mcts_iterations"]


def run_budget(budgets: tuple[float, ...], out_csv: Path, openings_list: list[str] | None = None) -> list[dict]:
    openings_list = openings() if openings_list is None else openings_list
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BUDGET_FIELDS)
        writer.writeheader()
        for budget in budgets:
            depths, iters = [], []
            t0 = time.perf_counter()
            s = match(f"{budget}s", id_agent(budget, depths),
                      mcts_agent(budget, int(budget * 1000), iters), openings_list)
            row = {"budget": budget, "minimax_points": s["a_points"], "mcts_points": s["b_points"],
                   "draws": s["draws"], "games": 2 * len(openings_list),
                   "mean_id_depth": round(statistics.mean(depths), 2),
                   "mean_mcts_iterations": round(statistics.mean(iters))}
            writer.writerow(row)
            f.flush()
            rows.append(row)
            print(f"{budget}s/move: minimax {s['a_points']} vs MCTS {s['b_points']} (draws {s['draws']}),"
                  f" ID depth {row['mean_id_depth']}, MCTS iters {row['mean_mcts_iterations']}"
                  f"  ({time.perf_counter() - t0:.0f} s)", flush=True)
    return rows


def plot_budget(csv_path: Path, out_dir: Path) -> None:
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    labels = [f"{float(r['budget']):g} s" for r in rows]
    pct = [100 * float(r["minimax_points"]) / int(r["games"]) for r in rows]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    bars = left.bar(labels, pct)
    left.bar_label(bars, fmt="%.1f%%")
    left.axhline(50, color="grey", linestyle="--", linewidth=1)
    left.set_ylim(0, 110)
    left.set_xlabel("Time per move")
    left.set_ylabel("Minimax (ID) score, % of games")
    left.set_title("Minimax vs MCTS, equal time")
    # Bars labelled with both numbers: twin-axis lines overlap exactly when there are only two budgets.
    depth = [float(r["mean_id_depth"]) for r in rows]
    effort = right.bar(labels, depth, color="C0")
    right.bar_label(effort, labels=[f"depth {d:.2f}\nMCTS {int(r['mean_mcts_iterations']):,} iters"
                                    for d, r in zip(depth, rows)])
    right.set_ylim(0, max(depth) * 1.3)
    right.set_xlabel("Time per move")
    right.set_ylabel("Mean depth reached by minimax (ID)")
    right.set_title("Search effort per move")
    fig.tight_layout()
    fig.savefig(out_dir / "budget.png", dpi=150)
    plt.close(fig)


# --- entry points ---------------------------------------------------------------

def main_nodes():
    run_nodes(openings()[::5], range(1, 7), range(1, 9), RESULTS / "nodes.csv")
    plot(RESULTS / "nodes.csv", RESULTS)
    print(f"wrote {RESULTS / 'nodes.csv'}, nodes.png, time.png")


def main_matches():
    run_matches(RESULTS / "matches.csv")
    plot_matches(RESULTS / "matches.csv", RESULTS)
    print(f"wrote {RESULTS / 'matches.csv'}, winrate.png")


def main_budget():
    run_budget((0.1, 0.5), RESULTS / "budget.csv")
    plot_budget(RESULTS / "budget.csv", RESULTS)
    print(f"wrote {RESULTS / 'budget.csv'}, budget.png")


COMMANDS = {"nodes": [main_nodes], "matches": [main_matches], "budget": [main_budget]}
COMMANDS["all"] = [main_nodes, main_matches, main_budget]


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

    assert play_game(minimax_agent(2), random_agent(0), "33") in (0, 1, 2)
    first = lambda b: b.valid_moves()[0]
    log = []
    class W:                                                  # fake writer that records rows
        def writerow(self, row): log.append(row)
    s = match("t", minimax_agent(2), first, ["00", "34"], writer=W())
    assert s["a_points"] + s["b_points"] == 4
    assert s["a_wins"] + s["b_wins"] + s["draws"] == 4
    assert [(r["opening"], r["red"]) for r in log] == [("00", "A"), ("00", "B"), ("34", "A"), ("34", "B")]
    assert match("t", minimax_agent(2), minimax_agent(2), ["33"]) == match("t", minimax_agent(2), minimax_agent(2), ["33"])
    scripted = lambda b: int(("0213465" * 6)[len(b.moves)])  # follows a sequence that ends in a draw
    draws = match("t", scripted, scripted, ["02"])
    assert draws["draws"] == 2 and draws["a_points"] == 1.0 and draws["b_points"] == 1.0
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "m.csv"
        p.write_text("match,opening,red,yellow,result\nx,00,A,B,1\nx,00,B,A,0\n")
        plot_matches(p, Path(tmp))                            # A: 1 win + 1 draw → 75%
        assert (Path(tmp) / "winrate.png").exists()

    with tempfile.TemporaryDirectory() as tmp:
        rows = run_budget((0.01,), Path(tmp) / "b.csv", openings_list=["33"])
        assert len(rows) == 1 and rows[0]["games"] == 2
        assert rows[0]["minimax_points"] + rows[0]["mcts_points"] == 2
        assert rows[0]["mean_id_depth"] >= 1 and rows[0]["mean_mcts_iterations"] >= 1
        plot_budget(Path(tmp) / "b.csv", Path(tmp))
        assert (Path(tmp) / "budget.png").exists()
    print("experiments.py OK")


if __name__ == "__main__":
    if "--check" in sys.argv:
        _self_check()
    else:
        cmd = sys.argv[1] if len(sys.argv) > 1 else "nodes"
        if cmd not in COMMANDS:
            sys.exit(f"usage: experiments.py [{' | '.join(COMMANDS)}] | --check")
        for fn in COMMANDS[cmd]:
            fn()
