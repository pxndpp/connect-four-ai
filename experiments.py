"""Experiments: search cost (nodes/time), paired 98-game matches, minimax vs MCTS."""

import csv
import itertools
import math
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, NullFormatter

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


# --- plotting helpers -----------------------------------------------------------

COLOR = {"minimax": "#2b6cb0", "alphabeta": "#e07b00", "mcts": "#6f42c1"}
VERDICT = {"ahead": "#2f9e44", "behind": "#d9480f", "even": "#868e96"}
STYLE = {"font.family": ["Tahoma", "Leelawadee UI", "DejaVu Sans"], "font.size": 12, "axes.titlesize": 14, "axes.titleweight": "bold", "axes.titlelocation": "left",
         "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True}
MATCH_LABELS = {"d2 vs random": "มอง 2 ตา\nสู้ สุ่มหมาก",
                "d4 vs d2": "มอง 4 ตา\nสู้ มอง 2 ตา",
                "d6 vs d4": "มอง 6 ตา\nสู้ มอง 4 ตา",
                "d6 vs d2": "มอง 6 ตา\nสู้ มอง 2 ตา",
                "H2 vs H1 (d4)": "สูตรประเมิน H2\nสู้ H1 (มอง 4 ตาเท่ากัน)"}


def _human(v: float) -> str:
    """12345 -> '12.3k', 0.15 -> '0.15': readable log-axis ticks and labels."""
    if v >= 1e6:
        return f"{v / 1e6:.3g}M"
    if v >= 1e3:
        return f"{v / 1e3:.3g}k"
    return f"{v:.3g}"


def _wilson(points: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Approximate 95% interval (in %) for a score of `points` out of `n` games."""
    p = points / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (centre - half), 100 * (centre + half)


def _verdict(lo: float, hi: float) -> str:
    return "ahead" if lo > 50 else "behind" if hi < 50 else "even"


def _legend_and_note(fig, who: str, note: str) -> None:
    fig.legend(handles=[Patch(color=VERDICT["ahead"], label=f"{who} ชนะชัดเจน"),
                        Patch(color=VERDICT["behind"], label=f"{who} แพ้ชัดเจน"),
                        Patch(color=VERDICT["even"], label="สูสี (ผลต่างน้อยเกินกว่าจะสรุปว่าใครเก่งกว่า)")],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.045))
    fig.text(0.01, 0.01, note, fontsize=9, color="#555")


def plot(csv_path: Path, out_dir: Path) -> None:
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    n_pos = len({r["position"] for r in rows})
    names = {"minimax": "Minimax ธรรมดา", "alphabeta": "Alpha-beta (ตัดกิ่ง)"}
    specs = (("nodes", "nodes.png", "จำนวนตำแหน่งที่ AI ต้องค้นต่อ 1 ตา (แกน log)", "",
              lambda r, d: f"Alpha-beta ค้นตำแหน่งน้อยลง {r[:-1]} เท่า (มองล่วงหน้า {d} ตา)"),
             ("ms", "time.png", "เวลาคิดต่อ 1 ตา (แกน log)", " ms",
              lambda r, d: f"Alpha-beta คิดเร็วขึ้น {r[:-1]} เท่า (มองล่วงหน้า {d} ตา)"))
    with plt.rc_context(STYLE):
        for metric, fname, ylabel, unit, title in specs:
            stats = {}
            for algo in names:
                stats[algo] = {}
                for d in sorted({int(r["depth"]) for r in rows if r["algo"] == algo}):
                    v = [float(r[metric]) for r in rows if r["algo"] == algo and int(r["depth"]) == d]
                    stats[algo][d] = (statistics.mean(v), min(v), max(v))
            fig, ax = plt.subplots(figsize=(8.5, 5.2))
            for algo, label in names.items():
                ds = list(stats[algo])
                ax.fill_between(ds, [stats[algo][d][1] for d in ds], [stats[algo][d][2] for d in ds],
                                color=COLOR[algo], alpha=0.15, linewidth=0)
                ax.plot(ds, [stats[algo][d][0] for d in ds], marker="o", markersize=7, linewidth=2.5,
                        color=COLOR[algo])
                end = stats[algo][ds[-1]][0]
                ax.annotate(f"{label}\n{_human(end)}{unit}", (ds[-1], end), xytext=(10, 0),
                            textcoords="offset points", va="center", color=COLOR[algo], fontweight="bold")
            dmax = max(set(stats["minimax"]) & set(stats["alphabeta"]))
            mm, ab = stats["minimax"][dmax][0], stats["alphabeta"][dmax][0]
            ratio = mm / ab
            ratio_text = f"{ratio:.0f}\u00d7" if ratio >= 10 else f"{ratio:.1f}\u00d7"
            ax.annotate("", xy=(dmax, ab), xytext=(dmax, mm),
                        arrowprops=dict(arrowstyle="<->", color="#333", lw=1.6))
            ax.text(dmax - 0.12, math.sqrt(mm * ab), ratio_text, ha="right", va="center",
                    fontsize=15, fontweight="bold", color="#333")
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _human(v)))
            ax.yaxis.set_minor_formatter(NullFormatter())
            depths = sorted(set(stats["minimax"]) | set(stats["alphabeta"]))
            ax.set_xticks(depths)
            ax.set_xlim(depths[0] - 0.3, depths[-1] + 1.3)
            ax.grid(axis="y", alpha=0.3)
            ax.set_xlabel("AI มองล่วงหน้ากี่ตา (ยิ่งมาก ยิ่งฉลาด แต่ช้าลง)")
            ax.set_ylabel(ylabel)
            ax.set_title(title(ratio_text, dmax))
            fig.text(0.01, 0.01, f"วิธีอ่าน: เส้น = ค่าเฉลี่ยจาก {n_pos} ท่าเปิด, แถบจาง = ช่วงต่ำสุดถึงสูงสุด\n"
                                 "แกนตั้งเป็น log: เส้นตารางแต่ละขั้นคือมากขึ้น 10 เท่า "
                                 "(ยิ่งเส้นอยู่ต่ำ ยิ่งประหยัด)", fontsize=10, color="#555")
            fig.tight_layout(rect=(0, 0.08, 1, 1))
            fig.savefig(out_dir / fname, dpi=200)
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
    stats = []
    for n in names:
        games = [r for r in rows if r["match"] == n]
        won = lost = drawn = 0
        for r in games:
            if r["result"] == "0":
                drawn += 1
            elif (r["result"] == str(RED)) == (r["red"] == "A"):
                won += 1
            else:
                lost += 1
        pts = won + drawn / 2
        lo, hi = _wilson(pts, len(games))
        stats.append((n, 100 * pts / len(games), lo, hi, won, drawn, lost))
    n_open = len({r["opening"] for r in rows})
    per_match = len(rows) // len(names)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(11, 0.95 * len(names) + 2.9))
        ys = list(range(len(names)))[::-1]
        for y, (n, pct, lo, hi, won, drawn, lost) in zip(ys, stats):
            ax.barh(y, pct, color=VERDICT[_verdict(lo, hi)], height=0.6)
            ax.errorbar(pct, y, xerr=[[max(0, pct - lo)], [max(0, hi - pct)]], color="#222", capsize=4, lw=1.5)
            ax.text(104, y, f"{pct:.1f}%   (ชนะ {won} เสมอ {drawn} แพ้ {lost})", va="center", fontsize=11)
        ax.axvline(50, color="#555", linestyle="--", linewidth=1)
        ax.text(50, len(names) - 0.3, "เสมอกัน (50%)", ha="center", va="bottom", color="#555", fontsize=10, bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
        ax.set_ylim(-0.6, len(names) - 0.1)
        ax.spines["bottom"].set_bounds(0, 100)
        ax.set_yticks(ys, [MATCH_LABELS.get(n, n) for n, *_ in stats])
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, 155)
        ax.set_xticks(range(0, 101, 25), [f"{v}%" for v in range(0, 101, 25)])
        ax.set_xlabel("คะแนนของ AI ฝั่งที่ชื่ออยู่ก่อน (ชนะ = 1 เสมอ = \u00bd)  ยิ่งเกิน 50% ยิ่งเก่งกว่า", loc="left")
        ax.set_title("AI ตัวไหนชนะ? ให้เล่นกันเองหลายเกม", pad=18)
        fig.tight_layout(rect=(0, 0.12, 1, 1))
        _legend_and_note(fig, "AI ฝั่งแรก",
                         f"วิธีอ่าน: แท่งยาวเกินเส้น 50% = ฝั่งแรกชนะมากกว่า | {n_open} ท่าเปิด \u00d7 สลับข้าง 2 รอบ = {per_match} เกมต่อคู่\n"
                         "เส้นดำ = ช่วงความเชื่อมั่น 95% โดยประมาณ ถ้าคร่อมเส้น 50% แปลว่ายังสรุปไม่ได้ว่าใครเก่งกว่า")
        fig.savefig(out_dir / "winrate.png", dpi=200)
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
    xs = range(len(rows))
    with plt.rc_context(STYLE):
        fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(13.5, 5.4), gridspec_kw={"width_ratios": [1.3, 1, 1]})
        for i, r in enumerate(rows):
            n, pts = int(r["games"]), float(r["minimax_points"])
            pct, (lo, hi) = 100 * pts / n, _wilson(pts, n)
            a1.bar(i, pct, color=VERDICT[_verdict(lo, hi)], width=0.6)
            a1.errorbar(i, pct, yerr=[[max(0, pct - lo)], [max(0, hi - pct)]], color="#222", capsize=5, lw=1.5)
            a1.text(i, hi + 2, f"{pct:.1f}%", ha="center", fontweight="bold")
        a1.axhline(50, color="#555", linestyle="--", linewidth=1)
        if len(rows) > 1:
            a1.text((len(rows) - 1) / 2, 50, "เสมอกัน", ha="center", va="center", color="#555", fontsize=10,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
        a1.set_ylim(0, 100)
        a1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
        a1.set_ylabel("คะแนน Minimax (ชนะ = 1 เสมอ = \u00bd)")
        a1.set_title("ใครชนะ?")
        depth = [float(r["mean_id_depth"]) for r in rows]
        a2.bar(xs, depth, color=COLOR["minimax"], width=0.6)
        for i, d in enumerate(depth):
            a2.text(i, d + 0.15, f"{d:.1f}", ha="center", fontweight="bold")
        a2.set_ylim(0, max(depth) * 1.2)
        a2.set_ylabel("มองล่วงหน้าเฉลี่ยกี่ตา")
        a2.set_title("Minimax มองลึกแค่ไหน")
        iters = [int(float(r["mean_mcts_iterations"])) for r in rows]
        a3.bar(xs, iters, color=COLOR["mcts"], width=0.6)
        for i, v in enumerate(iters):
            a3.text(i, v + max(iters) * 0.015, f"{v:,}", ha="center", fontweight="bold")
        a3.set_ylim(0, max(iters) * 1.2)
        a3.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _human(v)))
        a3.set_ylabel("จำนวนเกมจำลองเฉลี่ยต่อ 1 ตา")
        a3.set_title("MCTS จำลองเกมกี่รอบ")
        for ax in (a1, a2, a3):
            ax.set_xticks(list(xs), labels)
            ax.set_xlabel("เวลาคิดที่ให้ต่อ 1 ตา")
            ax.grid(axis="y", alpha=0.3)
        fig.suptitle("Minimax สู้ MCTS เมื่อให้เวลาคิดเท่ากัน", x=0.01, ha="left",
                     fontsize=16, fontweight="bold")
        fig.tight_layout(rect=(0, 0.12, 1, 0.95))
        _legend_and_note(fig, "Minimax",
                         f"วิธีอ่าน: แท่งเกินเส้น 50% = Minimax ชนะมากกว่า | {int(rows[0]['games'])} เกมต่อการตั้งค่า (ทุกท่าเปิด สลับข้าง)\n"
                         "เส้นดำ = ช่วงความเชื่อมั่น 95% โดยประมาณ | MCTS ตัดสินด้วยนาฬิกา รันซ้ำตัวเลขจะต่างเล็กน้อย")
        fig.savefig(out_dir / "budget.png", dpi=200)
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
