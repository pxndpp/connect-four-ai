"""Monte Carlo Tree Search (UCT, random rollouts)."""

import math
import random
import time
from typing import NamedTuple

from game import Board


class MctsResult(NamedTuple):
    move: int | None
    iterations: int
    win_rate: float   # of the chosen move, for the side to move


class _Node:
    __slots__ = ("parent", "move", "mover", "untried", "children", "visits", "wins")

    def __init__(self, parent, move, mover, untried):
        self.parent, self.move, self.mover = parent, move, mover
        self.untried = untried          # moves not expanded yet; [] for a finished game
        self.children = {}
        self.visits = 0
        self.wins = 0.0                 # reward for `mover`, the side that moved into this node


def _iterate(board: Board, root: _Node, rng: random.Random, c: float) -> None:
    node, pushed = root, 0
    try:
        while not node.untried and node.children:                      # selection
            log_n = math.log(node.visits)
            node = max(node.children.values(),
                       key=lambda ch: ch.wins / ch.visits + c * math.sqrt(log_n / ch.visits))
            board.drop(node.move)
            pushed += 1
        if node.untried:                                                # expansion
            col = node.untried.pop(rng.randrange(len(node.untried)))
            mover = board.turn
            board.drop(col)
            pushed += 1
            child = _Node(node, col, mover, [] if board.result() is not None else board.valid_moves())
            node.children[col] = child
            node = child
        result = board.result()                                         # rollout
        while result is None:
            board.drop(rng.choice(board.valid_moves()))
            pushed += 1
            result = board.result()
    finally:
        for _ in range(pushed):
            board.undo()
    while node is not None:                                             # backprop
        node.visits += 1
        node.wins += 1.0 if result == node.mover else 0.5 if result == 0 else 0.0
        node = node.parent


def mcts(board: Board, time_limit: float | None = None, iterations: int | None = None,
         rng: random.Random | None = None, c: float = math.sqrt(2)) -> MctsResult:
    if (time_limit is None) == (iterations is None):
        raise ValueError("give exactly one of time_limit or iterations")
    if board.result() is not None:
        return MctsResult(None, 0, 0.0)
    rng = rng or random.Random()
    root = _Node(None, None, None, board.valid_moves())
    deadline = time.perf_counter() + time_limit if time_limit is not None else None
    done = 0
    while True:
        _iterate(board, root, rng, c)
        done += 1
        if deadline is not None:
            if time.perf_counter() >= deadline:
                break
        elif done >= iterations:
            break
    best = None
    for col in board.valid_moves():                                     # most visited; ties → centre-first
        ch = root.children.get(col)
        if ch and (best is None or ch.visits > best.visits):
            best = ch
    return MctsResult(best.move, done, best.wins / best.visits)


def _self_check():
    rng = random.Random(0)
    assert mcts(Board.from_moves("001122"), iterations=2000, rng=rng).move == 3      # immediate win
    assert mcts(Board.from_moves("00112"), iterations=4000, rng=rng).move == 3       # must block
    b = Board.from_moves("3243")
    before = list(b.moves)
    r = mcts(b, time_limit=0.05, rng=rng)
    assert b.moves == before and r.move in b.valid_moves() and r.iterations >= 1
    assert mcts(Board.from_moves(""), time_limit=0.0, rng=rng).iterations >= 1
    assert mcts(Board.from_moves("0101010"), iterations=10).move is None
    last = Board.from_moves("0213465" * 6)
    last.undo()
    assert mcts(last, iterations=10).move == last.valid_moves()[0]                   # one cell left
    a = mcts(Board.from_moves("33"), iterations=500, rng=random.Random(7))
    b2 = mcts(Board.from_moves("33"), iterations=500, rng=random.Random(7))
    assert a == b2                                                                    # same seed, same result
    for bad in ({}, {"time_limit": 1.0, "iterations": 10}):
        try:
            mcts(Board.from_moves(""), **bad)
            assert False, bad
        except ValueError:
            pass
    print("ai_mcts.py OK")


if __name__ == "__main__":
    _self_check()
