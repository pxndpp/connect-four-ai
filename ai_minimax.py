"""Negamax search (plain or alpha-beta) with heuristic H1."""

import time
from typing import NamedTuple

from game import Board, RED, YELLOW, ROWS, COLS

WIN_SCORE = 1_000_000
_INF = float("inf")


class Result(NamedTuple):
    move: int | None
    score: int
    nodes: int
    depth: int


def _build_windows():
    windows = []
    for r in range(ROWS):
        for c in range(COLS):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                cells = [(r + dr * i, c + dc * i) for i in range(4)]
                if all(0 <= rr < ROWS and 0 <= cc < COLS for rr, cc in cells):
                    windows.append(cells)
    return windows


_WINDOWS = _build_windows()
assert len(_WINDOWS) == 69


def h1(board: Board, player: int) -> int:
    """Windows: own3+1 empty +5, own2+2 empty +2, opp3+1 empty -4; own piece in column 3: +3."""
    grid = board.grid
    opp = YELLOW if player == RED else RED
    score = 3 * sum(1 for r in range(ROWS) if grid[r][3] == player)
    for cells in _WINDOWS:
        own = theirs = 0
        for r, c in cells:
            v = grid[r][c]
            if v == player:
                own += 1
            elif v == opp:
                theirs += 1
        empty = 4 - own - theirs
        if own == 3 and empty == 1:
            score += 5
        elif own == 2 and empty == 2:
            score += 2
        elif theirs == 3 and empty == 1:
            score -= 4
    return score


def _terminal_score(board: Board, depth_left: int):
    """Score for the side to move if the game is over, else None."""
    if board.last_move_won():
        return -(WIN_SCORE + depth_left)   # losing sooner is worse
    if board.is_full():
        return 0
    return None


class _Timeout(Exception):
    pass


def _negamax(board, depth_left, alpha, beta, alphabeta, heuristic, nodes, deadline=None):
    nodes[0] += 1
    if deadline is not None and time.perf_counter() > deadline:
        raise _Timeout
    term = _terminal_score(board, depth_left)
    if term is not None:
        return term
    if depth_left == 0:
        return heuristic(board, board.turn)
    best = -_INF
    for col in board.valid_moves():
        board.drop(col)
        try:
            score = -_negamax(board, depth_left - 1, -beta, -alpha, alphabeta, heuristic, nodes, deadline)
        finally:
            board.undo()
        if score > best:
            best = score
        if alphabeta:
            alpha = max(alpha, best)
            if alpha >= beta:
                break
    return best


def _root(board, depth, alphabeta, heuristic, nodes, deadline=None):
    nodes[0] += 1
    best_move, best, alpha = None, -_INF, -_INF
    for col in board.valid_moves():
        board.drop(col)
        try:
            score = -_negamax(board, depth - 1, -_INF, -alpha, alphabeta, heuristic, nodes, deadline)
        finally:
            board.undo()
        if score > best:
            best_move, best = col, score
        if alphabeta:
            alpha = max(alpha, best)
    return best_move, best


def search(board: Board, depth: int, alphabeta: bool = True, heuristic=h1) -> Result:
    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth}")
    term = _terminal_score(board, depth)
    if term is not None:
        return Result(None, term, 1, depth)
    nodes = [0]
    move, score = _root(board, depth, alphabeta, heuristic, nodes)
    return Result(move, score, nodes[0], depth)


def iterative_deepening(board: Board, time_limit: float, heuristic=h1, max_depth: int = 42) -> Result:
    """Alpha-beta at depth 1, 2, ... until time runs out; depth 1 always completes.
    Returns the deepest fully searched result; nodes include the aborted depth."""
    term = _terminal_score(board, 0)
    if term is not None:
        return Result(None, term, 1, 0)
    deadline = time.perf_counter() + time_limit
    nodes = [0]
    move, score = _root(board, 1, True, heuristic, nodes)
    done = 1
    for depth in range(2, max_depth + 1):
        if abs(score) >= WIN_SCORE:
            break
        try:
            move, score = _root(board, depth, True, heuristic, nodes, deadline)
        except _Timeout:
            break
        done = depth
    return Result(move, score, nodes[0], done)


def _self_check():
    assert h1(Board.from_moves(""), RED) == 0
    assert h1(Board.from_moves("3"), RED) == 3
    assert h1(Board.from_moves("3"), YELLOW) == 0

    for d in (1, 2, 3, 4):
        assert search(Board.from_moves("001122"), d).move == 3   # immediate win
    for d in (2, 4):
        assert search(Board.from_moves("00112"), d).move == 3    # must block

    assert search(Board.from_moves(""), 1, alphabeta=False).nodes == 8
    assert search(Board.from_moves(""), 2, alphabeta=False).nodes == 57

    for pos in ("", "33", "3243", "001122", "0213465"):
        for d in (1, 2, 3, 4):
            mm = search(Board.from_moves(pos), d, alphabeta=False)
            ab = search(Board.from_moves(pos), d, alphabeta=True)
            assert (mm.move, mm.score) == (ab.move, ab.score), (pos, d)
    assert search(Board.from_moves(""), 4).nodes < search(Board.from_moves(""), 4, alphabeta=False).nodes

    done = Board.from_moves("0101010")
    assert search(done, 3).move is None and done.moves == [0, 1, 0, 1, 0, 1, 0]
    for bad in (0, -1):
        try:
            search(Board.from_moves("33"), bad)
            assert False, f"search depth {bad} should raise"
        except ValueError:
            pass

    b = Board.from_moves("3243")
    before = list(b.moves)
    r = iterative_deepening(b, 0.05)
    assert b.moves == before and r.move in b.valid_moves() and r.depth >= 1
    assert iterative_deepening(Board.from_moves(""), 0.0).depth == 1
    assert iterative_deepening(Board.from_moves("001122"), 1.0).move == 3
    assert iterative_deepening(Board.from_moves("0101010"), 0.1).move is None
    slow = iterative_deepening(Board.from_moves(""), 0.5)
    assert slow.move == search(Board.from_moves(""), slow.depth).move
    print("ai_minimax.py OK")


if __name__ == "__main__":
    _self_check()
