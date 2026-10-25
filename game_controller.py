"""Game flow for the UI, with no pygame dependency so it can be tested headless."""

import random
import time
from typing import NamedTuple

from ai_minimax import search, iterative_deepening
from game import Board, RED, YELLOW, COLS

DIFFICULTIES = {"easy": ("depth", 2), "medium": ("depth", 4), "hard": ("time", 1.0)}
BOARD_X0, CELL = 0, 90


def col_from_x(x: int) -> int | None:
    col = (x - BOARD_X0) // CELL
    return col if x >= BOARD_X0 and col < COLS else None


class MoveStats(NamedTuple):
    col: int
    nodes: int
    ms: float
    depth: int


class Game:
    def __init__(self, human_player: int = RED, difficulty: str = "medium", seed: int | None = None):
        self.rng = random.Random(seed)
        self.human_player = human_player
        self.set_difficulty(difficulty)
        self.new_game()

    def is_over(self) -> bool:
        return self.board.result() is not None

    def is_human_turn(self) -> bool:
        return not self.is_over() and self.board.turn == self.human_player

    def human_move(self, col: int) -> bool:
        if not self.is_human_turn() or col not in self.board.valid_moves():
            return False
        self.board.drop(col)
        return True

    def ai_move(self) -> MoveStats:
        if self.is_over() or self.is_human_turn():
            raise RuntimeError("not the AI's turn")
        kind, value = DIFFICULTIES[self.difficulty]
        t0 = time.perf_counter()
        if kind == "depth":
            r = search(self.board, value, rng=self.rng)
            depth = value
        else:
            r = iterative_deepening(self.board, value, rng=self.rng)
            depth = r.depth
        ms = (time.perf_counter() - t0) * 1000
        self.board.drop(r.move)
        self.last_ai = MoveStats(r.move, r.nodes, ms, depth)
        return self.last_ai

    def set_difficulty(self, name: str) -> None:
        if name not in DIFFICULTIES:
            raise ValueError(f"unknown difficulty {name!r}")
        self.difficulty = name

    def new_game(self, human_player: int | None = None) -> None:
        if human_player is not None:
            self.human_player = human_player
        self.board = Board()
        self.last_ai = None

    def switch_side(self) -> None:
        self.new_game(YELLOW if self.human_player == RED else RED)

    def status_text(self) -> str:
        result = self.board.result()
        if result is None:
            return "Your turn" if self.board.turn == self.human_player else "AI thinking..."
        if result == 0:
            return "Draw"
        return "You win!" if result == self.human_player else "AI wins"


def _self_check():
    assert col_from_x(0) == 0 and col_from_x(629) == 6 and col_from_x(630) is None and col_from_x(-1) is None

    g = Game(seed=0)
    assert g.is_human_turn() and g.status_text() == "Your turn"
    assert g.human_move(3) and not g.is_human_turn()
    assert not g.human_move(2)                                # still the AI's turn
    s = g.ai_move()
    assert s.col in range(7) and s.nodes > 0 and s.depth == 4 and g.last_ai == s and len(g.board.moves) == 2
    try:
        g.ai_move()
        assert False
    except RuntimeError:
        pass

    g = Game(human_player=YELLOW, difficulty="easy", seed=0)
    assert not g.is_human_turn() and g.status_text() == "AI thinking..."
    g.ai_move()
    assert g.is_human_turn()

    g = Game(seed=0)
    for c in "001122":
        g.board.drop(int(c))
    assert g.human_move(3) and g.is_over() and g.status_text() == "You win!"
    assert not g.human_move(4) and not g.is_human_turn()

    g = Game(seed=0)
    for c in "000000":
        g.board.drop(int(c))
    assert not g.human_move(0) and len(g.board.moves) == 6   # full column

    g = Game(seed=0)
    g.human_move(3)
    g.set_difficulty("hard")
    assert g.ai_move().depth >= 1
    g.switch_side()
    assert g.human_player == YELLOW and g.board.moves == [] and g.last_ai is None
    try:
        g.set_difficulty("insane")
        assert False
    except ValueError:
        pass

    g = Game(seed=0)
    for c in "0213465" * 6:
        g.board.drop(int(c))
    assert g.is_over() and g.status_text() == "Draw"
    print("game_controller.py OK")


if __name__ == "__main__":
    _self_check()
