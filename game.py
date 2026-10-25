"""Connect Four rules: board, moves, win/draw detection."""

ROWS, COLS = 6, 7
EMPTY, RED, YELLOW = 0, 1, 2
CENTER_ORDER = (3, 2, 4, 1, 5, 0, 6)
_DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


class Board:
    """grid[row][col], row 0 is the bottom. Red always moves first."""

    def __init__(self):
        self.grid = [[EMPTY] * COLS for _ in range(ROWS)]
        self.heights = [0] * COLS
        self.moves = []

    @classmethod
    def from_moves(cls, moves: str) -> "Board":
        b = cls()
        for ch in moves:
            b.drop(int(ch))
        return b

    @property
    def turn(self) -> int:
        return RED if len(self.moves) % 2 == 0 else YELLOW

    def valid_moves(self) -> list[int]:
        return [c for c in CENTER_ORDER if self.heights[c] < ROWS]

    def drop(self, col: int) -> None:
        if not 0 <= col < COLS or self.heights[col] >= ROWS:
            raise ValueError(f"cannot drop in column {col}")
        self.grid[self.heights[col]][col] = self.turn
        self.heights[col] += 1
        self.moves.append(col)

    def undo(self) -> None:
        col = self.moves.pop()
        self.heights[col] -= 1
        self.grid[self.heights[col]][col] = EMPTY

    def last_move_won(self) -> bool:
        if not self.moves:
            return False
        col = self.moves[-1]
        row = self.heights[col] - 1
        piece = self.grid[row][col]
        for dr, dc in _DIRECTIONS:
            count = 1
            for sign in (1, -1):
                r, c = row + dr * sign, col + dc * sign
                while 0 <= r < ROWS and 0 <= c < COLS and self.grid[r][c] == piece:
                    count += 1
                    r, c = r + dr * sign, c + dc * sign
            if count >= 4:
                return True
        return False

    def is_full(self) -> bool:
        return len(self.moves) == ROWS * COLS

    def result(self) -> int | None:
        """None = ongoing, 0 = draw, RED/YELLOW = winner."""
        if self.last_move_won():
            return YELLOW if self.turn == RED else RED
        return 0 if self.is_full() else None


def _self_check():
    b = Board.from_moves("")
    assert b.turn == RED and b.valid_moves() == [3, 2, 4, 1, 5, 0, 6]
    assert b.result() is None and not b.last_move_won()

    b = Board.from_moves("33")
    assert b.grid[0][3] == RED and b.grid[1][3] == YELLOW and b.heights[3] == 2
    b.undo()
    assert b.grid[1][3] == EMPTY and b.heights[3] == 1 and b.moves == [3]

    assert Board.from_moves("0011223").result() == RED       # horizontal
    assert Board.from_moves("0101010").result() == RED       # vertical
    assert Board.from_moves("01122323363").result() == RED   # diagonal /
    assert Board.from_moves("65544343303").result() == RED   # diagonal \ at right edge
    assert Board.from_moves("001122").result() is None

    draw = Board.from_moves("0213465" * 6)                   # full board, no winner
    assert draw.is_full() and draw.result() == 0 and draw.valid_moves() == []

    full = Board.from_moves("000000")
    for bad in (0, 7, -1):
        try:
            full.drop(bad)
            assert False, f"drop({bad}) should raise"
        except ValueError:
            pass
    assert full.moves == [0] * 6
    print("game.py OK")


if __name__ == "__main__":
    _self_check()
