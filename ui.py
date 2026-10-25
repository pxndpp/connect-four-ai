"""Pygame screen for playing Connect Four against the AI.

Keys: 1/2/3 difficulty, S switch side (new game), N new game, Esc quit.
Run `python ui.py --smoke` (with SDL_VIDEODRIVER=dummy) to render each state once and exit.
"""

import math
import sys
import time

import pygame

from game import ROWS, COLS, RED, YELLOW, EMPTY
from game_controller import Game, col_from_x, DIFFICULTIES, CELL

TOP = CELL                        # strip above the board for the hovering piece
BOARD_W = COLS * CELL             # 630
PANEL_X = BOARD_W
WIDTH, HEIGHT = BOARD_W + 280, TOP + ROWS * CELL   # 910 x 630
RADIUS = CELL // 2 - 8
FALL_ACCEL = 2 * (ROWS * CELL) / 0.25 ** 2         # a 6-row fall takes 0.25 s

BG = (24, 26, 33)
BOARD_BLUE = (36, 84, 196)
HOLE = (16, 18, 24)
PIECE = {RED: (220, 50, 47), YELLOW: (240, 196, 25)}
TEXT = (230, 232, 238)
MUTED = (150, 155, 168)
BUTTON = (52, 56, 70)
BUTTON_ON = (90, 120, 220)

BUTTONS = {                                        # name -> rect in the side panel
    "easy": pygame.Rect(PANEL_X + 20, 190, 76, 40),
    "medium": pygame.Rect(PANEL_X + 102, 190, 76, 40),
    "hard": pygame.Rect(PANEL_X + 184, 190, 76, 40),
    "switch": pygame.Rect(PANEL_X + 20, 250, 240, 40),
    "new": pygame.Rect(PANEL_X + 20, 300, 240, 40),
}
LABELS = {"easy": "Easy (1)", "medium": "Med (2)", "hard": "Hard (3)",
          "switch": "Switch side (S)", "new": "New game (N)"}


def cell_center(row: int, col: int) -> tuple[int, int]:
    return col * CELL + CELL // 2, TOP + (ROWS - 1 - row) * CELL + CELL // 2


class FallAnim:
    """A piece already placed in board.grid, drawn falling from the top strip."""

    def __init__(self, game: Game):
        self.col = game.board.moves[-1]
        self.row = game.board.heights[self.col] - 1
        self.player = game.board.grid[self.row][self.col]
        self.y0 = TOP // 2
        self.y1 = cell_center(self.row, self.col)[1]
        self.start = time.perf_counter()
        self.duration = math.sqrt(2 * (self.y1 - self.y0) / FALL_ACCEL)

    def y(self) -> float:
        t = time.perf_counter() - self.start
        return min(self.y1, self.y0 + 0.5 * FALL_ACCEL * t * t)

    def done(self) -> bool:
        return time.perf_counter() - self.start >= self.duration


def draw(screen, fonts, game: Game, hover_col, anim: FallAnim | None) -> None:
    big, mid, small = fonts
    screen.fill(BG)

    if hover_col is not None and anim is None and game.is_human_turn():   # translucent hovering piece
        ghost = pygame.Surface((CELL, CELL), pygame.SRCALPHA)
        pygame.draw.circle(ghost, (*PIECE[game.human_player], 110), (CELL // 2, CELL // 2), RADIUS)
        screen.blit(ghost, (hover_col * CELL, 0))
    if anim is not None:
        pygame.draw.circle(screen, PIECE[anim.player], (anim.col * CELL + CELL // 2, int(anim.y())), RADIUS)

    pygame.draw.rect(screen, BOARD_BLUE, (0, TOP, BOARD_W, ROWS * CELL))
    for r in range(ROWS):
        for c in range(COLS):
            v = game.board.grid[r][c]
            if anim is not None and (r, c) == (anim.row, anim.col):
                v = EMPTY
            pygame.draw.circle(screen, PIECE.get(v, HOLE), cell_center(r, c), RADIUS)
    if anim is not None:     # draw the falling piece again over the board so it shows through the holes
        pygame.draw.circle(screen, PIECE[anim.player], (anim.col * CELL + CELL // 2, int(anim.y())), RADIUS)

    # side panel
    x = PANEL_X + 20
    screen.blit(big.render("Connect Four", True, TEXT), (x, 20))
    screen.blit(mid.render(game.status_text(), True, TEXT), (x, 70))
    side = "Red (moves first)" if game.human_player == RED else "Yellow (AI first)"
    pygame.draw.circle(screen, PIECE[game.human_player], (x + 10, 128), 10)
    screen.blit(small.render(f"You: {side}", True, MUTED), (x + 28, 118))
    screen.blit(small.render("Difficulty", True, MUTED), (x, 165))
    for name, rect in BUTTONS.items():
        on = name == game.difficulty
        pygame.draw.rect(screen, BUTTON_ON if on else BUTTON, rect, border_radius=6)
        label = small.render(LABELS[name], True, TEXT)
        screen.blit(label, label.get_rect(center=rect.center))

    screen.blit(small.render("Last AI move", True, MUTED), (x, 370))
    s = game.last_ai
    rows = [("Column", "-" if s is None else str(s.col)),
            ("Nodes", "-" if s is None else f"{s.nodes:,}"),
            ("Time (ms)", "-" if s is None else f"{s.ms:,.0f}"),
            ("Depth", "-" if s is None else str(s.depth))]
    for i, (k, v) in enumerate(rows):
        screen.blit(mid.render(k, True, TEXT), (x, 400 + i * 34))
        val = mid.render(v, True, TEXT)
        screen.blit(val, (PANEL_X + 260 - val.get_width(), 400 + i * 34))
    kind, value = DIFFICULTIES[game.difficulty]
    hint = f"search depth {value}" if kind == "depth" else f"{value:g} s per move"
    screen.blit(small.render(hint, True, MUTED), (x, 545))

    if game.is_over() and anim is None:      # result banner
        banner = pygame.Surface((BOARD_W, 110), pygame.SRCALPHA)
        banner.fill((0, 0, 0, 170))
        screen.blit(banner, (0, TOP + ROWS * CELL // 2 - 55))
        msg = big.render(game.status_text(), True, TEXT)
        screen.blit(msg, msg.get_rect(center=(BOARD_W // 2, TOP + ROWS * CELL // 2 - 12)))
        sub = small.render("Press N for a new game", True, MUTED)
        screen.blit(sub, sub.get_rect(center=(BOARD_W // 2, TOP + ROWS * CELL // 2 + 28)))


def handle_action(game: Game, name: str) -> bool:
    """Apply a button/key action. Returns True when the board was reset (drop any animation)."""
    if name in DIFFICULTIES:
        game.set_difficulty(name)
        return False
    if name == "switch":
        game.switch_side()
    elif name == "new":
        game.new_game()
    return True


KEYS = {pygame.K_1: "easy", pygame.K_2: "medium", pygame.K_3: "hard",
        pygame.K_s: "switch", pygame.K_n: "new"}


def main(smoke: bool = False) -> None:
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Connect Four vs AI")
    fonts = (pygame.font.Font(None, 44), pygame.font.Font(None, 32), pygame.font.Font(None, 24))

    if smoke:
        game = Game(seed=0)
        draw(screen, fonts, game, 3, None)                         # start
        game.switch_side()
        draw(screen, fonts, game, None, None)                      # AI to move
        game.ai_move()
        draw(screen, fonts, game, 2, FallAnim(game))               # after AI move, mid-animation
        game.new_game(RED)
        for c in "001122":
            game.board.drop(int(c))
        assert game.human_move(3)
        draw(screen, fonts, game, None, None)                      # game over
        pygame.quit()
        print("ui.py smoke OK")
        return

    game = Game()
    clock = pygame.time.Clock()
    anim = None
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN and event.key in KEYS:
                if handle_action(game, KEYS[event.key]):
                    anim = None
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                hit = next((n for n, r in BUTTONS.items() if r.collidepoint(event.pos)), None)
                if hit is not None:
                    if handle_action(game, hit):
                        anim = None
                elif anim is None:
                    col = col_from_x(event.pos[0])
                    if col is not None and game.human_move(col):
                        anim = FallAnim(game)

        if anim is not None and anim.done():
            anim = None
        mx = pygame.mouse.get_pos()[0]
        draw(screen, fonts, game, col_from_x(mx), anim)
        pygame.display.flip()

        if anim is None and not game.is_over() and not game.is_human_turn():
            game.ai_move()            # the frame above already shows "AI thinking..."
            pygame.event.pump()
            anim = FallAnim(game)
        clock.tick(60)


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
