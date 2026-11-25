# Connect Four vs AI

Play Connect Four against an AI, and compare how different search algorithms play.
This is a ~3-week solo course project, presented as a live demo.

- **AI part (done):** game rules, minimax with alpha-beta pruning, Monte Carlo Tree Search (MCTS), experiments with graphs, and a Pygame UI.
- **Computer-vision part (planned):** read a real paper board from a camera and feed it to the same AI.

## Features

- **Search algorithms**
  - Minimax (negamax), with or without alpha-beta pruning. Columns are tried centre-first.
  - Iterative deepening with a time limit per move.
  - MCTS (UCT, c = √2, random rollouts).
  - A random-move baseline.
- **Two board heuristics**
  - **H1:** scores every 4-cell window, plus a bonus for pieces in the centre column.
  - **H2:** H1, but a three-in-a-row that can be completed right now counts more than one whose empty cell is still floating.
- **Game UI (Pygame)**
  - A translucent piece follows the mouse; pieces fall with an animation.
  - Three difficulty levels. The player can choose red or yellow (red always moves first).
  - A side panel shows the AI's last move: column, nodes searched, time (ms) and depth.
- **Experiments** that write CSV files and graphs: nodes and time per depth, win rates over paired games, and minimax vs MCTS at equal time.

## Requirements

- Python 3.14 (tested on Windows 11)
- `matplotlib` and `pygame-ce` (see `requirements.txt`)

`pygame-ce` is the community edition of pygame. It is used because pygame has no wheel for Python 3.14 yet. The code still does `import pygame`.

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Play

```bash
python ui.py
```

| Key | Action |
|---|---|
| Mouse click | Drop a piece in that column |
| `1` / `2` / `3` | Easy / Medium / Hard |
| `S` | Switch sides and start a new game |
| `N` | New game |
| `Esc` | Quit |

| Level | AI search |
|---|---|
| Easy | Alpha-beta, depth 2 |
| Medium | Alpha-beta, depth 4 |
| Hard | Iterative deepening, 1 second per move (reaches depth ~9) |

The side-panel buttons do the same as the keys. On-screen text is in English, because pygame's default font has no Thai glyphs.

## Run the experiments

```bash
python experiments.py nodes     # nodes and time per depth, minimax vs alpha-beta (< 1 min)
python experiments.py matches   # 5 matches x 98 games, win-rate graph (~2 min)
python experiments.py budget    # minimax vs MCTS at 0.1 s and 0.5 s per move (~25 min)
python experiments.py all       # all of the above
```

Output goes to `results/`, which is not committed. Each run overwrites the previous one.

Graph labels are in Thai, for the presentation. Matplotlib picks the first installed font from Tahoma, Leelawadee UI or DejaVu Sans; without a Thai font the labels show as boxes.

| File | Content |
|---|---|
| `nodes.csv`, `nodes.png`, `time.png` | Mean nodes and time per move at each depth |
| `matches.csv`, `winrate.png` | Every game of every match, and the score of side A |
| `budget.csv`, `budget.png` | Minimax vs MCTS score, mean depth and MCTS iterations |

In `winrate.png` and `budget.png`, each bar has an approximate 95% (Wilson) interval. Bar colour gives the verdict: green = ahead, orange = behind, grey = interval crosses 50%, so no clear winner.

### How the matches are run

- **Openings:** every game starts from one of the 49 possible two-move openings. All 49 are used, none are random.
- **Paired games:** each opening is played twice, so each AI plays red (moves first) once.
- So one match is 98 games. A win scores 1, a draw 0.5.
- **Tie-breaks:** in experiments, ties between equal moves go to the most central column, so runs repeat exactly. The only exception is the MCTS budget run, which stops on wall-clock time. The UI breaks ties at random, so the AI does not play the same game every time.

## Results so far

Node counts are means over 10 of the 49 two-move openings. Times are single runs.

| Depth | Minimax nodes | Alpha-beta nodes | Alpha-beta time |
|---|---|---|---|
| 4 | 2,801 | 234 | 3 ms |
| 6 | 135,534 | 2,368 (57x fewer) | 31 ms |
| 8 | — | 22,130 | 276 ms |

Score of side A over 98 paired games:

| Match | Score of A |
|---|---|
| depth 2 vs random | 100% |
| depth 4 vs depth 2 | 57.1% |
| depth 6 vs depth 4 | 61.2% |
| depth 6 vs depth 2 | 85.2% |
| H2 vs H1 (depth 4) | 44.9% |

Minimax (iterative deepening) vs MCTS at equal time per move:

| Time per move | Minimax score | Mean minimax depth | Mean MCTS iterations |
|---|---|---|---|
| 0.1 s | 57.7% | 6.07 | 5,541 |
| 0.5 s | 52.0% | 7.45 | 28,447 |

**How to read these numbers.** With 98 games, the 95% confidence interval is roughly ±10 percentage points. So:

- H2 vs H1 and minimax vs MCTS at 0.5 s show no significant difference.
- Do not read 44.9% as "H2 is worse".
- Both minimax and MCTS here are plain versions: no transposition table, and no tree reuse between moves.

## Project structure

| File | Role |
|---|---|
| `game.py` | Board (6x7 list, `drop`/`undo`), legal moves, win/draw detection |
| `ai_minimax.py` | H1, H2, minimax / alpha-beta search, iterative deepening, random tie-break |
| `ai_mcts.py` | MCTS (UCT) |
| `game_controller.py` | Game flow for the UI (turns, difficulty, AI move stats). Has no pygame import, so it can be tested headless. |
| `ui.py` | Pygame screen |
| `experiments.py` | Experiments, CSV output and graphs |
| `GLOSSARY.md` | Project vocabulary (Board, Move, Depth, Opening, Paired game, …) |
| `connect-four-project-summary.md` | Full project plan, in Thai |
| `connect-four-ai-slides.pptx` | Presentation slides for the demo |
| `docs/superpowers/plans/` | Step-by-step implementation plans for each round |

## Tests

Each module checks itself with plain `assert`s. No test framework is needed.

```bash
python game.py
python ai_minimax.py
python ai_mcts.py
python game_controller.py
python experiments.py --check
SDL_VIDEODRIVER=dummy python ui.py --smoke   # renders each UI state once, no window
```

Each command prints `... OK` when it passes.

## Roadmap

- **Camera input (computer-vision part):**
  - Find the paper board from four coloured corner stickers, warp it to a flat view, and read red/yellow cells by HSV colour.
  - Vote over several frames, and reject any read that is not exactly one legal move.
  - Measure accuracy on a held-out set of photos.
- **Optional:** smarter MCTS rollouts (take a winning move, block an immediate loss).
