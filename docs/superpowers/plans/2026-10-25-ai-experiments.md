# AI Experiments (รอบ 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** เพิ่ม H2, AI สุ่ม และ MCTS แล้วให้ AI แข่งกันเอง (49 opening × 2 ฝั่ง = 98 เกมต่อคู่) เพื่อได้กราฟอัตราชนะ, H1 เทียบ H2 และ minimax เทียบ MCTS ที่เวลาเท่ากัน

**Architecture:** เพิ่ม `h2` ใน `ai_minimax.py` และเขียน `ai_mcts.py` เป็นโมดูลใหม่ที่ใช้ `Board.drop/undo` ตัวเดิม ฝั่ง `experiments.py` จะมองผู้เล่นทุกแบบเป็น "agent" คือฟังก์ชันที่รับ `Board` แล้วคืนคอลัมน์ ฟังก์ชัน `play_game` / `match` ใช้กับ agent ได้ทุกชนิด ผลเขียนลง CSV แล้ววาดกราฟ

**Tech Stack:** Python 3.14, matplotlib, stdlib (`random`, `math`, `time`, `csv`)

**Spec:** `connect-four-project-summary.md` (ส่วนที่ 1) และ `GLOSSARY.md`
ต่อยอดจาก `docs/superpowers/plans/2026-10-25-ai-core.md` ซึ่ง merge แล้ว

## Global Constraints

- ใช้ข้อกำหนดทั้งหมดของแผนรอบแรกต่อ: `grid[row][col]` แถว 0 อยู่ล่าง, ลำดับคอลัมน์ `CENTER_ORDER`, เทสต์ด้วย `assert` ใน `_self_check()` ไม่ใช้ pytest, รันด้วย `.venv/Scripts/python`
- **Opening:** ใช้ครบ 49 แบบ ไม่สุ่ม แต่ละ opening เล่นสองเกมสลับฝั่ง (Paired game) ชนะได้ 1 แต้ม เสมอได้ 0.5
- **การเลือกเมื่อคะแนนเท่ากันในการทดลอง:** เลือกช่องที่ใกล้กลางที่สุด (พฤติกรรมเดิมของ `search`) ส่วนการสุ่มใช้เฉพาะใน UI ซึ่งอยู่ในแผนรอบ 3
- **MCTS:** UCT ค่า c = √2, rollout สุ่มล้วน, เลือกคอลัมน์ที่ถูกเยี่ยมมากที่สุด
- **ความสุ่ม:** ทุกแหล่งรับ `random.Random(seed)` ผลจึงทำซ้ำได้
- ห้ามติดตั้งไลบรารีเพิ่ม

## Review Focus

1. **MCTS เจอกระดานที่จบเกมแล้ว หรือเหลือช่องว่างช่องเดียว:** กระดานจบแล้วต้องคืน `move=None` ส่วนกระดานที่เหลือช่องเดียวต้องคืนช่องนั้น และไม่ crash (ทดสอบใน Task 2)
2. **MCTS ทำงานเสร็จหรือหมดเวลา:** `board.moves` ต้องกลับมาเหมือนก่อนเรียก (ทดสอบใน Task 2)
3. **เกมเสมอในการแข่ง:** ต้องนับคนละ 0.5 แต้ม และแต้มรวมของทั้งสองฝั่งต้องเท่ากับจำนวนเกม (ทดสอบใน Task 3)
4. **ฝั่งที่เดินก่อน:** agent A ต้องได้เป็นแดงในเกมแรกของแต่ละ opening และเป็นเหลืองในเกมที่สอง ถ้าสลับผิด อัตราชนะจะเอียงเพราะฝั่งที่เดินก่อนได้เปรียบ (ทดสอบใน Task 3)
5. **ตั้งงบเวลาเล็กมาก (0.0 วินาที):** MCTS ต้องทำอย่างน้อย 1 iteration และคืนคอลัมน์ที่ลงได้ (ทดสอบใน Task 2)

---

### Task 1: H2 + แก้ค่า depth ตอนกระดานจบเกม

**Files:**
- Modify: `ai_minimax.py`

**Interfaces:**
- Consumes: `_WINDOWS`, `h1`, `_terminal_score` ที่มีอยู่ใน `ai_minimax.py`
- Produces: `h2(board: Board, player: int) -> int`

ค่าที่กำหนดไว้สำหรับ H2:
- หน้าต่างที่ช่องว่างช่องเดียวเป็นช่องที่ลงได้ทันที (`board.heights[col] == row`) ถือว่า **playable** ไม่เช่นนั้นถือว่า **floating**

| สภาพหน้าต่าง | playable | floating |
|---|---|---|
| ของเรา 3 + ว่าง 1 | **+8** | **+4** |
| ของคู่แข่ง 3 + ว่าง 1 | **−8** | **−3** |

- ของเรา 2 + ว่าง 2 ยังได้ **+2** และหมากในคอลัมน์ 3 ยังได้ **+3** ต่อตัว (เหมือน H1)

แก้บั๊ก Minor 2 จากรอบแรกไปด้วย: เมื่อ `search` เจอกระดานที่จบเกมแล้ว ให้คืน `Result(None, _terminal_score(board, 0), 1, 0)` ให้ตรงกับ `iterative_deepening`

- [ ] **Step 1: เพิ่ม check ต่อท้าย `_self_check()`**

```python
    for pos in ("", "3", "33", "3243"):                      # ไม่มีหน้าต่าง 3 ตัว → เท่ากับ H1
        assert h2(Board.from_moves(pos), RED) == h1(Board.from_moves(pos), RED)
    b = Board.from_moves("061625")                           # แดงมี 3 ตัวที่ลงได้ทันที
    assert h2(b, RED) - h1(b, RED) == 3
    b = Board.from_moves("6001122")                          # แดง 3 ตัวลอยอยู่ เหลือง 3 ตัวลงได้ทันที
    assert h2(b, RED) - h1(b, RED) == -5
    assert search(Board.from_moves("0101010"), 3) == Result(None, -WIN_SCORE, 1, 0)
    assert search(Board.from_moves("001122"), 4, heuristic=h2).move == 3
```

- [ ] **Step 2: รันเพื่อดูว่าไม่ผ่าน**

Run: `.venv/Scripts/python ai_minimax.py`
Expected: FAIL (`NameError` เพราะยังไม่มี `h2`)

- [ ] **Step 3: เขียน `h2` และแก้ทางออกของ `search` เมื่อกระดานจบเกมแล้ว**

- [ ] **Step 4: รันเพื่อดูว่าผ่าน**

Run: `.venv/Scripts/python ai_minimax.py`
Expected: `ai_minimax.py OK`

- [ ] **Step 5: Commit**

```bash
git add ai_minimax.py
git commit -m "feat: heuristic H2; consistent depth for finished boards"
```

---

### Task 2: MCTS

**Files:**
- Create: `ai_mcts.py`

**Interfaces:**
- Consumes: `Board`, `RED`, `YELLOW` จาก `game.py`
- Produces (`ai_mcts.py`):
  - `class MctsResult(NamedTuple)`: `move: int | None`, `iterations: int`, `win_rate: float`
    - `win_rate` คืออัตราชนะของคอลัมน์ที่เลือก มองจากฝั่งที่กำลังจะเดิน
  - `mcts(board: Board, time_limit: float | None = None, iterations: int | None = None, rng: random.Random | None = None, c: float = math.sqrt(2)) -> MctsResult`

กติกา:
- ต้องระบุ `time_limit` หรือ `iterations` อย่างใดอย่างหนึ่งเท่านั้น ถ้าระบุทั้งคู่หรือไม่ระบุเลย ให้ `raise ValueError`
- ถ้าไม่ส่ง `rng` มา ให้ใช้ `random.Random()`
- แต่ละ iteration มี 4 ขั้น ทุกขั้นทำบนกระดานตัวเดิมด้วย `drop` / `undo` และห่อด้วย `try/finally` ให้ถอนหมากคืนเสมอ
  1. **Selection:** ลงไปตาม UCT
  2. **Expansion:** เพิ่มลูกหนึ่งตัว
  3. **Rollout:** สุ่มเดินจนเกมจบ
  4. **Backprop:** ส่งผลกลับขึ้นไป
- รางวัลคิดจากมุมของฝั่งที่เดินเข้า node นั้น ชนะ = 1, เสมอ = 0.5, แพ้ = 0
- ทำอย่างน้อย 1 iteration เสมอ แม้ `time_limit=0`
- คืนคอลัมน์ที่ถูกเยี่ยมมากที่สุด ถ้าเยี่ยมเท่ากันให้เลือกตามลำดับของ `valid_moves()`
- ถ้ากระดานจบเกมแล้ว ให้คืน `MctsResult(None, 0, 0.0)`

- [ ] **Step 1: เขียน self-check ที่ยังไม่ผ่านใน `ai_mcts.py`**

```python
def _self_check():
    rng = random.Random(0)
    assert mcts(Board.from_moves("001122"), iterations=2000, rng=rng).move == 3      # ชนะทันที
    assert mcts(Board.from_moves("00112"), iterations=4000, rng=rng).move == 3       # ต้องกัน
    b = Board.from_moves("3243")
    before = list(b.moves)
    r = mcts(b, time_limit=0.05, rng=rng)
    assert b.moves == before and r.move in b.valid_moves() and r.iterations >= 1
    assert mcts(Board.from_moves(""), time_limit=0.0, rng=rng).iterations >= 1
    assert mcts(Board.from_moves("0101010"), iterations=10).move is None
    last = Board.from_moves("0213465" * 6)
    last.undo()
    assert mcts(last, iterations=10).move == last.valid_moves()[0]                   # เหลือช่องเดียว
    a = mcts(Board.from_moves("33"), iterations=500, rng=random.Random(7))
    b2 = mcts(Board.from_moves("33"), iterations=500, rng=random.Random(7))
    assert a == b2                                                                    # seed เดียวกันได้ผลเดียวกัน
    for bad in ({}, {"time_limit": 1.0, "iterations": 10}):
        try:
            mcts(Board.from_moves(""), **bad)
            assert False, bad
        except ValueError:
            pass
    print("ai_mcts.py OK")
```

- [ ] **Step 2: รันเพื่อดูว่าไม่ผ่าน**

Run: `.venv/Scripts/python ai_mcts.py`
Expected: FAIL (`NameError` เพราะยังไม่มี `mcts`)

- [ ] **Step 3: เขียน `mcts` ตามกติกาด้านบน**

แต่ละ node เก็บ `children: dict[int, Node]`, `untried: list[int]`, `visits`, `wins` และ `parent` โดยสร้างใหม่ทุกครั้งที่เรียก `mcts` ไม่ต้องเก็บ tree ข้ามตา (YAGNI)

- [ ] **Step 4: รันเพื่อดูว่าผ่าน**

Run: `.venv/Scripts/python ai_mcts.py`
Expected: `ai_mcts.py OK`

ถ้าเทสต์ "ต้องกัน" ที่ 4000 iterations ยังไม่ผ่าน ห้ามแก้ค่า c ให้เพิ่ม iterations เป็น 10000 แล้วบันทึกเป็น Ruling

- [ ] **Step 5: Commit**

```bash
git add ai_mcts.py
git commit -m "feat: UCT MCTS with random rollouts"
```

---

### Task 3: Agent + การแข่ง 98 เกม + กราฟอัตราชนะ

**Files:**
- Modify: `experiments.py`

**Interfaces:**
- Consumes: `search`, `iterative_deepening`, `h1`, `h2` (ai_minimax), `mcts` (ai_mcts), `Board.from_moves`, `result()`
- Produces (`experiments.py`):
  - `RESULTS = Path(__file__).parent / "results"` แทน `Path("results")` (แก้ Minor 4 จากรอบแรก)
  - Agent คือ `Callable[[Board], int]`
    - `minimax_agent(depth: int, heuristic=h1)`
    - `id_agent(time_limit: float, stats: list | None = None)`: บันทึก `Result.depth` ลง `stats`
    - `mcts_agent(time_limit: float, seed: int, stats: list | None = None)`: บันทึก `iterations` ลง `stats`
    - `random_agent(seed: int)`
  - `play_game(red, yellow, opening: str) -> int`: คืน 1, 2 หรือ 0
  - `match(name: str, a, b, openings: list[str], writer=None) -> dict`
    - คืน dict ที่มีคีย์ `a_points`, `b_points`, `a_wins`, `b_wins`, `draws`
    - แต่ละ opening เล่นเกมแรกให้ a เป็นแดง เกมที่สองให้ b เป็นแดง
    - `writer` คือ `csv.DictWriter` ที่มีคอลัมน์ `match,opening,red,yellow,result` โดย `red`/`yellow` เป็น `"A"` หรือ `"B"`
  - `run_matches(out_csv: Path) -> list[dict]`: แข่งตามรายการด้านล่าง ใช้ opening ครบ 49 แบบ
  - `plot_matches(csv_path: Path, out_dir: Path) -> None`: วาด `winrate.png` เป็นแผนภูมิแท่งแนวนอนของเปอร์เซ็นต์แต้มของ A ในแต่ละคู่ มีเส้นอ้างอิงที่ 50% และแสดงตัวเลขบนแท่ง

รายการคู่แข่ง (A ทางซ้าย, B ทางขวา):

| name | A | B |
|---|---|---|
| `d2 vs random` | `minimax_agent(2)` | `random_agent(0)` |
| `d4 vs d2` | `minimax_agent(4)` | `minimax_agent(2)` |
| `d6 vs d4` | `minimax_agent(6)` | `minimax_agent(4)` |
| `d6 vs d2` | `minimax_agent(6)` | `minimax_agent(2)` |
| `H2 vs H1 (d4)` | `minimax_agent(4, h2)` | `minimax_agent(4, h1)` |

- [ ] **Step 1: เพิ่ม check ต่อท้าย `_self_check()`** (ก่อนบรรทัด `print`)

```python
    assert play_game(minimax_agent(2), random_agent(0), "33") in (0, 1, 2)
    first = lambda b: b.valid_moves()[0]
    log = []
    class W:                                                  # writer ปลอมที่จดแถวไว้
        def writerow(self, row): log.append(row)
    s = match("t", minimax_agent(2), first, ["00", "34"], writer=W())
    assert s["a_points"] + s["b_points"] == 4
    assert s["a_wins"] + s["b_wins"] + s["draws"] == 4
    assert [(r["opening"], r["red"]) for r in log] == [("00", "A"), ("00", "B"), ("34", "A"), ("34", "B")]
    assert match("t", minimax_agent(2), minimax_agent(2), ["33"]) == match("t", minimax_agent(2), minimax_agent(2), ["33"])
    scripted = lambda b: int(("0213465" * 6)[len(b.moves)])  # เดินตามลำดับที่จบเสมอ
    draws = match("t", scripted, scripted, ["02"])
    assert draws["draws"] == 2 and draws["a_points"] == 1.0 and draws["b_points"] == 1.0
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "m.csv"
        p.write_text("match,opening,red,yellow,result\nx,00,A,B,1\nx,00,B,A,0\n")
        plot_matches(p, Path(tmp))                            # A ชนะ 1 เสมอ 1 → 75%
        assert (Path(tmp) / "winrate.png").exists()
```

- [ ] **Step 2: รันเพื่อดูว่าไม่ผ่าน**

Run: `.venv/Scripts/python experiments.py --check`
Expected: FAIL (`NameError` เพราะยังไม่มี `play_game`)

- [ ] **Step 3: เขียนตาม Interfaces และเปลี่ยน CLI เป็น subcommand**

```
python experiments.py nodes | matches | budget | all
python experiments.py --check
```

ถ้าไม่ใส่ argument ให้ทำงานเหมือน `nodes` (คงพฤติกรรมเดิม)

- [ ] **Step 4: รันเพื่อดูว่าผ่าน**

Run: `.venv/Scripts/python experiments.py --check`
Expected: `experiments.py OK`

- [ ] **Step 5: รันการแข่งจริง**

Run: `.venv/Scripts/python experiments.py matches`
Expected:
- ได้ `results/matches.csv` 490 แถว (5 คู่ × 98 เกม) และ `results/winrate.png`
- `d2 vs random` ต้องได้มากกว่า 90%
- `d4 vs d2` และ `d6 vs d4` ต้องได้มากกว่า 50%
- `H2 vs H1` ได้เท่าไรก็ได้ เพราะเป็นผลการทดลอง

ถ้าข้อที่คาดไว้ไม่เป็นตามนั้น ให้รายงานตัวเลข อย่าแก้ heuristic ให้ผลออกมาตรง

- [ ] **Step 6: Commit**

```bash
git add experiments.py
git commit -m "feat: paired 98-game matches and win-rate graph"
```

---

### Task 4: Minimax เทียบ MCTS ที่เวลาเท่ากัน

**Files:**
- Modify: `experiments.py`

**Interfaces:**
- Consumes: `id_agent`, `mcts_agent`, `match`, `openings` (Task 3)
- Produces:
  - `run_budget(budgets: tuple[float, ...], out_csv: Path, openings_list: list[str] | None = None) -> list[dict]`
    - เขียน CSV ที่มีคอลัมน์ `budget,minimax_points,mcts_points,draws,games,mean_id_depth,mean_mcts_iterations`
    - ถ้า `openings_list` เป็น `None` ให้ใช้ `openings()`
  - `plot_budget(csv_path: Path, out_dir: Path) -> None`
    - วาด `budget.png` สองแผงข้างกัน
      - ซ้าย: เปอร์เซ็นต์แต้มของ minimax ต่องบเวลา
      - ขวา: depth เฉลี่ยของ ID กับ iterations เฉลี่ยของ MCTS แยกแกน y
  - ตอนรันจริง `budget` ใช้ `budgets=(0.1, 0.5)`
  - seed ของ MCTS คือ `int(budget * 1000)`

- [ ] **Step 1: เพิ่ม check ต่อท้าย `_self_check()`**

```python
    with tempfile.TemporaryDirectory() as tmp:
        rows = run_budget((0.01,), Path(tmp) / "b.csv", openings_list=["33"])
        assert len(rows) == 1 and rows[0]["games"] == 2
        assert rows[0]["minimax_points"] + rows[0]["mcts_points"] == 2
        assert rows[0]["mean_id_depth"] >= 1 and rows[0]["mean_mcts_iterations"] >= 1
        plot_budget(Path(tmp) / "b.csv", Path(tmp))
        assert (Path(tmp) / "budget.png").exists()
```

- [ ] **Step 2: รันเพื่อดูว่าไม่ผ่าน**

Run: `.venv/Scripts/python experiments.py --check`
Expected: FAIL (`NameError` เพราะยังไม่มี `run_budget`)

- [ ] **Step 3: เขียน `run_budget`, `plot_budget` และ subcommand `budget`**

- [ ] **Step 4: รันเพื่อดูว่าผ่าน**

Run: `.venv/Scripts/python experiments.py --check`
Expected: `experiments.py OK`

- [ ] **Step 5: รันจริงใน background (ใช้เวลาราว 40 นาที)**

Run: `.venv/Scripts/python experiments.py budget`
Expected:
- ได้ `results/budget.csv` 2 แถว และ `results/budget.png`
- depth เฉลี่ยของ ID ที่ 0.5 วินาทีต้องมากกว่าที่ 0.1 วินาที
- จะเป็นฝั่งไหนชนะก็ได้ แต่ต้องรายงานตัวเลขตามจริง

- [ ] **Step 6: Commit**

```bash
git add experiments.py
git commit -m "feat: minimax vs MCTS at equal time budget"
```

---

### Task 5: ตรวจทั้งหมด + ล็อกเวอร์ชัน

- [ ] **Step 1: ล็อกเวอร์ชัน matplotlib** (แก้ Minor 5 จากรอบแรก)

Run: `.venv/Scripts/python -m pip freeze | grep -i "^matplotlib=="`
แล้วแทนที่บรรทัดใน `requirements.txt` ด้วยค่าที่ได้

- [ ] **Step 2: รัน self-check ทุกไฟล์**

Run: `.venv/Scripts/python game.py && .venv/Scripts/python ai_minimax.py && .venv/Scripts/python ai_mcts.py && .venv/Scripts/python experiments.py --check`
Expected: ได้ `OK` ครบทั้ง 4 บรรทัด

- [ ] **Step 3: เปิดดู `results/winrate.png` และ `results/budget.png`**

Expected: ตัวเลขบนกราฟตรงกับค่าใน CSV และชื่อคู่อ่านออกครบทุกคู่

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "chore: pin matplotlib"
```
