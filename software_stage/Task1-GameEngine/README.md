# RoboGambit 2025-26 — Autonomous Chess Engine

**Task 1 submission** | Aries & Robotics Club, IIT Delhi

---

## Overview

`game.py` is the autonomous game engine for RoboGambit — a 6×6 chess-inspired competition where a robotic system must perceive the board, compute the best move, and physically execute it. The engine is built around a classical **Minimax search** enhanced with several modern pruning and heuristic techniques that dramatically reduce the number of nodes evaluated without sacrificing move quality.

---

## Board Representation

The board is a **6×6 NumPy integer array**. Each cell holds a piece ID or `0` for empty.

| ID | Piece         | ID | Piece        |
|----|---------------|----|--------------|
| 0  | Empty         | 6  | Black Pawn   |
| 1  | White Pawn    | 7  | Black Knight |
| 2  | White Knight  | 8  | Black Bishop |
| 3  | White Bishop  | 9  | Black Queen  |
| 4  | White Queen   | 10 | Black King   |
| 5  | White King    |    |              |

Coordinates: `A1 = [0][0]` (bottom-left), columns `A–F` left to right, rows `1–6` bottom to top.

---

## Move Format

```
<piece_id>:<source_cell>-><target_cell>
```
**Example:** `1:B2->B3` — White Pawn moves from B2 to B3.

**Promotion (compulsory on reaching the last rank):**
```
<piece_id>:<source_cell>-><target_cell>=<promoted_piece_id>
```
**Example:** `1:A5->A6=4` — White Pawn promotes to Queen.

> Promotion is only permitted to a piece type that has **already been captured** (e.g. cannot gain a second Queen if the original is still alive).

---

## Primary Interface

```python
get_best_move(board: np.ndarray, playing_white: bool = True) -> Optional[str]
```

Pass the current 6×6 board and the side to move. Returns the best move string, or `None` if no legal moves exist (checkmate / stalemate).

**Utility functions:**
```python
reset_game_state()          # Call at the start of every new game
set_time_remaining(seconds) # Update the clock before each call
```

---

## Search Algorithm

### Iterative Deepening Minimax
The engine searches from depth 1 up to a dynamically chosen maximum. Each completed depth seeds move ordering for the next, so if time runs out the best move from the deepest completed search is always available.

### Alpha-Beta Pruning
Standard two-player alpha-beta cuts branches that cannot possibly influence the final result, reducing the effective branching factor significantly.

### Null-Move Pruning (R = 2)
Before searching child nodes, the engine grants the opponent a free move. If they still cannot beat beta — the position is already so good that a full search is unnecessary and the node is pruned immediately. Applied at depths ≥ 3, skipped in check and endgame positions to avoid zugzwang errors.

### Late Move Reduction (LMR)
Quiet moves (non-captures, non-promotions) searched 4th or later at depth ≥ 3 are initially explored at a reduced depth. If the reduced search beats alpha, a full-depth re-search is triggered. This typically doubles the number of positions searched per second.

### Transposition Table (Zobrist Hashing)
Every position is fingerprinted with a 64-bit Zobrist hash. The table stores `(depth, score, flag, best_move)` per position. On revisiting a position the engine retrieves the stored result directly, avoiding redundant computation. The stored best move is also used to front-sort moves for better ordering in future searches.

---

## Move Ordering

Good move ordering is critical — it maximises the number of alpha-beta cutoffs. Moves are scored and sorted as follows, best first:

| Priority | Technique | Notes |
|---|---|---|
| 1st | **Transposition table move** | Best move from a prior search of this position |
| 2nd | **Promotions** | Always searched near the top |
| 3rd | **MVV-LVA captures** | Captures sorted by victim value minus attacker value |
| 4th | **Killer moves** | Two quiet moves per depth that recently caused cutoffs |
| 5th | **History heuristic** | Quiet moves weighted by how often they caused cutoffs (scaled by depth²) |
| 6th | **Centre bonus** | Moves landing on central squares D3–E4 |

---

## Board Evaluation — 10 Factors

The static evaluator scores positions from White's perspective (positive = White ahead).

| # | Factor | Description |
|---|--------|-------------|
| 1 | **Material** | Summed piece values: Pawn 100, Knight 300, Bishop 320, Queen 900, King 20000 |
| 2 | **Piece-Square Tables** | Positional bonuses per piece per square — encourages natural development |
| 3 | **Mobility** | Pseudo-legal move count difference; more options = better position |
| 4 | **Centre control** | Bonus for pieces occupying the four central squares |
| 5 | **King safety** | Pawn shield on the back rank; penalises exposed kings |
| 6 | **Passed pawns** | Pawns with no opposing pawn blocking their path to promotion |
| 7 | **Pawn structure** | Penalties for doubled and isolated pawns |
| 8 | **Bishop pair** | Bonus for owning both bishops simultaneously |
| 9 | **Endgame king activity** | King PST inverted when material is low — king becomes an attacker |
| 10 | **Check bonus** | Bonus for giving check, penalty for being in check |

---

## Quiescence Search

At leaf nodes the engine does not stop immediately — it continues searching **captures and promotions only** for up to 4 additional plies. This prevents the *horizon effect* where a piece hanging just beyond the search depth is missed.

---

## Dynamic Depth

Search depth is chosen automatically based on both the number of pieces remaining and the clock:

|                | < 1 min | 1–3 min | 3–7 min | 7–15 min |
|----------------|---------|---------|---------|----------|
| **< 5 pieces** | 3       | 5       | 6       | **7**    |
| **5–9 pieces** | 3       | 4       | 5       | 6        |
| **10–19 pcs**  | 2       | 3       | 4       | 5        |
| **20+ pcs**    | 2       | 3       | 3       | 4        |

Endgames get the deepest search; busy opening positions are kept shallow to respect time.

---

## Repetition Penalty

The engine records a hash of every position it visits. Moves leading to a position seen before are penalised by **150 × visit count** in the score, discouraging draw-by-repetition and encouraging decisive play.

---

## Dependencies

```
numpy
```
No other external libraries required. Python 3.8+.

---

## Quick Start

```python
import numpy as np
from game import get_best_move, reset_game_state, set_time_remaining

board = np.array([
    [ 2,  3,  4,  5,  3,  2],
    [ 1,  1,  1,  1,  1,  1],
    [ 0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0],
    [ 6,  6,  6,  6,  6,  6],
    [ 7,  8,  9, 10,  8,  7],
], dtype=int)

reset_game_state()
set_time_remaining(900)          # 15 minutes on the clock

move = get_best_move(board, playing_white=True)
print(move)   # e.g. "1:C2->C3"
```
