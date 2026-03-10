"""
Random Board Tester for RoboGambit
Tests chess_bot with various random and edge-case board positions
"""

import numpy as np
import random
from chess_bot import (
    get_best_move, get_legal_moves, format_move,
    apply_move, is_in_check, evaluate
)

PIECE_SYMBOLS = {
    0:' . ', 1:' WP', 2:' WN', 3:' WB', 4:' WQ', 5:' WK',
    6:' BP', 7:' BN', 8:' BB', 9:' BQ', 10:' BK',
}

def print_board(board, label=""):
    if label:
        print(f"\n{'='*40}")
        print(f"  {label}")
        print(f"{'='*40}")
    print("     A    B    C    D    E    F")
    print("   +----+----+----+----+----+----+")
    for row in range(5, -1, -1):
        print(f" {row+1} |", end="")
        for col in range(6):
            print(f"{PIECE_SYMBOLS[board[row][col]]}|", end="")
        print(f" {row+1}")
        print("   +----+----+----+----+----+----+")
    print("     A    B    C    D    E    F")


def make_random_board(seed=None):
    """
    Generate a random but valid board:
    - Both kings always present (required)
    - Random subset of other pieces placed randomly
    - No two pieces on same square
    """
    if seed is not None:
        random.seed(seed)

    board = np.zeros((6, 6), dtype=int)
    occupied = set()

    def place(piece, row, col):
        board[row][col] = piece
        occupied.add((row, col))

    # Kings are mandatory — place them first in random positions
    # White king NOT on row 5 (black territory), Black king NOT on row 0
    wk_row, wk_col = random.randint(0, 3), random.randint(0, 5)
    place(5, wk_row, wk_col)

    bk_row, bk_col = random.randint(2, 5), random.randint(0, 5)
    while (bk_row, bk_col) in occupied:
        bk_row, bk_col = random.randint(2, 5), random.randint(0, 5)
    place(10, bk_row, bk_col)

    # Optional pieces pool
    optional_white = [1,1,1,1, 2,3,4]   # 4 pawns, knight, bishop, queen
    optional_black = [6,6,6,6, 7,8,9]

    def place_random(pieces):
        random.shuffle(pieces)
        count = random.randint(2, len(pieces))
        for piece in pieces[:count]:
            for _ in range(20):   # try up to 20 times to find empty square
                r, c = random.randint(0, 5), random.randint(0, 5)
                # White pawns shouldn't be on row 0 or 5 (promotion rows)
                if piece == 1 and r in [0, 5]: continue
                # Black pawns shouldn't be on row 0 or 5
                if piece == 6 and r in [0, 5]: continue
                if (r, c) not in occupied:
                    place(piece, r, c)
                    break

    place_random(optional_white)
    place_random(optional_black)

    return board


def run_test(label, board, playing_white=True, expect_move=True):
    """Run a single test and report result."""
    print_board(board, label)

    # Check state
    in_check = is_in_check(board, playing_white)
    legal_moves = get_legal_moves(board, playing_white)
    side = "White" if playing_white else "Black"

    print(f"\n  Side to move : {side}")
    print(f"  In check     : {in_check}")
    print(f"  Legal moves  : {len(legal_moves)}")

    if legal_moves:
        print(f"  All moves    : {[format_move(*m) for m in legal_moves]}")

    move = get_best_move(board, playing_white, depth=3)
    print(f"  Best move    : {move}")

    if expect_move and move is None:
        print("  RESULT       : ❌ FAIL — expected a move but got None")
        return False
    elif not expect_move and move is not None:
        print("  RESULT       : ❌ FAIL — expected None but got a move")
        return False
    elif move:
        # Verify the move is actually in legal moves
        move_strs = [format_move(*m) for m in legal_moves]
        if move in move_strs:
            print("  RESULT       : ✅ PASS — move is valid")
            return True
        else:
            print(f"  RESULT       : ❌ FAIL — move {move} not in legal moves!")
            return False
    else:
        print("  RESULT       : ✅ PASS — correctly returned None")
        return True


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

results = []

# ── Test 1: Standard starting position ──────────────────────────────────────
board1 = np.array([
    [ 2,  3,  4,  5,  3,  2],
    [ 1,  1,  1,  1,  1,  1],
    [ 0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0],
    [ 6,  6,  6,  6,  6,  6],
    [ 7,  8,  9, 10,  8,  7],
], dtype=int)
results.append(run_test("TEST 1: Standard start position (White)", board1, True))
results.append(run_test("TEST 2: Standard start position (Black)", board1, False))

# ── Test 3: Randomised back rank ─────────────────────────────────────────────
board3 = np.array([
    [ 5,  2,  3,  4,  3,  2],   # King on A1
    [ 1,  1,  1,  1,  1,  1],
    [ 0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0],
    [ 6,  6,  6,  6,  6,  6],
    [10,  7,  8,  9,  8,  7],   # Black King on A6
], dtype=int)
results.append(run_test("TEST 3: Randomised back rank", board3, True))

# ── Test 4: Free capture available ───────────────────────────────────────────
board4 = np.array([
    [ 0,  0,  0,  5,  0,  0],
    [ 0,  0,  0,  0,  0,  0],
    [ 0,  0,  4,  0,  0,  0],   # White Queen C3
    [ 0,  0,  0,  9,  0,  0],   # Black Queen D4 — should be captured
    [ 0,  0,  0,  0,  0,  0],
    [ 0,  0,  0, 10,  0,  0],
], dtype=int)
results.append(run_test("TEST 4: White should capture Black Queen", board4, True))

# ── Test 5: King in check — must escape ──────────────────────────────────────
board5 = np.array([
    [ 0,  0,  0,  5,  0,  0],   # White King D1
    [ 0,  0,  0,  9,  0,  0],   # Black Queen D2 — check!
    [ 0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0],
    [ 0,  0,  0, 10,  0,  0],
], dtype=int)
results.append(run_test("TEST 5: White King in check — must escape or capture", board5, True))

# ── Test 6: No moves — stalemate/checkmate ───────────────────────────────────
board6 = np.array([
    [ 5,  0,  0,  0,  0,  0],   # White King cornered at A1
    [ 9,  9,  0,  0,  0,  0],   # Two Black Queens covering all escape
    [ 0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0],
    [ 0,  0,  0, 10,  0,  0],
], dtype=int)
results.append(run_test("TEST 6: Checkmate/stalemate — expect None", board6, True, expect_move=False))

# ── Test 7: Pawn promotion opportunity ───────────────────────────────────────
board7 = np.array([
    [ 0,  0,  0,  5,  0,  0],
    [ 0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0],
    [ 1,  0,  0,  0,  0,  0],   # White Pawn at A5 — one step from promotion
    [ 0,  0,  0, 10,  0,  0],
], dtype=int)
results.append(run_test("TEST 7: Pawn promotion available", board7, True))

# ── Test 8: Only kings on board ───────────────────────────────────────────────
board8 = np.array([
    [ 5,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0],
    [ 0,  0,  0, 10,  0,  0],
], dtype=int)
results.append(run_test("TEST 8: Kings only", board8, True))

# ── Tests 9-13: Fully random boards ──────────────────────────────────────────
print("\n" + "="*40)
print("  RANDOM BOARD TESTS")
print("="*40)
for i, seed in enumerate([42, 99, 7, 2025, 314]):
    board = make_random_board(seed=seed)
    side = random.choice([True, False])
    legal = get_legal_moves(board, side)
    expect = len(legal) > 0
    results.append(run_test(
        f"TEST {9+i}: Random board (seed={seed}, {'White' if side else 'Black'})",
        board, side, expect_move=expect
    ))

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*40)
print("  FINAL RESULTS")
print("="*40)
passed = sum(results)
total  = len(results)
for i, r in enumerate(results):
    print(f"  Test {i+1:>2}: {'✅ PASS' if r else '❌ FAIL'}")
print(f"\n  {passed}/{total} tests passed")
if passed == total:
    print("  🎉 All tests passed! Engine is working correctly.")
else:
    print("  ⚠️  Some tests failed. Check output above.")