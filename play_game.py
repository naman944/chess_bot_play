"""
RoboGambit 2025-26 — Game Runner
Supports: Engine vs Engine | Human vs Engine | Human vs Human
submission file
"""

import numpy as np
import time
from collections import defaultdict
from chess_bot import (
    get_best_move, get_all_moves, apply_move,
    is_in_check, format_move, reset_game_state, set_time_remaining,
    EMPTY, WHITE_KING, BLACK_KING, BOARD_SIZE
)

# ---------------------------------------------------------------------------
# Board display
# ---------------------------------------------------------------------------

PIECE_SYMBOLS = {
    0:  '  . ', 1:  ' WP ', 2:  ' WN ', 3:  ' WB ',
    4:  ' WQ ', 5:  ' WK ', 6:  ' BP ', 7:  ' BN ',
    8:  ' BB ', 9:  ' BQ ', 10: ' BK ',
}

def print_board(board, last_move=None):
    """Print board with optional last-move highlight marker."""
    # Parse last move highlight squares
    highlight = set()
    if last_move:
        try:
            _, cells = last_move.split(':')
            src, dst = cells.split('->')
            highlight.add(src.upper())
            highlight.add(dst.upper())
        except Exception:
            pass

    cols = "A B C D E F"
    print()
    print(f"      {'  '.join(cols.split())}")
    print("    +" + "----+" * BOARD_SIZE)
    for row in range(BOARD_SIZE - 1, -1, -1):
        print(f"  {row+1} |", end="")
        for col in range(BOARD_SIZE):
            cell = f"{chr(65+col)}{row+1}"
            sym  = PIECE_SYMBOLS[board[row][col]]
            if cell in highlight:
                print(f"\033[43m{sym}\033[0m|", end="")
            else:
                print(f"{sym}|", end="")
        print(f" {row+1}")
        print("    +" + "----+" * BOARD_SIZE)
    print(f"      {'  '.join(cols.split())}")
    print()

def fmt_time(seconds):
    m = int(max(seconds, 0)) // 60
    s = int(max(seconds, 0)) % 60
    return f"{m:02d}:{s:02d}"

# ---------------------------------------------------------------------------
# Draw / game-over detection (independent of game.py internals)
# ---------------------------------------------------------------------------

def check_game_over(board, position_history, no_progress):
    """Returns (is_over, result_string) or (False, None)."""
    # Threefold repetition
    key = board.tobytes()
    if position_history.get(key, 0) >= 3:
        return True, "Draw — Threefold Repetition"

    # 50-move rule
    if no_progress >= 50:
        return True, "Draw — 50-Move Rule"

    white_moves = get_all_moves(board, True)
    black_moves = get_all_moves(board, False)

    if not white_moves:
        return True, ("White is Checkmated — Black Wins! 🏆"
                      if is_in_check(board, True) else "Draw — Stalemate (White)")
    if not black_moves:
        return True, ("Black is Checkmated — White Wins! 🏆"
                      if is_in_check(board, False) else "Draw — Stalemate (Black)")
    return False, None


def was_progress(old_board, move):
    """Pawn move or capture resets 50-move counter."""
    piece, _, _, dr, dc, *_ = move
    if piece in (1, 6): return True
    if old_board[dr][dc] != EMPTY: return True
    return False


def record_pos(board, history):
    key = board.tobytes()
    history[key] = history.get(key, 0) + 1


def make_initial_board():
    return np.array([
        [ 2,  3,  4,  5,  3,  2],
        [ 1,  1,  1,  1,  1,  1],
        [ 0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0],
        [ 6,  6,  6,  6,  6,  6],
        [ 7,  8,  9, 10,  8,  7],
    ], dtype=int)

# ---------------------------------------------------------------------------
# Parse human move input
# ---------------------------------------------------------------------------

def parse_human_move(user_input, board, playing_white):
    """
    Accept multiple input formats:
      A2->A3        (short)
      1:A2->A3      (with piece id)
      a2a3          (compact)
    Returns matching move tuple or None.
    """
    raw = user_input.strip().upper().replace(' ', '')
    all_moves = get_all_moves(board, playing_white)

    for m in all_moves:
        full = format_move(*m[:5])          # e.g. "1:A2->A3"
        short = full.split(':')[1]          # e.g. "A2->A3"
        compact = short.replace('->', '')   # e.g. "A2A3"
        if raw in (full, short, compact):
            return m

    return None

# ---------------------------------------------------------------------------
# ENGINE TURN
# ---------------------------------------------------------------------------

def engine_turn(board, playing_white, pos_history, move_number,
                time_remaining, label="Engine"):
    """Run engine, apply move, return (new_board, move_str, elapsed)."""
    t_start = time.time()
    reset_game_state()
    set_time_remaining(time_remaining)   # tell engine how much clock is left
    move_str = get_best_move(board, playing_white)
    elapsed  = time.time() - t_start

    if move_str is None:
        return board, None, elapsed

    all_moves = get_all_moves(board, playing_white)
    for m in all_moves:
        if format_move(*m[:5]) == move_str:
            new_board = apply_move(board, *m)
            return new_board, move_str, elapsed

    return board, None, elapsed

# ---------------------------------------------------------------------------
# HUMAN TURN
# ---------------------------------------------------------------------------

def human_turn(board, playing_white, side_name):
    """Interactive human input. Returns (new_board, move_str) or None on quit."""
    all_moves   = get_all_moves(board, playing_white)
    valid_strs  = [format_move(*m[:5]) for m in all_moves]

    while True:
        raw = input(f"  Your move ({side_name}) → ").strip()

        if raw.lower() in ('q', 'quit', 'exit'):
            return None, None

        if raw.lower() in ('m', 'moves', '?'):
            short = [s.split(':')[1] for s in valid_strs]
            print(f"  Legal moves: {', '.join(sorted(short))}")
            continue

        if raw.lower() == 'board':
            print_board(board)
            continue

        matched = parse_human_move(raw, board, playing_white)
        if matched:
            new_board = apply_move(board, *matched)
            return new_board, format_move(*matched[:5])

        print("  ❌  Invalid move. Type 'moves' to see legal moves, 'q' to quit.")

# ---------------------------------------------------------------------------
# MODE 1 — Engine vs Engine
# ---------------------------------------------------------------------------

def engine_vs_engine(time_per_side=900, delay=0.4, max_moves=150):
    print("\n" + "="*55)
    print("          ♟  ENGINE  vs  ENGINE  ♟")
    print(f"  Time per side : {time_per_side//60} min")
    print(f"  Dynamic depth : ON  |  Gambling : ON")
    print(f"  Repetition    : ON  |  50-move  : ON")
    print("="*55)

    board        = make_initial_board()
    playing_white = True
    move_count   = 0
    no_progress  = 0
    pos_history  = {}
    move_log     = []
    clocks       = {True: float(time_per_side), False: float(time_per_side)}
    last_move    = None

    record_pos(board, pos_history)
    print_board(board)

    while move_count < max_moves:
        over, result = check_game_over(board, pos_history, no_progress)
        if over:
            print(f"\n  🏁  {result}")
            break

        side  = "White ♔" if playing_white else "Black ♚"
        clock = clocks[playing_white]

        if clock <= 0:
            winner = "Black" if playing_white else "White"
            print(f"\n  ⏱️  {side} flagged!  {winner} wins on time.")
            break

        pieces = int(np.count_nonzero(board))
        print(f"  Move {move_count+1:>3}  [{side}]  "
              f"pieces={pieces}  clock={fmt_time(clock)}", end="  ", flush=True)

        new_board, move_str, elapsed = engine_turn(
            board, playing_white, pos_history, move_count, clock)

        clocks[playing_white] -= elapsed

        if move_str is None:
            print(f"\n  {side} has no moves.")
            break

        old_board   = board
        board       = new_board
        last_move   = move_str

        # Find move tuple for progress check
        all_moves = get_all_moves(old_board, playing_white)
        for m in all_moves:
            if format_move(*m[:5]) == move_str:
                no_progress = 0 if was_progress(old_board, m) else no_progress + 1
                break

        record_pos(board, pos_history)
        rep = pos_history.get(board.tobytes(), 1)
        rep_note = f"  ♻ x{rep}" if rep > 1 else ""

        print(f"→  {move_str}  ({elapsed:.2f}s){rep_note}")
        move_log.append(f"{move_count+1:>3}. [{side[:5]}] {move_str}  ({elapsed:.2f}s){rep_note}")

        print_board(board, last_move)
        time.sleep(delay)

        playing_white = not playing_white
        move_count   += 1

    else:
        print(f"\n  ⏱️  Move limit ({max_moves}) reached — Draw.")

    print(f"\n  White clock : {fmt_time(clocks[True])}")
    print(f"  Black clock : {fmt_time(clocks[False])}")
    print("\n  --- MOVE LOG ---")
    for entry in move_log:
        print(" ", entry)

# ---------------------------------------------------------------------------
# MODE 2 — Human vs Engine
# ---------------------------------------------------------------------------

def human_vs_engine(human_is_white=True, engine_time=900):
    human_side  = "White ♔" if human_is_white else "Black ♚"
    engine_side = "Black ♚" if human_is_white else "White ♔"

    print("\n" + "="*55)
    print("          ♟  HUMAN  vs  ENGINE  ♟")
    print(f"  You play      : {human_side}")
    print(f"  Engine plays  : {engine_side}")
    print("  Commands      : 'moves' | 'board' | 'q' to quit")
    print("="*55)

    board         = make_initial_board()
    playing_white = True
    move_count    = 0
    no_progress   = 0
    pos_history   = {}
    engine_clock  = float(engine_time)
    last_move     = None

    record_pos(board, pos_history)
    print_board(board, last_move)

    while True:
        over, result = check_game_over(board, pos_history, no_progress)
        if over:
            print(f"\n  🏁  {result}")
            break

        is_human = (playing_white == human_is_white)
        side      = "White ♔" if playing_white else "Black ♚"

        if is_human:
            print(f"\n  Your turn  ({side})")
            new_board, move_str = human_turn(board, playing_white, side)
            if move_str is None:
                print("  Quit. Thanks for playing!")
                return
            elapsed = 0.0
        else:
            if engine_clock <= 0:
                print("\n  ⏱️  Engine ran out of time — You win!")
                break
            print(f"\n  Engine thinking  ({side}  clock={fmt_time(engine_clock)}) ...", flush=True)
            new_board, move_str, elapsed = engine_turn(
                board, playing_white, pos_history, move_count, engine_clock)
            engine_clock -= elapsed
            if move_str is None:
                print("  Engine has no moves.")
                break
            print(f"  Engine plays : {move_str}  ({elapsed:.2f}s)")

        old_board = board
        board     = new_board
        last_move = move_str

        all_moves = get_all_moves(old_board, playing_white)
        for m in all_moves:
            if format_move(*m[:5]) == move_str:
                no_progress = 0 if was_progress(old_board, m) else no_progress + 1
                break

        record_pos(board, pos_history)
        rep = pos_history.get(board.tobytes(), 1)
        if rep > 1:
            print(f"  ♻  Position seen {rep}x")

        print_board(board, last_move)
        playing_white = not playing_white
        move_count   += 1

# ---------------------------------------------------------------------------
# MODE 3 — Human vs Human
# ---------------------------------------------------------------------------

def human_vs_human():
    print("\n" + "="*55)
    print("          ♟  HUMAN  vs  HUMAN  ♟")
    print("  Commands : 'moves' | 'board' | 'q' to quit")
    print("="*55)

    board         = make_initial_board()
    playing_white = True
    move_count    = 0
    no_progress   = 0
    pos_history   = {}
    last_move     = None

    record_pos(board, pos_history)
    print_board(board, last_move)

    while True:
        over, result = check_game_over(board, pos_history, no_progress)
        if over:
            print(f"\n  🏁  {result}")
            break

        side = "White ♔" if playing_white else "Black ♚"
        print(f"\n  Turn {move_count+1}  —  {side}")

        new_board, move_str = human_turn(board, playing_white, side)
        if move_str is None:
            print("  Quit. Thanks for playing!")
            return

        old_board = board
        board     = new_board
        last_move = move_str

        all_moves = get_all_moves(old_board, playing_white)
        for m in all_moves:
            if format_move(*m[:5]) == move_str:
                no_progress = 0 if was_progress(old_board, m) else no_progress + 1
                break

        record_pos(board, pos_history)
        rep = pos_history.get(board.tobytes(), 1)
        if rep > 1:
            print(f"  ♻  Position seen {rep}x")

        print_board(board, last_move)
        playing_white = not playing_white
        move_count   += 1

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("\n  ╔══════════════════════════════╗")
    print("  ║     RoboGambit 2025-26       ║")
    print("  ║     IIT Delhi — Task 1       ║")
    print("  ╚══════════════════════════════╝")
    print()
    print("  1.  Engine  vs  Engine")
    print("  2.  Human   vs  Engine  (You = White)")
    print("  3.  Human   vs  Engine  (You = Black)")
    print("  4.  Human   vs  Human")
    print()

    choice = input("  Choose mode (1–4): ").strip()

    if choice == '1':
        engine_vs_engine(time_per_side=900, delay=0.4)
    elif choice == '2':
        human_vs_engine(human_is_white=True)
    elif choice == '3':
        human_vs_engine(human_is_white=False)
    elif choice == '4':
        human_vs_human()
    else:
        print("  Invalid choice.")


if __name__ == "__main__":
    main()