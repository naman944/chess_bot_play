"""
Complete Game Runner for RoboGambit
Run: python play_game.py
"""

import numpy as np
import time

# IMPORT strictly from the logic engine
from chess_bot import (
    get_best_move, 
    get_legal_moves, 
    format_move, 
    apply_move, 
    is_in_check
)

# ---------------------------------------------------------------------------
# Board Display
# ---------------------------------------------------------------------------

PIECE_SYMBOLS = {
    0:  ' . ',
    1:  ' WP',   2:  ' WN',   3:  ' WB',   4:  ' WQ',   5:  ' WK',
    6:  ' BP',   7:  ' BN',   8:  ' BB',   9:  ' BQ',  10:  ' BK',
}

def print_board(board):
    print("\n     A    B    C    D    E    F")
    print("   +----+----+----+----+----+----+")
    for row in range(5, -1, -1):   
        print(f" {row+1} |", end="")
        for col in range(6):
            print(f"{PIECE_SYMBOLS[board[row][col]]}|", end="")
        print(f" {row+1}")
        print("   +----+----+----+----+----+----+")
    print("     A    B    C    D    E    F\n")

# ---------------------------------------------------------------------------
# Game state checks
# ---------------------------------------------------------------------------

def is_game_over(board):
    """Returns (True, winner) or (False, None) using strict chess rules."""
    white_moves = get_legal_moves(board, playing_white=True)
    black_moves = get_legal_moves(board, playing_white=False)

    if not white_moves:
        if is_in_check(board, playing_white=True):
            return True, "Black (Checkmate)"
        return True, "Draw (Stalemate)"
        
    if not black_moves:
        if is_in_check(board, playing_white=False):
            return True, "White (Checkmate)"
        return True, "Draw (Stalemate)"

    return False, None

# ---------------------------------------------------------------------------
# Game modes
# ---------------------------------------------------------------------------

def engine_vs_engine(depth_white=3, depth_black=3, max_moves=100, delay=0.5):
    """Watch two engines play against each other."""
    board = np.array([
        [ 2,  3,  4,  5,  3,  2],
        [ 1,  1,  1,  1,  1,  1],
        [ 0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0],
        [ 6,  6,  6,  6,  6,  6],
        [ 7,  8,  9, 10,  8,  7],
    ], dtype=int)

    playing_white = True
    move_count = 0
    move_history = []

    print("=" * 50)
    print("       ENGINE vs ENGINE")
    print(f"  White depth: {depth_white}  |  Black depth: {depth_black}")
    print("=" * 50)
    print_board(board)

    while move_count < max_moves:
        game_over, winner = is_game_over(board)
        if game_over:
            print(f"\n🏆 Game Over! Result: {winner}")
            break

        side = "White" if playing_white else "Black"
        depth  = depth_white if playing_white else depth_black

        print(f"Move {move_count + 1} — {side} thinking (depth={depth})...")
        move_str = get_best_move(board, playing_white, depth)

        if move_str is None:
            break

        # Apply move via matching the string
        all_moves = get_legal_moves(board, playing_white)
        for m in all_moves:
            if format_move(*m) == move_str:
                board = apply_move(board, *m)
                break

        move_history.append(f"{move_count+1}. [{side}] {move_str}")
        print(f"  → {side} plays: {move_str}")
        print_board(board)
        time.sleep(delay)

        playing_white = not playing_white
        move_count += 1
    else:
        print(f"\nDraw — reached {max_moves} move limit.")

    print("\n--- MOVE HISTORY ---")
    for entry in move_history:
        print(entry)

def human_vs_engine(human_white=True, engine_depth=3):
    """Play against the engine."""
    board = np.array([
        [ 5,  3,  2,  4,  3,  2],
        [ 1,  1,  1,  1,  1,  1],
        [ 0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0],
        [ 6,  6,  6,  6,  6,  6],
        [ 10,  8,  7, 9,  8,  7],
    ], dtype=int)

    playing_white = True

    print("=" * 50)
    print("       HUMAN vs ENGINE")
    print(f"  You are: {'White' if human_white else 'Black'}")
    print("  Move format: e.g.  A2->A3  (or A5->A6=4 for promotion)")
    print("  Type 'quit' to exit")
    print("=" * 50)
    print_board(board)

    while True:
        game_over, winner = is_game_over(board)
        if game_over:
            print(f"\n🏆  Game Over! Result: {winner}")
            break

        side = "White" if playing_white else "Black"
        is_human_turn = (playing_white == human_white)

        if is_human_turn:
            all_moves = get_legal_moves(board, playing_white)
            valid_moves = [format_move(*m) for m in all_moves]

            print(f"\nYour turn ({side})")
            print(f"Legal moves: {', '.join(valid_moves)}")

            while True:
                user_input = input("Enter your move: ").strip().upper()
                if user_input == 'QUIT':
                    print("Thanks for playing!")
                    return

                matched = None
                for m in all_moves:
                    move_str = format_move(*m)
                    # Support entering "1:A2->A3" or just "A2->A3"
                    if user_input == move_str or user_input == move_str.split(':')[1]:
                        matched = m
                        break

                if matched:
                    board = apply_move(board, *matched)
                    print(f"  → You played: {format_move(*matched)}")
                    break
                else:
                    print(f"  Invalid move. Try again.")

        else:
            print(f"\nEngine ({side}) thinking...")
            move_str = get_best_move(board, playing_white, engine_depth)

            if move_str is None:
                break

            all_moves = get_legal_moves(board, playing_white)
            for m in all_moves:
                if format_move(*m) == move_str:
                    board = apply_move(board, *m)
                    break

            print(f"  → Engine plays: {move_str}")

        print_board(board)
        playing_white = not playing_white

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\nRoboGambit 2025-26")
    print("==================")
    print("1. Engine vs Engine")
    print("2. Human vs Engine (You = White)")
    print("3. Human vs Engine (You = Black)")
    choice = input("\nChoose mode (1/2/3): ").strip()

    if choice == '1':
        engine_vs_engine(depth_white=3, depth_black=3, delay=0.5)
    elif choice == '2':
        human_vs_engine(human_white=True, engine_depth=3)
    elif choice == '3':
        human_vs_engine(human_white=False, engine_depth=3)
    else:
        print("Invalid choice.")