"""
RoboGambit 2025-26 — Task 1: Autonomous Game Engine
Organised by Aries and Robotics Club, IIT Delhi

Board: 6x6 NumPy array
  - 0  : Empty cell
  - 1  : White Pawn
  - 2  : White Knight
  - 3  : White Bishop
  - 4  : White Queen
  - 5  : White King
  - 6  : Black Pawn
  - 7  : Black Knight
  - 8  : Black Bishop
  - 9  : Black Queen
  - 10 : Black King

Board coordinates:
  - Bottom-left  = A1  (index [0][0])
  - Columns   = A-F (left to right)
  - Rows      = 6-1 (top to bottom)(from white's perspective)

Move output format:  "<piece_id>:<source_cell>-><target_cell>"
  e.g.  "1:B3->B4"   (White Pawn moves from B3 to B4)
"""

import numpy as np
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMPTY = 0

# Piece IDs
WHITE_PAWN   = 1
WHITE_KNIGHT = 2
WHITE_BISHOP = 3
WHITE_QUEEN  = 4
WHITE_KING   = 5
BLACK_PAWN   = 6
BLACK_KNIGHT = 7
BLACK_BISHOP = 8
BLACK_QUEEN  = 9
BLACK_KING   = 10

WHITE_PIECES = {WHITE_PAWN, WHITE_KNIGHT, WHITE_BISHOP, WHITE_QUEEN, WHITE_KING}
BLACK_PIECES = {BLACK_PAWN, BLACK_KNIGHT, BLACK_BISHOP, BLACK_QUEEN, BLACK_KING}

BOARD_SIZE = 6

PIECE_VALUES = {
    WHITE_PAWN:   100,
    WHITE_KNIGHT: 300,
    WHITE_BISHOP: 320,
    WHITE_QUEEN:  900,
    WHITE_KING:  20000,
    BLACK_PAWN:  -100,
    BLACK_KNIGHT:-300,
    BLACK_BISHOP:-320,
    BLACK_QUEEN: -900,
    BLACK_KING: -20000,
}

# Initial piece counts for promotion rules
INITIAL_COUNTS = {'Q': 1, 'B': 2, 'N': 2}

# Column index -> letter
COL_TO_FILE = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F'}
FILE_TO_COL = {v: k for k, v in COL_TO_FILE.items()}

# ---------------------------------------------------------------------------
# Piece-Square Tables (6x6)
# From White's perspective. Flipped vertically for Black.
# ---------------------------------------------------------------------------

PAWN_PST = np.array([
    [ 0,  0,  0,  0,  0,  0],   # row 1 — start rank
    [ 5,  5,  5,  5,  5,  5],   # row 2
    [10, 10, 15, 15, 10, 10],   # row 3 — centre bonus
    [20, 20, 25, 25, 20, 20],   # row 4 — advanced
    [35, 35, 35, 35, 35, 35],   # row 5 — near promotion
    [ 0,  0,  0,  0,  0,  0],   # row 6 — promoted
], dtype=float)

KNIGHT_PST = np.array([
    [-30,-20,-10,-10,-20,-30],
    [-20,  0,  5,  5,  0,-20],
    [-10,  5, 20, 20,  5,-10],
    [-10,  5, 20, 20,  5,-10],
    [-20,  0,  5,  5,  0,-20],
    [-30,-20,-10,-10,-20,-30],
], dtype=float)

BISHOP_PST = np.array([
    [-10, -5, -5, -5, -5,-10],
    [ -5,  5,  5,  5,  5, -5],
    [ -5,  5, 10, 10,  5, -5],
    [ -5,  5, 10, 10,  5, -5],
    [ -5,  5,  5,  5,  5, -5],
    [-10, -5, -5, -5, -5,-10],
], dtype=float)

QUEEN_PST = np.array([
    [-10, -5, -5,  0, -5,-10],
    [ -5,  0,  5,  5,  0, -5],
    [ -5,  5, 10, 10,  5, -5],
    [ -5,  5, 10, 10,  5, -5],
    [ -5,  0,  5,  5,  0, -5],
    [-10, -5, -5,  0, -5,-10],
], dtype=float)

KING_PST = np.array([
    [ 20, 25,  5,  5, 25, 20],   # back rank — stay safe
    [  5,  5,  0,  0,  5,  5],
    [ -5,-10,-15,-15,-10, -5],
    [-10,-15,-20,-20,-15,-10],
    [-15,-20,-25,-25,-20,-15],
    [-20,-25,-30,-30,-25,-20],
], dtype=float)

PST = {
    WHITE_PAWN:   PAWN_PST,
    WHITE_KNIGHT: KNIGHT_PST,
    WHITE_BISHOP: BISHOP_PST,
    WHITE_QUEEN:  QUEEN_PST,
    WHITE_KING:   KING_PST,
}

# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def idx_to_cell(row: int, col: int) -> str:
    """Convert (row, col) zero-indexed to board notation e.g. (0,0) -> 'A1'."""
    return f"{COL_TO_FILE[col]}{row + 1}"

def cell_to_idx(cell: str):
    """Convert board notation e.g. 'A1' -> (row=0, col=0)."""
    col = FILE_TO_COL[cell[0].upper()]
    row = int(cell[1]) - 1
    return row, col

def in_bounds(row: int, col: int) -> bool:
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE

def is_white(piece: int) -> bool:
    return piece in WHITE_PIECES

def is_black(piece: int) -> bool:
    return piece in BLACK_PIECES

def same_side(p1: int, p2: int) -> bool:
    if p1 == EMPTY or p2 == EMPTY:
        return False
    return (is_white(p1) and is_white(p2)) or (is_black(p1) and is_black(p2))

# ---------------------------------------------------------------------------
# Pawn Promotion Rule Logic
# ---------------------------------------------------------------------------

def get_available_promotions(board: np.ndarray, playing_white: bool):
    """Promotion only allowed to pieces already lost in the game."""
    q_cnt, b_cnt, n_cnt = 0, 0, 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            p = board[r][c]
            if playing_white:
                if p == WHITE_QUEEN:   q_cnt += 1
                elif p == WHITE_BISHOP: b_cnt += 1
                elif p == WHITE_KNIGHT: n_cnt += 1
            else:
                if p == BLACK_QUEEN:   q_cnt += 1
                elif p == BLACK_BISHOP: b_cnt += 1
                elif p == BLACK_KNIGHT: n_cnt += 1
    promos = []
    if q_cnt < INITIAL_COUNTS['Q']: promos.append(WHITE_QUEEN  if playing_white else BLACK_QUEEN)
    if b_cnt < INITIAL_COUNTS['B']: promos.append(WHITE_BISHOP if playing_white else BLACK_BISHOP)
    if n_cnt < INITIAL_COUNTS['N']: promos.append(WHITE_KNIGHT if playing_white else BLACK_KNIGHT)
    return promos

# ---------------------------------------------------------------------------
# Move generation
# ---------------------------------------------------------------------------

def get_pawn_moves(board: np.ndarray, row: int, col: int, piece: int):
    """
    White Pawns move upward (increasing row index).
    Black Pawns move downward (decreasing row index).
    Captures are diagonal-forward.
    """
    moves = []
    is_w = is_white(piece)
    direction = 1 if is_w else -1
    last_rank = 5 if is_w else 0

    # Forward move
    nr = row + direction
    if in_bounds(nr, col) and board[nr][col] == EMPTY:
        if nr == last_rank:
            for promo in get_available_promotions(board, is_w):
                moves.append((piece, row, col, nr, col, promo))
        else:
            moves.append((piece, row, col, nr, col, None))

    # Diagonal captures
    for dc in [-1, 1]:
        nr, nc = row + direction, col + dc
        if in_bounds(nr, nc):
            target = board[nr][nc]
            if target != EMPTY and not same_side(piece, target):
                if nr == last_rank:
                    for promo in get_available_promotions(board, is_w):
                        moves.append((piece, row, col, nr, nc, promo))
                else:
                    moves.append((piece, row, col, nr, nc, None))
    return moves


def get_knight_moves(board: np.ndarray, row: int, col: int, piece: int):
    moves = []
    for dr, dc in [(-2,-1),(-2,+1),(+2,-1),(+2,+1),(-1,-2),(-1,+2),(+1,-2),(+1,+2)]:
        nr, nc = row + dr, col + dc
        if in_bounds(nr, nc):
            target = board[nr][nc]
            if target == EMPTY or not same_side(piece, target):
                moves.append((piece, row, col, nr, nc, None))
    return moves


def get_sliding_moves(board: np.ndarray, row: int, col: int, piece: int, directions):
    """Generic sliding piece (bishop / queen directions)."""
    moves = []
    for dr, dc in directions:
        nr, nc = row + dr, col + dc
        while in_bounds(nr, nc):
            target = board[nr][nc]
            if target == EMPTY:
                moves.append((piece, row, col, nr, nc, None))
            elif not same_side(piece, target):
                moves.append((piece, row, col, nr, nc, None))
                break
            else:
                break
            nr += dr
            nc += dc
    return moves


def get_bishop_moves(board: np.ndarray, row: int, col: int, piece: int):
    diagonals = [(-1,-1),(-1,1),(1,-1),(1,1)]
    return get_sliding_moves(board, row, col, piece, diagonals)


def get_queen_moves(board: np.ndarray, row: int, col: int, piece: int):
    all_dirs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    return get_sliding_moves(board, row, col, piece, all_dirs)


def get_king_moves(board: np.ndarray, row: int, col: int, piece: int):
    moves = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = row + dr, col + dc
            if in_bounds(nr, nc):
                target = board[nr][nc]
                if target == EMPTY or not same_side(piece, target):
                    moves.append((piece, row, col, nr, nc, None))
    return moves


MOVE_GENERATORS = {
    WHITE_PAWN:   get_pawn_moves,
    WHITE_KNIGHT: get_knight_moves,
    WHITE_BISHOP: get_bishop_moves,
    WHITE_QUEEN:  get_queen_moves,
    WHITE_KING:   get_king_moves,
    BLACK_PAWN:   get_pawn_moves,
    BLACK_KNIGHT: get_knight_moves,
    BLACK_BISHOP: get_bishop_moves,
    BLACK_QUEEN:  get_queen_moves,
    BLACK_KING:   get_king_moves,
}


def get_pseudo_legal_moves(board: np.ndarray, playing_white: bool):
    """All moves ignoring whether king is left in check."""
    moves = []
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            piece = board[row][col]
            if piece == EMPTY:
                continue
            if playing_white and not is_white(piece):
                continue
            if not playing_white and not is_black(piece):
                continue
            generator = MOVE_GENERATORS.get(piece)
            if generator:
                moves.extend(generator(board, row, col, piece))
    return moves

# ---------------------------------------------------------------------------
# Apply move
# ---------------------------------------------------------------------------

def apply_move(board: np.ndarray, piece, src_row, src_col, dst_row, dst_col, promo_piece=None) -> np.ndarray:
    new_board = board.copy()
    new_board[src_row][src_col] = EMPTY
    new_board[dst_row][dst_col] = promo_piece if promo_piece is not None else piece
    return new_board

# ---------------------------------------------------------------------------
# Check detection
# ---------------------------------------------------------------------------

def is_in_check(board: np.ndarray, playing_white: bool) -> bool:
    """Is the king of the given side currently in check?"""
    king_id = WHITE_KING if playing_white else BLACK_KING
    king_pos = None
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] == king_id:
                king_pos = (r, c)
                break
        if king_pos:
            break
    if not king_pos:
        return False
    opp_moves = get_pseudo_legal_moves(board, not playing_white)
    for m in opp_moves:
        if m[3] == king_pos[0] and m[4] == king_pos[1]:
            return True
    return False

# ---------------------------------------------------------------------------
# Legal move generation — filters out moves that leave king in check
# get_legal_moves is the main public API used by play_game.py
# get_all_moves is an alias kept for backward compatibility
# ---------------------------------------------------------------------------

def get_legal_moves(board: np.ndarray, playing_white: bool):
    """
    Return only strictly legal moves — moves that don't leave own king in check.
    Format: (piece_id, src_row, src_col, dst_row, dst_col, promo_piece)
    """
    pseudo = get_pseudo_legal_moves(board, playing_white)
    legal  = []
    for m in pseudo:
        temp_board = apply_move(board, *m)
        if not is_in_check(temp_board, playing_white):
            legal.append(m)
    return legal

# Alias — keeps backward compat if anything calls get_all_moves
def get_all_moves(board: np.ndarray, playing_white: bool):
    return get_legal_moves(board, playing_white)

# ---------------------------------------------------------------------------
# Board evaluation — tuned for 6x6 + random back rank awareness
# ---------------------------------------------------------------------------

def detect_back_rank_pieces(board: np.ndarray, playing_white: bool):
    """
    Read actual back rank layout (randomised each game).
    Returns dict: col -> piece
    """
    back_row = 0 if playing_white else 5
    return {col: board[back_row][col] for col in range(BOARD_SIZE)
            if board[back_row][col] != EMPTY}


def evaluate(board: np.ndarray) -> float:
    """
    Static board evaluation from White's perspective.
    Positive  -> advantage for White
    Negative  -> advantage for Black

    Components:
      1. Material score
      2. Piece-square table bonuses (positional)
      3. Mobility bonus
      4. Centre control bonus
      5. King safety — pawn shield based on actual king position
      6. Check penalty/bonus
    """
    score = 0.0

    # 3. Mobility
    white_mobility = len(get_pseudo_legal_moves(board, True))
    black_mobility = len(get_pseudo_legal_moves(board, False))
    score += 5 * (white_mobility - black_mobility)

    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            piece = board[row][col]
            if piece == EMPTY:
                continue

            # 1. Material
            score += PIECE_VALUES.get(piece, 0)

            # 2. Piece-square tables
            if is_white(piece) and piece in PST:
                score += PST[piece][row][col]
            elif is_black(piece):
                white_equiv = piece - 5
                if white_equiv in PST:
                    score -= PST[white_equiv][BOARD_SIZE - 1 - row][col]

            # 4. Centre control
            if row in [2, 3] and col in [2, 3]:
                score += 15 if is_white(piece) else -15

    # 5. King safety — find king dynamically (works for any back rank layout)
    for col in range(BOARD_SIZE):
        if board[0][col] == WHITE_KING:
            if in_bounds(1, col)   and board[1][col]   == WHITE_PAWN: score += 20
            if in_bounds(1, col-1) and board[1][col-1] == WHITE_PAWN: score += 10
            if in_bounds(1, col+1) and board[1][col+1] == WHITE_PAWN: score += 10
        if board[5][col] == BLACK_KING:
            if in_bounds(4, col)   and board[4][col]   == BLACK_PAWN: score -= 20
            if in_bounds(4, col-1) and board[4][col-1] == BLACK_PAWN: score -= 10
            if in_bounds(4, col+1) and board[4][col+1] == BLACK_PAWN: score -= 10

    # 6. Check penalty
    if is_in_check(board, True):  score -= 80
    if is_in_check(board, False): score += 80

    return score

# ---------------------------------------------------------------------------
# Move ordering — MVV-LVA for better alpha-beta pruning
# ---------------------------------------------------------------------------

def score_move_for_ordering(board: np.ndarray, move) -> int:
    """Score a move for ordering. Higher = search first."""
    piece, sr, sc, dr, dc, promo = move
    target = board[dr][dc]
    order_score = 0

    if promo is not None:
        order_score += 800

    if target != EMPTY:
        victim_val   = abs(PIECE_VALUES.get(target, 0))
        attacker_val = abs(PIECE_VALUES.get(piece, 0))
        order_score += victim_val - (attacker_val // 10)

    if dr in [2, 3] and dc in [2, 3]:
        order_score += 20

    return order_score


def order_moves(board: np.ndarray, moves: list) -> list:
    return sorted(moves, key=lambda m: score_move_for_ordering(board, m), reverse=True)

# ---------------------------------------------------------------------------
# Format move string
# ---------------------------------------------------------------------------

def format_move(piece: int, src_row: int, src_col: int,
                dst_row: int, dst_col: int, promo_piece: Optional[int] = None) -> str:
    """Return move in required format: '<piece_id>:<source_cell>-><target_cell>'."""
    src_cell = idx_to_cell(src_row, src_col)
    dst_cell = idx_to_cell(dst_row, dst_col)
    move_str = f"{piece}:{src_cell}->{dst_cell}"
    if promo_piece is not None:
        move_str += f"={promo_piece}"
    return move_str

# ---------------------------------------------------------------------------
# Minimax with Alpha-Beta pruning + move ordering
# ---------------------------------------------------------------------------

def minimax(board: np.ndarray, depth: int, alpha: float, beta: float,
            maximizing: bool) -> float:
    if depth == 0:
        return evaluate(board)

    moves = get_legal_moves(board, maximizing)

    if not moves:
        if is_in_check(board, maximizing):
            return (-99999 - depth) if maximizing else (99999 + depth)
        return 0  # Stalemate

    moves = order_moves(board, moves)

    if maximizing:
        max_score = float('-inf')
        for move in moves:
            new_board = apply_move(board, *move)
            score = minimax(new_board, depth - 1, alpha, beta, False)
            max_score = max(max_score, score)
            alpha = max(alpha, score)
            if beta <= alpha:
                break
        return max_score
    else:
        min_score = float('inf')
        for move in moves:
            new_board = apply_move(board, *move)
            score = minimax(new_board, depth - 1, alpha, beta, True)
            min_score = min(min_score, score)
            beta = min(beta, score)
            if beta <= alpha:
                break
        return min_score

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def get_best_move(board: np.ndarray, playing_white: bool = True,
                  depth: int = 4) -> Optional[str]:
    """
    Given the current board state, return the best move string.

    Parameters
    ----------
    board        : 6x6 NumPy array representing the current game state.
    playing_white: True if the engine is playing as White, False for Black.
    depth        : Minimax search depth (default 4).

    Returns
    -------
    Move string in the format '<piece_id>:<src_cell>-><dst_cell>', or
    None if no legal moves are available.
    """
    all_moves = get_legal_moves(board, playing_white)
    if not all_moves:
        return None

    best_move  = None
    best_score = float('-inf') if playing_white else float('inf')

    all_moves = order_moves(board, all_moves)

    for move in all_moves:
        new_board = apply_move(board, *move)
        score = minimax(new_board, depth - 1, float('-inf'), float('inf'), not playing_white)

        if playing_white and score > best_score:
            best_score = score
            best_move  = move
        elif not playing_white and score < best_score:
            best_score = score
            best_move  = move

    return format_move(*best_move)

# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

# if __name__ == "__main__":
#     initial_board = np.array([
#         [ 2,  3,  4,  5,  3,  2],   # Row 1 (A1-F1) — White back rank
#         [ 1,  1,  1,  1,  1,  1],   # Row 2         — White pawns
#         [ 0,  0,  0,  0,  0,  0],   # Row 3
#         [ 0,  0,  0,  0,  0,  0],   # Row 4
#         [ 6,  6,  6,  6,  6,  6],   # Row 5         — Black pawns
#         [ 7,  8,  9, 10,  8,  7],   # Row 6 (A6-F6) — Black back rank
#     ], dtype=int)

#     print("Board:\n", initial_board)
#     move = get_best_move(initial_board, playing_white=True)
#     print("Best move for White:", move)