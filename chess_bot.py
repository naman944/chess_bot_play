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
  - Columns   = A–F (left to right)
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

# Column index → letter
COL_TO_FILE = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F'}
FILE_TO_COL = {v: k for k, v in COL_TO_FILE.items()}

# Promotion rules — only allowed to pieces already lost
INITIAL_COUNTS = {'Q': 1, 'B': 2, 'N': 2}

# ---------------------------------------------------------------------------
# Piece-Square Tables (6x6) — tuned for 6x6 board
# White's perspective. Flipped vertically for Black.
# ---------------------------------------------------------------------------

PAWN_PST = np.array([
    [ 0,  0,  0,  0,  0,  0],
    [ 5,  8,  8,  8,  8,  5],
    [10, 12, 18, 18, 12, 10],
    [22, 24, 28, 28, 24, 22],
    [38, 38, 40, 40, 38, 38],
    [ 0,  0,  0,  0,  0,  0],
], dtype=float)

KNIGHT_PST = np.array([
    [-40,-25,-15,-15,-25,-40],
    [-25, -5,  8,  8, -5,-25],
    [-15,  8, 25, 25,  8,-15],
    [-15,  8, 25, 25,  8,-15],
    [-25, -5,  8,  8, -5,-25],
    [-40,-25,-15,-15,-25,-40],
], dtype=float)

BISHOP_PST = np.array([
    [-12, -6, -6, -6, -6,-12],
    [ -6,  8,  8,  8,  8, -6],
    [ -6,  8, 14, 14,  8, -6],
    [ -6,  8, 14, 14,  8, -6],
    [ -6,  8,  8,  8,  8, -6],
    [-12, -6, -6, -6, -6,-12],
], dtype=float)

QUEEN_PST = np.array([
    [-12, -6, -4, -4, -6,-12],
    [ -6,  2,  8,  8,  2, -6],
    [ -4,  8, 14, 14,  8, -4],
    [ -4,  8, 14, 14,  8, -4],
    [ -6,  2,  8,  8,  2, -6],
    [-12, -6, -4, -4, -6,-12],
], dtype=float)

KING_PST = np.array([
    [ 30, 35, 10, 10, 35, 30],
    [ 10, 10,  0,  0, 10, 10],
    [ -8,-15,-22,-22,-15, -8],
    [-15,-22,-30,-30,-22,-15],
    [-22,-30,-38,-38,-30,-22],
    [-30,-38,-45,-45,-38,-30],
], dtype=float)

PST = {
    WHITE_PAWN:   PAWN_PST,
    WHITE_KNIGHT: KNIGHT_PST,
    WHITE_BISHOP: BISHOP_PST,
    WHITE_QUEEN:  QUEEN_PST,
    WHITE_KING:   KING_PST,
}

# ---------------------------------------------------------------------------
# Internal state — repetition tracking (module-level, no extra params needed)
# ---------------------------------------------------------------------------

# position_history: maps board bytes -> visit count
# reset each game via reset_game_state()
_position_history = {}
_move_number      = 0
_killer_moves     = [[None, None] for _ in range(20)]
_time_remaining   = 900.0   # seconds left on clock (default 15 min)

def reset_game_state():
    """Call this at the start of each new game."""
    global _position_history, _move_number, _killer_moves, _time_remaining
    _position_history = {}
    _move_number      = 0
    _killer_moves     = [[None, None] for _ in range(20)]
    _time_remaining   = 900.0

def set_time_remaining(seconds: float):
    """Call this before get_best_move to update the clock."""
    global _time_remaining
    _time_remaining = max(0.0, seconds)

def _record_position(board: np.ndarray):
    """Record current board position for repetition detection."""
    key = board.tobytes()
    _position_history[key] = _position_history.get(key, 0) + 1

def _repetition_penalty(board: np.ndarray) -> int:
    """Return penalty score for positions seen before (higher = more repeats)."""
    key   = board.tobytes()
    count = _position_history.get(key, 0)
    return count * 150   # 150 pts per repeat visit

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
                if p == WHITE_QUEEN:    q_cnt += 1
                elif p == WHITE_BISHOP: b_cnt += 1
                elif p == WHITE_KNIGHT: n_cnt += 1
            else:
                if p == BLACK_QUEEN:    q_cnt += 1
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
    is_w      = is_white(piece)
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
    for dr, dc in [(-2,-1),(-2,+1),(+2,-1),(+2,+1),
                   (-1,-2),(-1,+2),(+1,-2),(+1,+2)]:
        nr, nc = row + dr, col + dc
        if in_bounds(nr, nc):
            target = board[nr][nc]
            if target == EMPTY or not same_side(piece, target):
                moves.append((piece, row, col, nr, nc, None))
    return moves


def get_sliding_moves(board: np.ndarray, row: int, col: int, piece: int, directions):
    """Generic sliding piece (bishop / queen / rook directions)."""
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


def get_all_moves(board: np.ndarray, playing_white: bool):
    """Return list of (piece_id, src_row, src_col, dst_row, dst_col, promo) for all legal moves."""
    pseudo = []
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            piece = board[row][col]
            if piece == EMPTY: continue
            if playing_white  and not is_white(piece): continue
            if not playing_white and not is_black(piece): continue
            gen = MOVE_GENERATORS.get(piece)
            if gen:
                pseudo.extend(gen(board, row, col, piece))

    # Filter — remove moves that leave own king in check
    legal = []
    for m in pseudo:
        temp = apply_move(board, *m)
        if not is_in_check(temp, playing_white):
            legal.append(m)
    return legal

# ---------------------------------------------------------------------------
# Apply move
# ---------------------------------------------------------------------------

def apply_move(board: np.ndarray, piece, src_row, src_col,
               dst_row, dst_col, promo_piece=None) -> np.ndarray:
    new_board = board.copy()
    new_board[src_row][src_col] = EMPTY
    new_board[dst_row][dst_col] = promo_piece if promo_piece is not None else piece
    return new_board

# ---------------------------------------------------------------------------
# Check detection
# ---------------------------------------------------------------------------

def is_in_check(board: np.ndarray, playing_white: bool) -> bool:
    """Is the king of the given side currently in check?"""
    king_id  = WHITE_KING if playing_white else BLACK_KING
    king_pos = None
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] == king_id:
                king_pos = (r, c)
                break
        if king_pos: break
    if not king_pos: return False

    # Generate all opponent pseudo-legal moves and see if any attack king
    opp_pieces = BLACK_PIECES if playing_white else WHITE_PIECES
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            p = board[r][c]
            if p not in opp_pieces: continue
            gen = MOVE_GENERATORS.get(p)
            if not gen: continue
            for m in gen(board, r, c, p):
                if m[3] == king_pos[0] and m[4] == king_pos[1]:
                    return True
    return False

# ---------------------------------------------------------------------------
# Board evaluation heuristic
# ---------------------------------------------------------------------------

def evaluate(board: np.ndarray) -> float:
    """
    Static board evaluation from White's perspective.
    Positive  → advantage for White
    Negative  → advantage for Black

    Includes:
      1.  Material
      2.  Piece-square tables
      3.  Mobility
      4.  Centre control
      5.  King safety (dynamic pawn shield)
      6.  Passed pawns
      7.  Pawn structure (doubled / isolated)
      8.  Bishop pair bonus
      9.  Endgame king activity
      10. Check bonus
    """
    score   = 0.0
    # Endgame check — fewer pieces = king should be active
    total_mat = sum(abs(PIECE_VALUES.get(board[r][c], 0))
                    for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)
                    if board[r][c] not in (EMPTY, WHITE_KING, BLACK_KING))
    endgame = total_mat < 1800

    # Mobility (fast approximation — count pseudo moves)
    w_mob = len(_pseudo_moves_fast(board, True))
    b_mob = len(_pseudo_moves_fast(board, False))
    score += 4 * (w_mob - b_mob)

    w_pawns_col  = [0] * BOARD_SIZE
    b_pawns_col  = [0] * BOARD_SIZE
    w_bishop_cnt = 0
    b_bishop_cnt = 0

    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            piece = board[row][col]
            if piece == EMPTY: continue

            # 1. Material
            score += PIECE_VALUES.get(piece, 0)

            # 2. PST
            if is_white(piece) and piece in PST:
                pst_val = PST[piece][row][col]
                if piece == WHITE_KING and endgame:
                    pst_val = -pst_val * 0.5
                score += pst_val
            elif is_black(piece):
                equiv = piece - 5
                if equiv in PST:
                    pst_val = PST[equiv][BOARD_SIZE - 1 - row][col]
                    if piece == BLACK_KING and endgame:
                        pst_val = -pst_val * 0.5
                    score -= pst_val

            # 4. Centre control
            if row in [2, 3] and col in [2, 3]:
                score += 18 if is_white(piece) else -18

            if piece == WHITE_PAWN:  w_pawns_col[col] += 1
            if piece == BLACK_PAWN:  b_pawns_col[col] += 1
            if piece == WHITE_BISHOP: w_bishop_cnt += 1
            if piece == BLACK_BISHOP: b_bishop_cnt += 1

    # 5. King safety
    for col in range(BOARD_SIZE):
        if board[0][col] == WHITE_KING:
            shield = 0
            if in_bounds(1, col)   and board[1][col]   == WHITE_PAWN: shield += 25
            if in_bounds(1, col-1) and board[1][col-1] == WHITE_PAWN: shield += 12
            if in_bounds(1, col+1) and board[1][col+1] == WHITE_PAWN: shield += 12
            score += shield
            if shield == 0: score -= 30
        if board[5][col] == BLACK_KING:
            shield = 0
            if in_bounds(4, col)   and board[4][col]   == BLACK_PAWN: shield += 25
            if in_bounds(4, col-1) and board[4][col-1] == BLACK_PAWN: shield += 12
            if in_bounds(4, col+1) and board[4][col+1] == BLACK_PAWN: shield += 12
            score -= shield
            if shield == 0: score += 30

    # 6. Passed pawns
    for col in range(BOARD_SIZE):
        for row in range(BOARD_SIZE):
            if board[row][col] == WHITE_PAWN:
                blocked = any(in_bounds(r, col+dc) and board[r][col+dc] == BLACK_PAWN
                              for r in range(row+1, BOARD_SIZE) for dc in [-1,0,1])
                if not blocked: score += 15 + row * 8
            if board[row][col] == BLACK_PAWN:
                blocked = any(in_bounds(r, col+dc) and board[r][col+dc] == WHITE_PAWN
                              for r in range(row-1, -1, -1) for dc in [-1,0,1])
                if not blocked: score -= 15 + (BOARD_SIZE-1-row) * 8

    # 7. Pawn structure
    for col in range(BOARD_SIZE):
        if w_pawns_col[col] > 1: score -= 20 * (w_pawns_col[col] - 1)
        if b_pawns_col[col] > 1: score += 20 * (b_pawns_col[col] - 1)
        if w_pawns_col[col] > 0:
            l = w_pawns_col[col-1] if col > 0 else 0
            r = w_pawns_col[col+1] if col < BOARD_SIZE-1 else 0
            if l == 0 and r == 0: score -= 15 * w_pawns_col[col]
        if b_pawns_col[col] > 0:
            l = b_pawns_col[col-1] if col > 0 else 0
            r = b_pawns_col[col+1] if col < BOARD_SIZE-1 else 0
            if l == 0 and r == 0: score += 15 * b_pawns_col[col]

    # 8. Bishop pair
    if w_bishop_cnt >= 2: score += 40
    if b_bishop_cnt >= 2: score -= 40

    # 10. Check bonus
    if is_in_check(board, True):  score -= 90
    if is_in_check(board, False): score += 90

    return score


def _pseudo_moves_fast(board: np.ndarray, playing_white: bool) -> list:
    """Fast pseudo-legal move count for mobility (no legality filter)."""
    moves = []
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            piece = board[row][col]
            if piece == EMPTY: continue
            if playing_white  and not is_white(piece): continue
            if not playing_white and not is_black(piece): continue
            gen = MOVE_GENERATORS.get(piece)
            if gen: moves.extend(gen(board, row, col, piece))
    return moves

# ---------------------------------------------------------------------------
# Move ordering — MVV-LVA + killer moves
# ---------------------------------------------------------------------------

def _update_killer(move, depth):
    global _killer_moves
    if depth < 20:
        if _killer_moves[depth][0] != move:
            _killer_moves[depth][1] = _killer_moves[depth][0]
            _killer_moves[depth][0] = move


def _score_move(board: np.ndarray, move, depth: int = 0) -> int:
    piece, sr, sc, dr, dc, promo = move
    target      = board[dr][dc]
    order_score = 0

    if promo is not None:
        order_score += 900

    if target != EMPTY:
        victim_val   = abs(PIECE_VALUES.get(target, 0))
        attacker_val = abs(PIECE_VALUES.get(piece, 0))
        see = victim_val - attacker_val
        order_score += (700 + see) if see >= 0 else (100 + see)
    elif depth < 20:
        if move == _killer_moves[depth][0]: order_score += 80
        elif move == _killer_moves[depth][1]: order_score += 70

    if dr in [2, 3] and dc in [2, 3]: order_score += 25
    if piece == WHITE_PAWN: order_score += dr * 3
    if piece == BLACK_PAWN: order_score += (BOARD_SIZE - 1 - dr) * 3

    return order_score


def _order_moves(board: np.ndarray, moves: list, depth: int = 0) -> list:
    return sorted(moves, key=lambda m: _score_move(board, m, depth), reverse=True)

# ---------------------------------------------------------------------------
# Format move string
# ---------------------------------------------------------------------------

def format_move(piece: int, src_row: int, src_col: int,
                dst_row: int, dst_col: int) -> str:
    """Return move in required format: '<piece_id>:<source_cell>-><target_cell>'."""
    src_cell = idx_to_cell(src_row, src_col)
    dst_cell = idx_to_cell(dst_row, dst_col)
    return f"{piece}:{src_cell}->{dst_cell}"

# ---------------------------------------------------------------------------
# Quiescence search — prevents horizon effect
# ---------------------------------------------------------------------------

def _quiescence(board: np.ndarray, alpha: float, beta: float,
                maximizing: bool, qdepth: int = 4) -> float:
    stand_pat = evaluate(board)
    if maximizing:
        if stand_pat >= beta:  return beta
        alpha = max(alpha, stand_pat)
    else:
        if stand_pat <= alpha: return alpha
        beta  = min(beta, stand_pat)
    if qdepth == 0: return stand_pat

    captures = [m for m in get_all_moves(board, maximizing)
                if board[m[3]][m[4]] != EMPTY or m[5] is not None]
    for move in _order_moves(board, captures):
        score = _quiescence(apply_move(board, *move),
                            alpha, beta, not maximizing, qdepth - 1)
        if maximizing:
            alpha = max(alpha, score)
            if alpha >= beta: return beta
        else:
            beta = min(beta, score)
            if beta <= alpha: return alpha
    return alpha if maximizing else beta

# ---------------------------------------------------------------------------
# Minimax + Alpha-Beta
# ---------------------------------------------------------------------------

def _minimax(board: np.ndarray, depth: int, alpha: float, beta: float,
             maximizing: bool) -> float:
    if depth == 0:
        return _quiescence(board, alpha, beta, maximizing)

    moves = get_all_moves(board, maximizing)
    if not moves:
        if is_in_check(board, maximizing):
            return (-99999 - depth) if maximizing else (99999 + depth)
        return 0   # stalemate

    moves = _order_moves(board, moves, depth)

    if maximizing:
        best = float('-inf')
        for move in moves:
            score = _minimax(apply_move(board, *move),
                             depth - 1, alpha, beta, False)
            best  = max(best, score)
            alpha = max(alpha, score)
            if beta <= alpha:
                _update_killer(move, depth)
                break
        return best
    else:
        best = float('inf')
        for move in moves:
            score = _minimax(apply_move(board, *move),
                             depth - 1, alpha, beta, True)
            best = min(best, score)
            beta = min(beta, score)
            if beta <= alpha:
                _update_killer(move, depth)
                break
        return best

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Dynamic depth — pieces left × time left, working together
# ---------------------------------------------------------------------------
#
# LOGIC:
#   pieces_score  : fewer pieces  → higher base depth (less branching)
#   time_score    : more time left → allow higher depth
#   Both scores combined → final depth
#
#   Grid (pieces → rows, time → cols):
#
#              <1min   1-3min  3-7min  7-15min
#   20+ pcs     2       3       3       4      (opening — many branches)
#   10-19 pcs   2       3       4       5      (midgame)
#   5-9  pcs    3       4       5       6      (late game)
#   <5   pcs    3       5       6       7      (endgame — deep!)
#
# ---------------------------------------------------------------------------

#  depth_table[piece_bucket][time_bucket]
#  piece buckets : 0=(<5), 1=(5-9), 2=(10-19), 3=(20+)
#  time  buckets : 0=(<60s), 1=(60-180s), 2=(180-420s), 3=(420s+)
_DEPTH_TABLE = [
    [3, 5, 6, 7],   # <5  pieces  — endgame, go very deep when time allows
    [3, 4, 5, 6],   # 5-9 pieces  — late game
    [2, 3, 4, 5],   # 10-19 pcs   — midgame
    [2, 3, 3, 4],   # 20+  pcs    — opening, keep shallow (too many branches)
]

def _get_depth(board: np.ndarray) -> int:
    """
    Choose search depth based on BOTH pieces remaining AND time remaining.
    More time + fewer pieces = deeper search.
    Emergency cap applied when clock is critically low.
    """
    pieces = int(np.count_nonzero(board))
    secs   = _time_remaining

    # Time bucket
    if secs < 60:    t_bucket = 0   # danger zone  — play fast
    elif secs < 180: t_bucket = 1   # 1–3 min left
    elif secs < 420: t_bucket = 2   # 3–7 min left
    else:            t_bucket = 3   # 7–15 min     — plenty of time

    # Piece bucket
    if pieces < 5:    p_bucket = 0
    elif pieces < 10: p_bucket = 1
    elif pieces < 20: p_bucket = 2
    else:             p_bucket = 3

    depth = _DEPTH_TABLE[p_bucket][t_bucket]

    # Hard safety cap — never exceed depth 7 (too slow on 6x6 with queens)
    return min(depth, 7)

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def get_best_move(board: np.ndarray, playing_white: bool = True
                  ) -> Optional[str]:
    """
    Given the current board state, return the best move string.

    Parameters
    ----------
    board        : 6×6 NumPy array representing the current game state.
    playing_white: True if the engine is playing as White, False for Black.

    Returns
    -------
    Move string in the format '<piece_id>:<src_cell>-><dst_cell>', or
    None if no legal moves are available.
    """
    global _move_number, _killer_moves
    _killer_moves = [[None, None] for _ in range(20)]

    all_moves = get_all_moves(board, playing_white)
    if not all_moves:
        return None

    # Record this position for repetition tracking
    _record_position(board)

    # Dynamic depth based on piece count
    depth = _get_depth(board)

    # Temperature for move selection (decreases as game goes on)
    # Opening: slight randomness to be unpredictable
    # Endgame: deterministic
    if _move_number < 6:
        temperature = 0.4
    elif _move_number < 20:
        temperature = 0.3
    elif _move_number < 30:
        temperature = 0.1
    else:
        temperature = 0.0

    moves_with_scores = []

    # Iterative deepening — search depth 1 → depth
    for current_depth in range(1, depth + 1):
        iter_scores  = {}
        all_moves    = _order_moves(board, all_moves, current_depth)

        for move in all_moves:
            new_board = apply_move(board, *move)
            score     = _minimax(new_board, current_depth - 1,
                                 float('-inf'), float('inf'), not playing_white)

            # Repetition penalty — discourage revisiting positions
            rep_penalty = _repetition_penalty(new_board)
            score = score - rep_penalty if playing_white else score + rep_penalty

            # Tiny noise to break ties and prevent oscillation
            score += (hash(move) % 31 - 15) * 0.1

            iter_scores[id(move)] = (move, score)

        if current_depth == depth:
            moves_with_scores = list(iter_scores.values())

        # Bring best move to front for next iteration
        if iter_scores:
            if playing_white:
                best_id = max(iter_scores, key=lambda k: iter_scores[k][1])
            else:
                best_id = min(iter_scores, key=lambda k: iter_scores[k][1])
            best_iter = iter_scores[best_id][0]
            if best_iter in all_moves:
                all_moves.remove(best_iter)
                all_moves.insert(0, best_iter)

    if not moves_with_scores:
        return None

    # Flip for black so we always work with "higher = better"
    if not playing_white:
        moves_with_scores = [(m, -s) for m, s in moves_with_scores]

    # Top-N selection with temperature — only consider moves within
    # THRESHOLD points of best (moderate gambling)
    THRESHOLD = 50
    best_score  = max(s for _, s in moves_with_scores)
    candidates  = [(m, s) for m, s in moves_with_scores
                   if best_score - s <= THRESHOLD]

    if temperature < 0.01 or len(candidates) == 1:
        # Deterministic — pick best
        chosen = max(candidates, key=lambda x: x[1])[0]
    else:
        # Softmax pick from candidates
        import math, random
        scores  = [s for _, s in candidates]
        best_c  = max(scores)
        weights = [math.exp((s - best_c) / (temperature * 100 + 1e-9))
                   for s in scores]
        total   = sum(weights)
        probs   = [w / total for w in weights]
        chosen  = random.choices([m for m, _ in candidates],
                                 weights=probs, k=1)[0]

    _move_number += 1

    # format_move takes 5 args (strip promo from tuple)
    return format_move(chosen[0], chosen[1], chosen[2], chosen[3], chosen[4])


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Example: standard-ish starting position on a 6x6 board
    # White pieces on rows 4-5, Black pieces on rows 0-1
    initial_board = np.array([
        [ 2,  3,  4,  5,  3,  2],   # Row 1 (A1–F1) — White back rank
        [ 1,  1,  1,  1,  1,  1],   # Row 2         — White pawns
        [ 0,  0,  0,  0,  0,  0],   # Row 3
        [ 0,  0,  0,  0,  0,  0],   # Row 4
        [ 6,  6,  6,  6,  6,  6],   # Row 5         — Black pawns
        [ 7,  8,  9, 10,  8,  7],   # Row 6 (A6–F6) — Black back rank
    ], dtype=int)

    print("Board:\n", initial_board)
    move = get_best_move(initial_board, playing_white=True)
    print("Best move for White:", move)