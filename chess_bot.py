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

INITIAL_COUNTS = {'Q': 1, 'B': 2, 'N': 2}

COL_TO_FILE = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F'}
FILE_TO_COL = {v: k for k, v in COL_TO_FILE.items()}

# ---------------------------------------------------------------------------
# Piece-Square Tables (6x6) — tuned for 6x6 board
# White's perspective. Flipped vertically for Black.
# ---------------------------------------------------------------------------

# Pawns: strongly reward centre advance, penalise edges slightly
PAWN_PST = np.array([
    [ 0,  0,  0,  0,  0,  0],   # row 1 — start
    [ 5,  8,  8,  8,  8,  5],   # row 2
    [10, 12, 18, 18, 12, 10],   # row 3 — centre
    [22, 24, 28, 28, 24, 22],   # row 4 — advanced
    [38, 38, 40, 40, 38, 38],   # row 5 — near promotion
    [ 0,  0,  0,  0,  0,  0],   # row 6 — promoted
], dtype=float)

# Knights: strongly penalise edges/corners on 6x6 (very limited reach)
KNIGHT_PST = np.array([
    [-40,-25,-15,-15,-25,-40],
    [-25, -5,  8,  8, -5,-25],
    [-15,  8, 25, 25,  8,-15],
    [-15,  8, 25, 25,  8,-15],
    [-25, -5,  8,  8, -5,-25],
    [-40,-25,-15,-15,-25,-40],
], dtype=float)

# Bishops: reward long diagonals and open centre
BISHOP_PST = np.array([
    [-12, -6, -6, -6, -6,-12],
    [ -6,  8,  8,  8,  8, -6],
    [ -6,  8, 14, 14,  8, -6],
    [ -6,  8, 14, 14,  8, -6],
    [ -6,  8,  8,  8,  8, -6],
    [-12, -6, -6, -6, -6,-12],
], dtype=float)

# Queen: reward centre control, penalise corners
QUEEN_PST = np.array([
    [-12, -6, -4,  -4, -6,-12],
    [ -6,  2,  8,   8,  2, -6],
    [ -4,  8, 14,  14,  8, -4],
    [ -4,  8, 14,  14,  8, -4],
    [ -6,  2,  8,   8,  2, -6],
    [-12, -6, -4,  -4, -6,-12],
], dtype=float)

# King: STRONGLY penalise centre, reward corners/back rank with pawn cover
KING_PST = np.array([
    [ 30, 35, 10,  10, 35, 30],   # back rank — corners safest
    [ 10, 10,  0,   0, 10, 10],
    [ -8,-15,-22, -22,-15, -8],
    [-15,-22,-30, -30,-22,-15],
    [-22,-30,-38, -38,-30,-22],
    [-30,-38,-45, -45,-38,-30],
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
    return f"{COL_TO_FILE[col]}{row + 1}"

def cell_to_idx(cell: str):
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
    moves = []
    is_w = is_white(piece)
    direction = 1 if is_w else -1
    last_rank = 5 if is_w else 0

    nr = row + direction
    if in_bounds(nr, col) and board[nr][col] == EMPTY:
        if nr == last_rank:
            for promo in get_available_promotions(board, is_w):
                moves.append((piece, row, col, nr, col, promo))
        else:
            moves.append((piece, row, col, nr, col, None))

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
    return get_sliding_moves(board, row, col, piece, [(-1,-1),(-1,1),(1,-1),(1,1)])

def get_queen_moves(board: np.ndarray, row: int, col: int, piece: int):
    return get_sliding_moves(board, row, col, piece, [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)])

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
    WHITE_PAWN:   get_pawn_moves,   BLACK_PAWN:   get_pawn_moves,
    WHITE_KNIGHT: get_knight_moves, BLACK_KNIGHT: get_knight_moves,
    WHITE_BISHOP: get_bishop_moves, BLACK_BISHOP: get_bishop_moves,
    WHITE_QUEEN:  get_queen_moves,  BLACK_QUEEN:  get_queen_moves,
    WHITE_KING:   get_king_moves,   BLACK_KING:   get_king_moves,
}


def get_pseudo_legal_moves(board: np.ndarray, playing_white: bool):
    moves = []
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            piece = board[row][col]
            if piece == EMPTY: continue
            if playing_white  and not is_white(piece): continue
            if not playing_white and not is_black(piece): continue
            gen = MOVE_GENERATORS.get(piece)
            if gen:
                moves.extend(gen(board, row, col, piece))
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
    king_id = WHITE_KING if playing_white else BLACK_KING
    king_pos = None
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] == king_id:
                king_pos = (r, c)
                break
        if king_pos: break
    if not king_pos: return False
    for m in get_pseudo_legal_moves(board, not playing_white):
        if m[3] == king_pos[0] and m[4] == king_pos[1]:
            return True
    return False

# ---------------------------------------------------------------------------
# Legal moves — strict (no leaving own king in check)
# ---------------------------------------------------------------------------

def get_legal_moves(board: np.ndarray, playing_white: bool):
    """
    Strictly legal moves only.
    Format: (piece_id, src_row, src_col, dst_row, dst_col, promo_piece)
    """
    legal = []
    for m in get_pseudo_legal_moves(board, playing_white):
        if not is_in_check(apply_move(board, *m), playing_white):
            legal.append(m)
    return legal

def get_all_moves(board: np.ndarray, playing_white: bool):
    """Alias for backward compatibility."""
    return get_legal_moves(board, playing_white)

# ---------------------------------------------------------------------------
# Evaluation — competition-tuned
# ---------------------------------------------------------------------------

def count_material(board: np.ndarray, playing_white: bool) -> float:
    """Count total material value for one side (absolute)."""
    total = 0
    pieces = WHITE_PIECES if playing_white else BLACK_PIECES
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            p = board[r][c]
            if p in pieces and p != (WHITE_KING if playing_white else BLACK_KING):
                total += abs(PIECE_VALUES[p])
    return total


def is_endgame(board: np.ndarray) -> bool:
    """
    Endgame = both sides have lost significant material.
    On a 6x6 board this happens fast — triggers when total non-king
    material drops below a threshold.
    """
    white_mat = count_material(board, True)
    black_mat = count_material(board, False)
    return (white_mat + black_mat) < 1800   # roughly 2 minor pieces each


def evaluate(board: np.ndarray) -> float:
    """
    Competition-grade evaluation from White's perspective.
    Positive  -> advantage for White
    Negative  -> advantage for Black

    Layers:
      1.  Material
      2.  Piece-square tables (positional)
      3.  Mobility (legal move count difference)
      4.  Centre control
      5.  King safety — dynamic pawn shield + exposure penalty
      6.  Passed pawns — unstoppable promotion threats
      7.  Pawn structure — doubled/isolated pawn penalties
      8.  Piece coordination — bishop pair bonus
      9.  Endgame king activity — push king forward when winning
      10. Check bonus
    """
    score = 0.0
    endgame = is_endgame(board)

    # ── 3. Mobility ──────────────────────────────────────────────────────────
    w_mob = len(get_pseudo_legal_moves(board, True))
    b_mob = len(get_pseudo_legal_moves(board, False))
    score += 4 * (w_mob - b_mob)

    # Track per-column pawn counts for structure eval
    w_pawns_per_col = [0] * BOARD_SIZE
    b_pawns_per_col = [0] * BOARD_SIZE
    w_bishop_count  = 0
    b_bishop_count  = 0

    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            piece = board[row][col]
            if piece == EMPTY:
                continue

            # ── 1. Material ───────────────────────────────────────────────
            score += PIECE_VALUES.get(piece, 0)

            # ── 2. Piece-square tables ────────────────────────────────────
            if is_white(piece) and piece in PST:
                pst_val = PST[piece][row][col]
                # Endgame: king should be active, not hiding
                if piece == WHITE_KING and endgame:
                    pst_val = -pst_val * 0.5   # flip penalty → reward
                score += pst_val

            elif is_black(piece):
                equiv = piece - 5
                if equiv in PST:
                    pst_val = PST[equiv][BOARD_SIZE - 1 - row][col]
                    if piece == BLACK_KING and endgame:
                        pst_val = -pst_val * 0.5
                    score -= pst_val

            # ── 4. Centre control ─────────────────────────────────────────
            if row in [2, 3] and col in [2, 3]:
                score += 18 if is_white(piece) else -18

            # Track for structure
            if piece == WHITE_PAWN:  w_pawns_per_col[col] += 1
            if piece == BLACK_PAWN:  b_pawns_per_col[col] += 1
            if piece == WHITE_BISHOP: w_bishop_count += 1
            if piece == BLACK_BISHOP: b_bishop_count += 1

    # ── 5. King safety ────────────────────────────────────────────────────────
    for col in range(BOARD_SIZE):
        # White king — find dynamically (works for any back rank)
        if board[0][col] == WHITE_KING:
            shield = 0
            if in_bounds(1, col)   and board[1][col]   == WHITE_PAWN: shield += 25
            if in_bounds(1, col-1) and board[1][col-1] == WHITE_PAWN: shield += 12
            if in_bounds(1, col+1) and board[1][col+1] == WHITE_PAWN: shield += 12
            score += shield
            # Penalty if king is on open file (no pawn cover ahead)
            if shield == 0:
                score -= 30

        if board[5][col] == BLACK_KING:
            shield = 0
            if in_bounds(4, col)   and board[4][col]   == BLACK_PAWN: shield += 25
            if in_bounds(4, col-1) and board[4][col-1] == BLACK_PAWN: shield += 12
            if in_bounds(4, col+1) and board[4][col+1] == BLACK_PAWN: shield += 12
            score -= shield
            if shield == 0:
                score += 30

    # ── 6. Passed pawns ───────────────────────────────────────────────────────
    # White pawn is "passed" if no black pawn can block or capture it
    for col in range(BOARD_SIZE):
        for row in range(BOARD_SIZE):
            if board[row][col] == WHITE_PAWN:
                blocked = False
                for r in range(row + 1, BOARD_SIZE):
                    for dc in [-1, 0, 1]:
                        nc = col + dc
                        if in_bounds(r, nc) and board[r][nc] == BLACK_PAWN:
                            blocked = True
                            break
                    if blocked: break
                if not blocked:
                    # Bonus scales with how advanced the pawn is
                    score += 15 + (row * 8)

            if board[row][col] == BLACK_PAWN:
                blocked = False
                for r in range(row - 1, -1, -1):
                    for dc in [-1, 0, 1]:
                        nc = col + dc
                        if in_bounds(r, nc) and board[r][nc] == WHITE_PAWN:
                            blocked = True
                            break
                    if blocked: break
                if not blocked:
                    score -= 15 + ((BOARD_SIZE - 1 - row) * 8)

    # ── 7. Pawn structure ─────────────────────────────────────────────────────
    for col in range(BOARD_SIZE):
        # Doubled pawns — two pawns on same file
        if w_pawns_per_col[col] > 1:
            score -= 20 * (w_pawns_per_col[col] - 1)
        if b_pawns_per_col[col] > 1:
            score += 20 * (b_pawns_per_col[col] - 1)

        # Isolated pawns — no friendly pawn on adjacent files
        if w_pawns_per_col[col] > 0:
            left  = w_pawns_per_col[col-1] if col > 0 else 0
            right = w_pawns_per_col[col+1] if col < BOARD_SIZE-1 else 0
            if left == 0 and right == 0:
                score -= 15 * w_pawns_per_col[col]

        if b_pawns_per_col[col] > 0:
            left  = b_pawns_per_col[col-1] if col > 0 else 0
            right = b_pawns_per_col[col+1] if col < BOARD_SIZE-1 else 0
            if left == 0 and right == 0:
                score += 15 * b_pawns_per_col[col]

    # ── 8. Bishop pair bonus ──────────────────────────────────────────────────
    # Two bishops are very strong on open 6x6 board
    if w_bishop_count >= 2: score += 40
    if b_bishop_count >= 2: score -= 40

    # ── 10. Check bonus ───────────────────────────────────────────────────────
    if is_in_check(board, True):  score -= 90
    if is_in_check(board, False): score += 90

    return score

# ---------------------------------------------------------------------------
# Move ordering — critical for alpha-beta efficiency
# ---------------------------------------------------------------------------

# Killer move table: stores 2 killer moves per depth
# Killer moves are quiet moves that caused a beta cutoff — try them early
_killer_moves = [[None, None] for _ in range(20)]

def update_killer(move, depth):
    if depth < 20:
        if _killer_moves[depth][0] != move:
            _killer_moves[depth][1] = _killer_moves[depth][0]
            _killer_moves[depth][0] = move

def score_move_for_ordering(board: np.ndarray, move, depth: int = 0) -> int:
    piece, sr, sc, dr, dc, promo = move
    target = board[dr][dc]
    order_score = 0

    # 1. Promotions — always search first
    if promo is not None:
        order_score += 900

    # 2. Winning captures — MVV-LVA
    #    Most Valuable Victim / Least Valuable Attacker
    if target != EMPTY:
        victim_val   = abs(PIECE_VALUES.get(target, 0))
        attacker_val = abs(PIECE_VALUES.get(piece, 0))
        see = victim_val - attacker_val  # Static Exchange Evaluation approx
        if see >= 0:
            order_score += 700 + see       # Winning capture
        else:
            order_score += 100 + see       # Losing capture (search last)

    # 3. Killer moves — quiet moves that were good at this depth before
    elif depth < 20:
        if move == _killer_moves[depth][0]: order_score += 80
        elif move == _killer_moves[depth][1]: order_score += 70

    # 4. Centre moves bonus
    if dr in [2, 3] and dc in [2, 3]:
        order_score += 25

    # 5. Pawn advances toward promotion
    if piece == WHITE_PAWN: order_score += dr * 3
    if piece == BLACK_PAWN: order_score += (BOARD_SIZE - 1 - dr) * 3

    return order_score


def order_moves(board: np.ndarray, moves: list, depth: int = 0) -> list:
    return sorted(moves, key=lambda m: score_move_for_ordering(board, m, depth), reverse=True)

# ---------------------------------------------------------------------------
# Format move string
# ---------------------------------------------------------------------------

def format_move(piece: int, src_row: int, src_col: int,
                dst_row: int, dst_col: int, promo_piece: Optional[int] = None) -> str:
    src_cell = idx_to_cell(src_row, src_col)
    dst_cell = idx_to_cell(dst_row, dst_col)
    move_str = f"{piece}:{src_cell}->{dst_cell}"
    if promo_piece is not None:
        move_str += f"={promo_piece}"
    return move_str

# ---------------------------------------------------------------------------
# Minimax + Alpha-Beta + Null Move Pruning + Quiescence Search
# ---------------------------------------------------------------------------

def quiescence(board: np.ndarray, alpha: float, beta: float,
               maximizing: bool, qdepth: int = 4) -> float:
    """
    Quiescence search — keep searching captures until position is 'quiet'.
    Prevents the horizon effect where engine misses a capture just beyond
    its search depth.
    """
    stand_pat = evaluate(board)

    if maximizing:
        if stand_pat >= beta: return beta
        alpha = max(alpha, stand_pat)
    else:
        if stand_pat <= alpha: return alpha
        beta = min(beta, stand_pat)

    if qdepth == 0:
        return stand_pat

    # Only look at captures in quiescence
    all_moves = get_legal_moves(board, maximizing)
    captures  = [m for m in all_moves if board[m[3]][m[4]] != EMPTY or m[5] is not None]
    captures  = order_moves(board, captures)

    for move in captures:
        new_board = apply_move(board, *move)
        score = quiescence(new_board, alpha, beta, not maximizing, qdepth - 1)

        if maximizing:
            alpha = max(alpha, score)
            if alpha >= beta: return beta
        else:
            beta = min(beta, score)
            if beta <= alpha: return alpha

    return alpha if maximizing else beta


def minimax(board: np.ndarray, depth: int, alpha: float, beta: float,
            maximizing: bool) -> float:
    """
    Minimax + Alpha-Beta + move ordering + killer moves + quiescence.
    """
    if depth == 0:
        # Drop into quiescence instead of raw evaluate
        return quiescence(board, alpha, beta, maximizing)

    moves = get_legal_moves(board, maximizing)

    if not moves:
        if is_in_check(board, maximizing):
            return (-99999 - depth) if maximizing else (99999 + depth)
        return 0  # Stalemate

    moves = order_moves(board, moves, depth)

    if maximizing:
        max_score = float('-inf')
        for move in moves:
            new_board = apply_move(board, *move)
            score = minimax(new_board, depth - 1, alpha, beta, False)
            if score > max_score:
                max_score = score
            alpha = max(alpha, score)
            if beta <= alpha:
                update_killer(move, depth)   # record killer
                break
        return max_score
    else:
        min_score = float('inf')
        for move in moves:
            new_board = apply_move(board, *move)
            score = minimax(new_board, depth - 1, alpha, beta, True)
            if score < min_score:
                min_score = score
            beta = min(beta, score)
            if beta <= alpha:
                update_killer(move, depth)
                break
        return min_score

# ---------------------------------------------------------------------------
# Iterative Deepening — search depth 1,2,3...N
# Better move ordering at each level, and can stop early if time is tight
# ---------------------------------------------------------------------------

def get_best_move(board: np.ndarray, playing_white: bool = True,
                  depth: int = 4) -> Optional[str]:
    """
    Given the current board state, return the best move string.

    Parameters
    ----------
    board        : 6x6 NumPy array representing the current game state.
    playing_white: True if the engine is playing as White, False for Black.
    depth        : Max search depth (default 4).

    Returns
    -------
    Move string '<piece_id>:<src_cell>-><dst_cell>', or None if no moves.
    """
    # Reset killer moves for new search
    global _killer_moves
    _killer_moves = [[None, None] for _ in range(20)]

    all_moves = get_legal_moves(board, playing_white)
    if not all_moves:
        return None

    best_move  = None
    best_score = float('-inf') if playing_white else float('inf')

    # ── Iterative Deepening ──────────────────────────────────────────────────
    # Search depth 1 first, use result to order moves for depth 2, etc.
    # This dramatically improves alpha-beta pruning efficiency.
    for current_depth in range(1, depth + 1):
        iter_best_move  = None
        iter_best_score = float('-inf') if playing_white else float('inf')

        # Order moves using results from previous iteration
        all_moves = order_moves(board, all_moves, current_depth)

        for move in all_moves:
            new_board = apply_move(board, *move)
            score = minimax(new_board, current_depth - 1,
                            float('-inf'), float('inf'), not playing_white)

            if playing_white and score > iter_best_score:
                iter_best_score = score
                iter_best_move  = move
            elif not playing_white and score < iter_best_score:
                iter_best_score = score
                iter_best_move  = move

        if iter_best_move is not None:
            best_move  = iter_best_move
            best_score = iter_best_score

            # Bring best move to front for next iteration ordering
            all_moves.remove(best_move)
            all_moves.insert(0, best_move)

    return format_move(*best_move)

# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

# if __name__ == "__main__":
#     initial_board = np.array([
#         [ 2,  3,  4,  5,  3,  2],
#         [ 1,  1,  1,  1,  1,  1],
#         [ 0,  0,  0,  0,  0,  0],
#         [ 0,  0,  0,  0,  0,  0],
#         [ 6,  6,  6,  6,  6,  6],
#         [ 7,  8,  9, 10,  8,  7],
#     ], dtype=int)
#     print("Board:\n", initial_board)
#     move = get_best_move(initial_board, playing_white=True)
#     print("Best move for White:", move)