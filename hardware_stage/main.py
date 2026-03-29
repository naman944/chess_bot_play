import game
import numpy as np
import serial
import sys
import threading
import time

# ── Match configuration ────────────────────────────────────────────────────────
PLAYING_WHITE = True   # Set to False if your team is assigned Black pieces

# ── Board geometry — CALIBRATE TOP_LEFT_X / TOP_LEFT_Y for the arm's frame ───
# SQUARE_SIZE matches the rulebook (360 mm board / 6 = 60 mm per square).
# TOP_LEFT_X/Y must be the arm's XY coordinates for the top-left corner of the
# board (cell A1).  Jog the arm to A1, run {"T":105} to read the position,
# and set these values accordingly.
SQUARE_SIZE = 60      # mm per square (rulebook §3.3 — do not change)
TOP_LEFT_X  = 180     # arm-frame x of board top-left (mm) — CALIBRATE
TOP_LEFT_Y  = 180     # arm-frame y of board top-left (mm) — CALIBRATE

FILE_TO_COL = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5}

# ── Robot arm motion parameters — CALIBRATE for your physical setup ───────────
Z_HOVER = 150   # safe travel height above the board (mm)
Z_PICK  = 42    # height to engage electromagnet with piece (mm)
                # pieces are 40 mm tall cylinders — tune this to just above top
T_ANGLE = 3.14  # wrist angle pointing straight down (rad)
SPD     = 1000  # speed (steps/s, 0 = max)
ACC     = 100   # acceleration (units of 100 steps/s²)

# ── Piece collection area (arm coordinates, mm) — CALIBRATE ───────────────────
# Rulebook §5.4: captured/promoted pieces must be placed in the designated
# collection area closest to the robotic arm.
COLLECTION_X = 300    # CALIBRATE
COLLECTION_Y =   0    # CALIBRATE
COLLECTION_Z = Z_PICK # same pick height as the board

PIECE_NAMES = {
    1: 'White Pawn',   2: 'White Knight', 3: 'White Bishop',
    4: 'White Queen',  5: 'White King',
    6: 'Black Pawn',   7: 'Black Knight', 8: 'Black Bishop',
    9: 'Black Queen',  10: 'Black King',
}

# ── Logging helper ────────────────────────────────────────────────────────────
def log(msg: str):
    """Print a timestamped status line."""
    ts = time.strftime('%H:%M:%S')
    print(f"[{ts}] {msg}")


# ── Start perception in a background thread ───────────────────────────────────
log("Starting perception thread (camera + ArUco board detection)...")

def _start_perception():
    import perception

_perception_thread = threading.Thread(target=_start_perception, daemon=True)
_perception_thread.start()
log("Perception thread launched — waiting 2 s for module initialisation...")
time.sleep(2)   # give the module time to initialise its module-level variables

perception = sys.modules.get('perception')  # dict lookup — no import lock

if perception is not None:
    log("Perception module loaded OK.")
else:
    log("WARNING: Perception module not yet in sys.modules — board state may be unavailable.")

# ── Hardware ──────────────────────────────────────────────────────────────────
log("Connecting to COM3 (solenoid / electromagnet)...")
ser  = serial.Serial('COM3', 115200)
log("COM3 connected (solenoid).")

log("Connecting to COM4 (robotic arm)...")
ser2 = serial.Serial('COM4', 115200)
ser2.setRTS(False)
ser2.setDTR(False)
log("COM4 connected (robotic arm). RTS/DTR cleared.")


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_board_state() -> np.ndarray:
    """Return the latest board state from the perception module."""
    if perception is not None and perception.prev_board is not None:
        return perception.prev_board.copy()
    return np.zeros((6, 6), dtype=int)


def cell_to_xy(cell: str):
    """Convert cell notation (e.g. 'B3') to arm XY coordinates in mm."""
    col = FILE_TO_COL[cell[0].upper()]
    row = int(cell[1]) - 1
    x = TOP_LEFT_X - (row * SQUARE_SIZE + SQUARE_SIZE / 2)
    y = TOP_LEFT_Y - (col * SQUARE_SIZE + SQUARE_SIZE / 2)
    return x, y


def _arm_cmd(x, y, z) -> str:
    """Build a T:104 blocking Cartesian-move JSON command (Waveshare format)."""
    return (f'{{"T":104,"x":{x:.1f},"y":{y:.1f},"z":{z:.1f},'
            f'"t":{T_ANGLE},"spd":{SPD},"acc":{ACC}}}')


def movetocmd(move_str: str) -> list:
    """Convert a move string to an ordered list of (step, description) tuples.

    Each step is either a JSON command string for send_cmd(), or the special
    tokens 'pick' / 'place' for the electromagnet.

    Move format:  '<piece_id>:<src_cell>-><dst_cell>'
    Promotion  :  '<piece_id>:<src_cell>-><dst_cell>=<promo_piece_id>'
    """
    _, cell_part = move_str.split(':')
    src_cell, rest = cell_part.split('->')
    dst_cell = rest.split('=')[0]

    sx, sy = cell_to_xy(src_cell)
    dx, dy = cell_to_xy(dst_cell)

    return [
        (_arm_cmd(sx, sy, Z_HOVER), f"Arm hovering above source {src_cell}  (x={sx:.1f}, y={sy:.1f}, z={Z_HOVER})"),
        (_arm_cmd(sx, sy, Z_PICK),  f"Arm lowering onto piece at {src_cell} (z={Z_PICK})"),
        ('pick',                     "Electromagnet ON  — grabbing piece"),
        (_arm_cmd(sx, sy, Z_HOVER), f"Arm lifting from {src_cell}           (z={Z_HOVER})"),
        (_arm_cmd(dx, dy, Z_HOVER), f"Arm travelling to above {dst_cell}    (x={dx:.1f}, y={dy:.1f}, z={Z_HOVER})"),
        (_arm_cmd(dx, dy, Z_PICK),  f"Arm lowering to board at {dst_cell}   (z={Z_PICK})"),
        ('place',                    "Electromagnet OFF — releasing piece"),
        (_arm_cmd(dx, dy, Z_HOVER), f"Arm raising clear of {dst_cell}       (z={Z_HOVER})"),
    ]


def _collect_piece(cell: str):
    """Pick up the piece on 'cell' and drop it in the collection area."""
    cx, cy = cell_to_xy(cell)
    steps = [
        (_arm_cmd(cx, cy, Z_HOVER),                       f"Arm hovering above captured piece at {cell}"),
        (_arm_cmd(cx, cy, Z_PICK),                        f"Arm lowering to capture piece at {cell}"),
        ('pick',                                           "Electromagnet ON  — grabbing captured piece"),
        (_arm_cmd(cx, cy, Z_HOVER),                       f"Arm lifting captured piece from {cell}"),
        (_arm_cmd(COLLECTION_X, COLLECTION_Y, Z_HOVER),   "Arm moving to collection area (hover)"),
        (_arm_cmd(COLLECTION_X, COLLECTION_Y, COLLECTION_Z), "Arm lowering into collection area"),
        ('place',                                          "Electromagnet OFF — dropping captured piece in collection"),
        (_arm_cmd(COLLECTION_X, COLLECTION_Y, Z_HOVER),   "Arm raising clear of collection area"),
    ]
    for step, desc in steps:
        _dispatch(step, desc)


def _dispatch(step: str, desc: str = ""):
    """Send one step — either a JSON arm command or a pick/place action."""
    if desc:
        log(f"  >> {desc}")
    if step == 'pick':
        pick()
    elif step == 'place':
        place()
    else:
        send_cmd(step)


def pick():
    """Energise the electromagnet to grab a piece."""
    ser.write(b'1')


def place():
    """De-energise the electromagnet to release a piece."""
    ser.write(b'0')


def send_cmd(command: str):
    """Send a single JSON command to the arm via serial and wait for reply."""
    log(f"     ARM TX: {command}")
    ser2.write((command + '\n').encode())
    response = ser2.readline().decode().strip()
    log(f"     ARM RX: {response}")


def log_move(move_str: str):
    """Append the move to move_log.txt (required by rulebook §5.4)."""
    with open('move_log.txt', 'a') as f:
        f.write(move_str + '\n')
    log(f"Move logged to move_log.txt: {move_str}")


def _piece_name(piece_id: int) -> str:
    return PIECE_NAMES.get(piece_id, f"Piece#{piece_id}")


def execute_move(move_str: str, board_state: np.ndarray):
    """Execute a complete move: home → [capture] → move → home.

    Handles:
    - Captures  (§5.4): captured piece is moved to the collection area first.
    - Promotions(§5.4): pawn is moved to the collection area; Arm Player is
                        prompted to place the promoted piece manually.
    - Safety    (§9)  : arm returns to init position before and after every move.
    """
    piece_id_str, cell_part = move_str.split(':')
    src_cell, rest = cell_part.split('->')
    dst_cell   = rest.split('=')[0]
    is_promo   = '=' in rest
    piece_id   = int(piece_id_str)

    dst_col = FILE_TO_COL[dst_cell[0].upper()]
    dst_row = int(dst_cell[1]) - 1

    captured_id = board_state[dst_row][dst_col]

    log(f"")
    log(f"{'='*55}")
    log(f"  EXECUTING MOVE: {move_str}")
    log(f"  Piece  : {_piece_name(piece_id)} (ID={piece_id})")
    log(f"  Source : {src_cell}  →  Destination : {dst_cell}")
    if captured_id != 0:
        log(f"  Capture: {_piece_name(captured_id)} (ID={captured_id}) on {dst_cell}")
    if is_promo:
        promo_id = int(rest.split('=')[1])
        log(f"  Promotion → {_piece_name(promo_id)} (ID={promo_id})")
    log(f"{'='*55}")

    # Safety: return to home before starting (rulebook §9)
    log("Sending arm to HOME position (safety, §9)...")
    send_cmd('{"T":100}')
    log("Arm at HOME.")

    # Handle capture: remove enemy piece to collection area first (§5.4)
    if captured_id != 0:
        log(f"CAPTURE PHASE: removing {_piece_name(captured_id)} from {dst_cell} to collection area...")
        _collect_piece(dst_cell)
        log("Captured piece cleared from board.")

    if is_promo:
        # Promotion: move the pawn to the collection area (§5.4)
        log(f"PROMOTION PHASE: moving pawn from {src_cell} to collection area...")
        cx, cy = cell_to_xy(src_cell)
        promo_steps = [
            (_arm_cmd(cx, cy, Z_HOVER),                       f"Arm hovering above pawn at {src_cell}"),
            (_arm_cmd(cx, cy, Z_PICK),                        f"Arm lowering to pawn at {src_cell}"),
            ('pick',                                           "Electromagnet ON  — grabbing pawn for promotion"),
            (_arm_cmd(cx, cy, Z_HOVER),                       f"Arm lifting pawn from {src_cell}"),
            (_arm_cmd(COLLECTION_X, COLLECTION_Y, Z_HOVER),   "Arm moving pawn to collection area (hover)"),
            (_arm_cmd(COLLECTION_X, COLLECTION_Y, COLLECTION_Z), "Arm lowering pawn into collection area"),
            ('place',                                          "Electromagnet OFF — pawn placed in collection"),
            (_arm_cmd(COLLECTION_X, COLLECTION_Y, Z_HOVER),   "Arm raising clear of collection area"),
        ]
        for step, desc in promo_steps:
            _dispatch(step, desc)

        log("Sending arm to HOME (before manual promotion placement)...")
        send_cmd('{"T":100}')
        log("Arm at HOME.")
        log(f"ACTION REQUIRED: manually place promoted piece on {dst_cell}, then press Enter.")
        input(f">>> Place the promoted piece on {dst_cell} and press Enter to continue...")
        log("User confirmed — promoted piece placed.")
    else:
        # Normal move
        log(f"MOVE PHASE: executing {src_cell} → {dst_cell}...")
        for step, desc in movetocmd(move_str):
            _dispatch(step, desc)
        log("Move phase complete.")

    # Safety: return to home after move (rulebook §9)
    log("Sending arm to HOME position (safety, §9)...")
    send_cmd('{"T":100}')
    log("Arm at HOME.")

    # Log the move (rulebook §5.4)
    log_move(move_str)
    log(f"Move {move_str} complete.\n")


if __name__ == "__main__":
    log("RoboGambit starting up...")
    log(f"Playing as: {'WHITE' if PLAYING_WHITE else 'BLACK'}")
    log(f"Board config: SQUARE_SIZE={SQUARE_SIZE}mm, TOP_LEFT=({TOP_LEFT_X},{TOP_LEFT_Y})")

    game.reset_game_state()
    log("Game state reset.")

    # Wait until perception has locked the homography and seen the board
    log("Waiting for camera to detect board and lock homography...")
    wait_count = 0
    while True:
        board_state = get_board_state()
        if np.any(board_state):
            break
        wait_count += 1
        if wait_count % 20 == 0:   # print a reminder every ~2 s
            log("  ...still waiting for board state from camera...")
        time.sleep(0.1)

    log("Board state received from camera!")
    log(f"Board coordinates (ArUco piece IDs on 6x6 grid):\n{board_state}")

    # Main match loop
    turn = 0
    while True:
        turn += 1
        log(f"--- TURN {turn} --- Press Enter when ready for autonomous move ---")
        input("\n>>> Press Enter to start your autonomous turn...")

        log("Reading current board state from camera...")
        board_state = get_board_state()
        log(f"Board state:\n{board_state}")

        log("Computing best move (minimax search)...")
        best_move = game.get_best_move(board_state, PLAYING_WHITE)

        if best_move is None:
            log("No legal moves available — GAME OVER.")
            break

        side = 'White' if PLAYING_WHITE else 'Black'
        log(f"Best move for {side}: {best_move}")

        execute_move(best_move, board_state)