for i, seed in enumerate([42, 99, 7, 2025, 314]):
    board = make_random_board(seed)
    side = random.choice([True, False])
    legal = get_legal_moves(board, side)
    expect = len(legal) > 0
    results.append(run_test(
        f"TEST {9+i}: Random board (seed={seed}, {'White' if side else 'Black'})",
        board, side, expect_move=expect
    ))