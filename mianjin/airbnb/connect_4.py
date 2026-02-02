'''
Input: int[] plays = [0, 1, 2, 1, ...] 代表user放checker進哪一個column， 你可以assume 兩個user來回放，board 是7x6的matrix Implement： DropDisc(column) CheckWinner(player) PrintBoard() Follow-up问能不能优化?
'''


class Connect4:
    ROWS = 6
    COLS = 7

    def __init__(self):
        # board[row][col], row=0 is bottom for easy dropping
        self.board = [[0] * self.COLS for _ in range(self.ROWS)]
        self.heights = [0] * self.COLS
        self.turn = 1  # player 1 starts

        # Useful for optimization follow-up
        self.last_move = None  # (row, col, player)

    def DropDisc(self, col):
        if col < 0 or col >= self.COLS:
            return False  # invalid column

        row = self.heights[col]
        if row >= self.ROWS:
            return False  # column is full

        player = self.turn
        self.board[row][col] = player
        self.heights[col] += 1

        self.last_move = (row, col, player)

        # switch player
        self.turn = 2 if self.turn == 1 else 1
        return True

    def CheckWinner(self, player):
        b = self.board
        R, C = self.ROWS, self.COLS

        # Check horizontal
        for r in range(R):
            for c in range(C - 3):
                if b[r][c] == player and b[r][c+1] == player and b[r][c+2] == player and b[r][c+3] == player:
                    return True

        # Check vertical
        for r in range(R - 3):
            for c in range(C):
                if b[r][c] == player and b[r+1][c] == player and b[r+2][c] == player and b[r+3][c] == player:
                    return True

        # Check diagonal up-right (r increases, c increases)
        for r in range(R - 3):
            for c in range(C - 3):
                if b[r][c] == player and b[r+1][c+1] == player and b[r+2][c+2] == player and b[r+3][c+3] == player:
                    return True

        # Check diagonal down-right (r decreases, c increases)
        for r in range(3, R):
            for c in range(C - 3):
                if b[r][c] == player and b[r-1][c+1] == player and b[r-2][c+2] == player and b[r-3][c+3] == player:
                    return True

        return False

    def PrintBoard(self):
        # print from top row to bottom row for human-friendly view
        for r in range(self.ROWS - 1, -1, -1):
            print(" ".join(str(self.board[r][c]) for c in range(self.COLS)))
        print("-" * (2 * self.COLS - 1))
        print(" ".join(str(c) for c in range(self.COLS)))


##### test case #####
g = Connect4()
plays = [0,1, 1,2, 2,3, 2,3, 3,4, 3,4, 3]
for c in plays:
    assert g.DropDisc(c)

assert g.CheckWinner(1) == True

'''
optimize:

Check winner using only the last move (best practical answer)

Instead of scanning the entire board, after each DropDisc, you only need to check lines passing through (row, col) in 4 directions:

horizontal, vertical, diag1, diag2

That becomes O(1) per move (bounded by at most 3 steps each side × 4 directions).

If you want, I can provide the CheckWinnerLastMove() function that does this cleanly.
'''


