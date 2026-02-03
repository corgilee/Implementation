'''
You are given:

A 2D grid grid representing a map:
'O' = robot
'E' = empty
'X' = blocker

A query vector q = [L, T, B, R] representing exact distances from a robot cell to the nearest blocker in the four directions:

L: distance to the closest blocker when moving left
T: distance to the closest blocker when moving up
B: distance to the closest blocker when moving down
R: distance to the closest blocker when moving right

Important: The grid boundary is also considered a blocker.
Distance definition: number of empty steps from the robot cell until you hit the first blocker/boundary in that direction.

If the neighbor cell in that direction is a blocker/boundary, the distance is 0.

Task: Return coordinates (row, col) of all robots 'O' whose four directional distances match the query exactly.
'''

def find_robots_by_blocker_dist(grid, query):
    """
    grid: List[List[str]] or List[str]
          Each cell is one of {'O', 'E', 'X'}.
    query: [L, T, B, R] distances to nearest blocker/boundary.

    Returns: List[Tuple[int,int]] of (row, col) for robots matching query.
    """

    # Normalize grid to List[List[str]], optional
    if not grid:
        return []
    if isinstance(grid[0], str):
        grid = [list(row) for row in grid]

    R, C = len(grid), len(grid[0])
    Lq, Tq, Bq, Rq = query

    # Precompute distances to nearest blocker/boundary in 4 directions
    left = [[0] * C for _ in range(R)]
    right = [[0] * C for _ in range(R)]
    top = [[0] * C for _ in range(R)]
    bottom = [[0] * C for _ in range(R)]

    # Left distances (scan each row left -> right)
    for i in range(R):
        last_blocker = -1  # boundary acts like blocker at col = -1
        for j in range(C):
            if grid[i][j] == 'X':
                last_blocker = j
                left[i][j] = 0
            else:
                left[i][j] = j - last_blocker - 1

    # Right distances (scan each row right -> left)
    for i in range(R):
        last_blocker = C  # boundary acts like blocker at col = C
        for j in range(C - 1, -1, -1):
            if grid[i][j] == 'X':
                last_blocker = j
                right[i][j] = 0
            else:
                right[i][j] = last_blocker - j - 1

    # Top distances (scan each col top -> bottom)
    for j in range(C):
        last_blocker = -1  # boundary at row = -1
        for i in range(R):
            if grid[i][j] == 'X':
                last_blocker = i
                top[i][j] = 0
            else:
                top[i][j] = i - last_blocker - 1

    # Bottom distances (scan each col bottom -> top)
    for j in range(C):
        last_blocker = R  # boundary at row = R
        for i in range(R - 1, -1, -1):
            if grid[i][j] == 'X':
                last_blocker = i
                bottom[i][j] = 0
            else:
                bottom[i][j] = last_blocker - i - 1

    # Collect matching robots
    ans = []
    for i in range(R):
        for j in range(C):
            if grid[i][j] != 'O':
                continue
            if (left[i][j] == Lq and top[i][j] == Tq and
                bottom[i][j] == Bq and right[i][j] == Rq):
                ans.append((i, j))

    return ans


# Example usage:
grid = [
    ['X','E','X'],
    ['E','O','E'],
    ['X','E','X']
]

print(find_robots_by_blocker_dist(grid, [1,1,1,1]))