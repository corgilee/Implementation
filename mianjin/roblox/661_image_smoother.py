class Solution:
    def imageSmoother(self, img: List[List[int]]) -> List[List[int]]:
        '''
        steps:
        1. For each cell (i, j): Look at the 3x3 window centered at (i, j) → 9 potential cells.
            Only include valid neighbors (those within bounds).
            Compute average = floor(sum / count)
        2. Store the result in a new matrix.

        Time: O(m × n × 9) ≈ O(m × n)
        Space: O(m × n) for the result matrix
        
        '''
        # Get dimensions of the input matrix
        m = len(img)
        n = len(img[0])

        # Initialize the output matrix with same dimensions filled with zeros
        result = [[0 for _ in range(n)] for _ in range(m)]

        # Define the 8 directions plus the cell itself (total 9)
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            ( 0, -1), ( 0, 0), ( 0, 1),
            ( 1, -1), ( 1, 0), ( 1, 1)
        ]

        # Traverse every cell in the input matrix
        for i in range(m):
            for j in range(n):
                total = 0  # Sum of valid neighbor values
                count = 0  # Count of valid neighbor cells

                # Check all 9 directions
                for dx, dy in directions:
                    ni = i + dx  # Neighbor row index
                    nj = j + dy  # Neighbor col index

                    # Check if neighbor is within bounds
                    if 0 <= ni < m and 0 <= nj < n:
                        total += img[ni][nj]
                        count += 1

                # Compute floor of average and store in result matrix
                result[i][j] = total // count

        return result
