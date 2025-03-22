'''
Strategy: Transpose + Reverse

To rotate a matrix 90° clockwise in-place:

Transpose the matrix (convert rows to columns).
Reverse each row.

This avoids using extra space and modifies the matrix directly.

Time: O(n²)

Space: O(1) (in-place)


'''


def rotate(matrix):
        """
        Do not return anything, modify matrix in-place instead.
        """
        n = len(matrix)

        # Step 1: Transpose the matrix
        for i in range(n):
            for j in range(i + 1, n):
                # Swap symmetric elements across the diagonal
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

        # Step 2: Reverse each row
        for row in matrix:
            row.reverse()
