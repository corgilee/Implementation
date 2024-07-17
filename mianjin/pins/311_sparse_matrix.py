class Solution:
    def multiply(self, mat1: List[List[int]], mat2: List[List[int]]) -> List[List[int]]:
        '''
        定义一个compressed function
        用一个condense matrix 储存 mat1，一个 condense matrix 储存 mat2
        matrix multiplication C(m,n)+=A(m,k)*B(k,n),只要 A.c==B.r, 那么C[r1][c2]+=v1*v2
        T:O(m1 * n1 + m2 * n2 + a * b)
        '''

        def compressed_matrix(mat):
            m=len(mat)
            n=len(mat[0])
            dense=[]
            for i in range(m):
                for j in range(n):
                    if mat[i][j]!=0:
                        dense.append((i,j,mat[i][j]))
            return dense

        A=compressed_matrix(mat1)
        B=compressed_matrix(mat2)

        m=len(mat1)
        n=len(mat2[0])
        res=[[0]*n for _ in range(m)]


        for r1,c1,v1 in A:
            for r2,c2,v2 in B:
                if c1==r2:
                    res[r1][c2]+=v1*v2
        
        return res



        