def print_pascals_triangle(n):
    triangle=[[1]*(i+1) for i in range(n)]

    for i in range(2,n):
        for j in range(2,i):
            triangle[i][j]=triangle[i-1][j-1]+triangle[i-1][j]

    max_width=len(" ".join(map(str,triangle[-1])))
    print('max_width',max_width)
    for line in triangle:
        print(" ".join(map(str,line)).center(max_width))

print_pascals_triangle(10)