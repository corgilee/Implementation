'''
Given A and B two interval lists, A has no overlap inside A and B has no overlap inside B. Write the function to merge two interval lists, output the result with no overlap. Ask for a very efficient solution

A naive method can combine the two list, and sort and apply merge interval in the leetcode, but is not efficient enough.

For example,
A: [1,5], [10,14], [16,18]
B: [2,6], [8,10], [11,20]

output [1,6], [8, 20]

https://leetcode.com/discuss/interview-question/124616/Merge-two-interval-lists
'''
A = [[1,5],[10,14],[16,18]]
B = [[2,6],[8,10],[11,20]]

'''
思路，两个指针，p1 for A, p2 for B, 根据指针位置 和 A[0],B[0]的大小决定curr的取值

每一个循环都更新curr，然后和stack里面的top区间做比较，如果和stack top的有重叠，更新stack top
'''
p1,p2=0,0
n1,n2=len(A),len(B)
res=[]

while p1<n1 or p2<n2:
    # 分情况讨论，curr改assign什么值
    if p1==n1:
        curr=B[p2]
        p2+=1
    elif p2==n2:
        curr=A[p1]
        p1+=1
    elif A[p1][0]<B[p2][0]:
        # 比较起始点
        curr=A[p1]
        p1+=1
    else:
        curr=B[p2]
        p2+=1
    
    #比较 curr 和res
    if res and res[-1][1]>=curr[0]:
        res[-1][1]=max(res[-1][1],curr[1])

    else:
        res.append(curr)

print(res)

        














'''
i = 0
j = 0
res = []
while i < len(A) or j < len(B): # 这里注意是or，不是and
    if i==len(A):
        print('case 1')
        curr = B[j]
        j+=1
    elif j==len(B):
        print('case 2')
        curr = A[i]
        i+=1
    elif A[i][0] < B[j][0]:
        curr = A[i]
        i+=1
    else:
        curr = B[j]
        j+=1
    if res and res[-1][-1] >= curr[0]:
        res[-1][-1] = max(res[-1][-1],curr[-1])
    else:
        res.append(curr)
    print(i,j)
    

print(res)
'''
