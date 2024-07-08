'''
https://leetcode.com/discuss/interview-question/algorithms/124758/Pinterest-or-Find-the-index-of-the-pair-wise-sum

Given two arrays, (say [1,3,5] and [2,4]) and a number target (say 7), assume we sort by the sum of any pair of elements from each array, return the smallest index.

In this example, the result is 3.
Pair, Sum, Index
(1,2), 3, 0
(1,4), 5, 1
(3,2), 5, 2
(3,4), 7, 3 <- result
(5,2), 7, 4
(5,4), 9, 5

'''

arr1=[1,3,5]
arr2=[2,4]

from bisect import bisect_left
def smallest_sum_index(arr1, arr2, target):
    found = False
    res = 0
    for num1 in arr1:        #O(n)
        diff = target - num1
        number_of_combine = bisect_left(arr2, diff) #bisect_left 本身是log(m)
        # with num1, nubmer_of_combine 就是diff插入arr2 的index
        res += number_of_combine
        if number_of_combine < len(arr2) and arr2[number_of_combine] + num1 == target:
            # 如果number_
            found = True
            break

    return res if found else -1

# 如果要define bisect_left:
def bisect_left(arr, target):
    left, right = 0, len(arr)
    
    while left < right:
        mid = (left + right) // 2
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid
    
    return left

assert smallest_sum_index([1,3,5], [2,4], 7) == 3
assert smallest_sum_index([1,3,5], [2,4], 9) == 5
assert smallest_sum_index([1,3,5], [2,4], 10) == -1