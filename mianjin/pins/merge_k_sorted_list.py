'''
大概一共有 depth = log(k), 每一层在merge的时候大概有 k 个 array，然后array 的size 是n
所以 T 大约是 O(kn*log(k))
'''

class Solution:
    def mergeKArrays(self, arrays):
        # Helper function to merge two sorted arrays
        def merge2Arrays(arr1, arr2):
            merged = []
            i, j = 0, 0
            while i < len(arr1) and j < len(arr2):
                if arr1[i] <= arr2[j]:
                    merged.append(arr1[i])
                    i += 1
                else:
                    merged.append(arr2[j])
                    j += 1
            # Append remaining elements
            merged.extend(arr1[i:])
            merged.extend(arr2[j:])
            return merged

        # Recursive function to divide and conquer
        def mergeSort(arrays, start, end):
            if start == end:
                return arrays[start]
            mid = (start + end) // 2
            left = mergeSort(arrays, start, mid)
            right = mergeSort(arrays, mid + 1, end)
            return merge2Arrays(left, right)

        if not arrays:
            return []
        return mergeSort(arrays, 0, len(arrays) - 1)

#------------Test case --------------------


#test=[[1, 4, 5], [1, 3, 4], [2, 6]], expected=[1, 1, 2, 3, 4, 4, 5, 6]
#test=[[], [1]], expected=[1]
#test=[[-1, 1], [-3, 1, 4], [-2, -1, 0, 2]], expected=[-3, -2, -1, -1, 0, 1, 1, 1, 2, 4]

test = [[1, 4, 5], [1, 3, 4], [2, 6]]
expected=[1, 1, 2, 3, 4, 4, 5, 6]
solution = Solution()

assert solution.mergeKArrays(test) == expected