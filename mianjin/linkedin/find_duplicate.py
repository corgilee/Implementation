'''
第一题是给定一个排序好的一个长度为n的数组，数组从1开始增，每次增加1，现在假设这个数组里有个数是重复了，也就是说a[n-1] = n-1，请找出这个重复数，算是一个简单的二分法吧，logn的做法可以找到
'''


def find_duplicate(arr):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        # If we find the point where arr[mid] is not in the expected order, it indicates a duplicate.
        if arr[mid] == arr[mid - 1] or arr[mid] == arr[mid + 1]:
            return arr[mid]
        
        # If arr[mid] > mid + 1, the duplicate is in the left half.
        if arr[mid] > mid + 1:
            right = mid - 1
        # Otherwise, it's in the right half.
        else:
            left = mid + 1
    
    return -1  # In case no duplicate is found

# Example usage
arr = [1, 2, 3, 4, 5, 6, 6, 7, 8, 9, 10]  # The duplicate is 6
duplicate = find_duplicate(arr)
print(f"The duplicate number is: {duplicate}")