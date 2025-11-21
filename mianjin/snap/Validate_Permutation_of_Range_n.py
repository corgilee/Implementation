'''
You are given an integer array nums of length n + 1.

A valid permutation is defined as an array that:

Contains every integer from 0 to n exactly once,

Contains no duplicates,

Does not include any number outside the range [0, n].

Your task is to determine whether nums is a valid permutation.

Return:

True if the array is a valid permutation

False otherwise
'''

#Part 1 — O(n) time, O(n) space
def is_permutation(nums):
    n = len(nums) - 1
    seen = [False] * (n + 1)

    for x in nums:
        if x < 0 or x > n:
            return False
        if seen[x]:
            return False
        seen[x] = True
    
    return True

# Part 2 Code: Cyclic Sort (In-place, O(1) space)

def is_permutation(nums):
    n = len(nums) - 1

    i = 0
    while i < len(nums):
        x = nums[i]

        # Check range first
        if x < 0 or x > n:
            return False

        # If nums[i] is not at its correct index, try to swap
        if nums[i] != nums[x]:
            # If conflict: nums[i] = x but nums[x] already = x → duplicate
            if nums[x] == x:
                return False
            nums[i], nums[x] = nums[x], nums[i]
        else:
            i += 1

    # Final verification: nums[i] == i for all i
    for i in range(len(nums)):
        if nums[i] != i:
            return False

    return True
