'''
https://leetcode.com/discuss/interview-question/5158808/Pinterest-Phone-Screen-MLE

You can ignore order of operations
Numbers must be used left to right and cannot reorder numbers in the list

这道题是简化版的，不考虑乘法的优先顺序

list = [2,3,5], target = 25 -> true, because (2 + 3) * 5 = 25
list = [2,3,5], target = 12 -> false, because no way to get 12
'''
# 复杂度是 2^n


def can_hit_target(nums,target):
    def helper(index, current_value):
        if index==len(nums):
            return current_value==target

        #option1:add
        add=helper(index+1,current_value+nums[index])
        #option2:multi
        multi=helper(index+1,current_value*nums[index])

        return add or multi

    return helper(1,nums[0]) # 这里index 从1 开始，current_val=nums[0]

# Test cases
print(can_hit_target([2, 3, 5], 25))  # Output: True
print(can_hit_target([2, 3, 5], 12))  # Output: False





# def can_hit_target(nums, target):
#     def helper(index, current_value):
#         if index == len(nums):
#             return current_value == target
        
#         # Try adding the next number
#         if helper(index + 1, current_value + nums[index]):
#             return True
        
#         # Try multiplying by the next number
#         if helper(index + 1, current_value * nums[index]):
#             return True
        
#         return False

#     # Start the recursion from the first index and the first number
#     return helper(1, nums[0])




