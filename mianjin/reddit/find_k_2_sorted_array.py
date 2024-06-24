'''
https://leetcode.com/discuss/interview-question/351782/Google-or-Phone-Screen-or-Kth-Largest-Element-of-Two-Sorted&#8205;&#8204;&#8204;&#8204;&#8205;&#8205;&#8204;&#8205;&#8204;&#8204;&#8204;&#8205;&#8205;&#8205;&#8204;&#8204;&#8204;&#8204;-Arrays

log(k)
'''


def findK(nums1, nums2, k):
        if len(nums1) == 0:
            return nums2[k - 1]
        if len(nums2) == 0:
            return nums1[k - 1]
        
        if k == 1:
            return min(nums1[0], nums2[0])
        
        a = nums1[k // 2 - 1] if k // 2 <= len(nums1) else None
        b = nums2[k // 2 - 1] if k // 2 <= len(nums2) else None

        if b is None or (a is not None and a < b):
            # b 的数量不够，或者 a 的前面k//2 太小了，不到第k个要的大小，所以a 直接从 k//2 算起
            return findK(nums1[k // 2:], nums2, k - k // 2)
        return findK(nums1, nums2[k // 2:], k - k // 2)


def findMedianSortedArrays(nums1, nums2):
    n1 = len(nums1)
    n2 = len(nums2)
    if (n1 + n2) % 2 == 1:
        return findK(nums1, nums2, (n1 + n2) // 2 + 1)
    else:
        return (findK(nums1, nums2, (n1 + n2) // 2) + findK(nums1, nums2, (n1 + n2) // 2 + 1)) // 2.0


nums1=[3,4,5]
nums2=[4,6,7]
k=4

print(findK(nums1,nums2,k))