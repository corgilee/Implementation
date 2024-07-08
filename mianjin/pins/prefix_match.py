'''
find index of a given prefix in a sorted ‍‌‌‌‍‍‌‍‌‌‌‍‍‍‌‌‌‌array. ['ab', 'app', 'apple'], prefix = 'ap', return 1
我猜是返回第一个含有指定prefix 的
这里可以用 str.startswith("xx")
'''

# 冷知识，string 是可以用来比大小的
#print("ac"<"ab")

def find_prefix_index(arr,prefix):
    l,r=0,len(arr)-1
    res=-1
    while l<=r:
        mid=(l+r)//2

        if arr[mid].startswith(prefix):
            res=mid
            r=mid-1
        elif arr[mid]<prefix:
            l=mid+1
        else:
            r=mid-1

    return res


# Test the function
arr = ['ab', 'app', 'apple']
prefix = 'ap'
result = find_prefix_index(arr, prefix)
print(f"Index of prefix '{prefix}': {result}")

