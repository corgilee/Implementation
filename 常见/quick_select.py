
# return the kth number

def quick_select(l,r):
    # 最后的目的是确保 p index 左边的值都小于p index 右边的值
    p=l
    base=nums[r]
    for i in range(l,r):
        if nums[i]<=base:
            nums[p],nums[i]=nums[i],nums[p]
            p+=1
        #最后把nums[p] 和nums[r] 换一下
    nums[p],nums[r]=nums[r],nums[p]
    if k==p:
        return nums[p]
    elif k<p:
        return quick_select(0,p-1)
    else:
        return quick_select(p+1,r)

print(quick_select(0,len(nums)-1))