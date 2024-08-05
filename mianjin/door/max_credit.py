'''
给定一些餐厅优惠券credit，每张优惠券有固定的value且只能在一天内的限定时间范围内使用，
e.g. [["10:00-11:00", "5"], ["13:00-15:00", "10"]]. 让计算用户最多一次可以用掉多少credit. 需要自己写几个test case，
但很常规，也没怎么看edge cases。 follow-up：(1)同时返回能用出最多credit的时间范围（2）问时间/空间复杂度。结束后时间还有多，于是面试官随口问了1-2个BQ（感觉只是为了充时间的），最后是Q&A

input=[[9,13,5],[10,14,10],[15,16,15]]
output=15
'''
input=[[9,13,5],[10,14,10],[15,16,15]]

def find_max_credit(input):
    #find the min_start time, and max end_time
    min_start=input[0][0]
    max_end=input[0][1]

    for start,end,_ in input:
        min_start=min(min_start,start)
        max_end=max(end,max_end)

    res=[0]*(max_end-min_start+2)
    for start,end,val in input:
        start_index=start-min_start
        end_index=end-min_start
        res[start_index]+=val
        res[end_index+1]-=val
    
    presum=0
    for i,val in enumerate(res):
        presum+=val
        res[i]=presum

    #print(res[:-1])
    return max(res[:-1])
        

def find_max_credit(input):
    # Find the min_start time, and max end_time
    min_start = input[0][0]
    max_end = input[0][1]

    for start, end, _ in input:
        min_start = min(min_start, start)
        max_end = max(end, max_end)

    res = [0] * (max_end - min_start + 2)
    for start, end, val in input:
        start_index = start - min_start
        end_index = end - min_start
        res[start_index] += val
        res[end_index + 1] -= val

    presum = 0
    max_credit = float('-inf')
    max_ranges = []
    current_start = None

    for i, val in enumerate(res):
        presum += val
        res[i] = presum

        if presum > max_credit:
            max_credit = presum
            max_ranges = []
            current_start = i
        elif presum == max_credit:
            if current_start is None:
                current_start = i
        elif presum < max_credit and current_start is not None:
            max_ranges.append((current_start + min_start, i + min_start - 1))
            current_start = None

    if current_start is not None:
        max_ranges.append((current_start + min_start, len(res) + min_start - 2))

    return max_ranges

input = [[9, 13, 5], [10, 14, 10], [15, 16, 15]]

print(find_max_credit(input))

