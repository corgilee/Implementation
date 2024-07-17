
'''
In this problem, the input provided to us is "1211" and our goal is to return "21"
'''
import collections

memo=collections.defaultdict(list)# memo has the list comb of each (tmp)string

def reverse_say(s):
    if len(s)==0:
        return [[]]
    elif s in memo:
        return memo[s]

    n=len(s)
    for i in range(1,n):
        # 必须从1 开始算起
        count=int(s[:i])
        num=s[i]
        for subres in reverse_say(s[i+1:]):
            c=[count*num]
            memo[s].append(c+subres)
    return memo[s]

# main
s="1211"
reverse_say(s)

res=[]
for comb in memo[s]:
    res.append("".join(comb))

print(res)
#print([4*"3"])




    

# 
