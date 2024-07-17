# question 1, count and say
def countAndSay(self, n: int) -> str:
    '''
    双指针解法，同向
    time complexity, 每一个循环i的complexity 是2^i (lenght of s)
    O(2^0+2^1+2^2+...2^(n-1)) =O(2^n)
    '''
    def say(s):
        res=""
        n=len(s)
        i,j=0,0 # j is the leader, i is the follower
        while j<n:
            if s[j]!=s[i]:
                res+=str(j-i)
                res+=s[i]
                i=j
            j+=1
        # add the last
        res+=str(j-i)
        res+=s[i]
        return res
        #main
        s="1"
        for _ in range(n-1):
            s=say(s)
        
        return s


# question 2, reverse count and say
'''
In this problem, the input provided to us is "1211" and our goal is to return "21"
'''

'''
复杂度: 
* 有 memo, O(n^2),because for each of the n positions, it processes substrings in 
O(n) time, leveraging memoization to avoid redundant computations
* 没有memo，O(2^n) due to the exponential number of possible substrings

当时跟面试官确认了位数可以是多个digit但是数字只能是1个digit（例如753可以表示75个3但不能表示7个53），然后考虑了digit是0的情况和最后只剩一位digit的情况，其他还有什么edge case我暂时想不出来了...

'''

from collections import defaultdict

def reverseCountSay(s):
    memo = defaultdict(list)
    # memo has the list comb of each (tmp)string

    def recursive(s):
        if len(s) == 0:
            return [[]]
        elif s in memo:
            return memo[s]
        for i in range(1, len(s)):
            count = int(s[:i])
            curNum = s[i] #数字只能是一位数的
            # skip if curNum is '0'
            if curNum =='0':
                continue
            for subRes in recursive(s[i + 1:]):
                # subRes 可以是[],也可以是['123']
                c=[count * curNum]
                memo[s].append( c+ subRes)
        return memo[s]

    #main
    recursive(s) #运行了recursive之后memo里面就有东西了
    res=[]
    for comb in memo[s]:
        res.append(''.join(comb))
    return res



s='03'
s='110'
s='753'
s="1211" # 数字只能是一位数的， 【1个2， 1个1】， 【121 个 1】
print(reverseCountSay(s))

