# question 1, count and say
def countAndSay(self, n: int) -> str:
    '''
    双指针解法，同向
    time complexity, 每一个循环i的complexity 是2^i
    O(2^0+2^1+2^2+...2^(n-1)) =O(2^n)
    '''
    def get_cns(word):
        res = ""
        i = 0
        while i < len(word):
            j = i + 1
            cnt = 1
            # 一直增加j 到i!=j
            while j < len(word) and word[j] == word[i]:
                cnt += 1
                j += 1
                
            res += str(cnt) + word[i]
            i = j
        return res

    sol = "1"
    for i in range(n-1):
        sol = get_cns(sol)       
    return sol


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
    # memo 就是每一种排列组合都有一个list

    def recursive(s):
        if len(s) == 0:
            return [[]]
        elif s in memo:
            return memo[s]
        for i in range(1, len(s)):
            count = int(s[:i])
            curNum = s[i]
            for subRes in recursive(s[i + 1:]):
                # subRes 可以是[],也可以是['123']
                memo[s].append([curNum * count] + subRes)
        return memo[s]
    recursive(s) #运行了recursive之后memo里面就有东西了
    #print(memo[s])
    return [''.join(output) for output in memo[s]]

s="1211"
print(reverseCountSay(s))