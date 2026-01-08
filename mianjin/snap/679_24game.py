########## original question #######
class Solution(object):
    def judgePoint24(self, cards):
        """
        :type cards: List[int]
        :rtype: bool
        """
        eps=1e-6

        if len(cards)==1:
            #print(cards)
            return abs(cards[0]-24)<=eps

        
        n=len(cards)
        for i in range(n):
            a=cards[i]
            for j in range(n):
                if i==j:
                    continue

                b=cards[j]

                reminder=[]
                for k in range(n):
                    if k!=i and k!=j:
                        reminder.append(cards[k])

                ops=[a+b,a-b,b-a,a*b]
                for val in ops:
                    if self.judgePoint24(reminder+[val])==True:
                        return True
                
                # a/b
                if b!=0 and self.judgePoint24(reminder+[a*1.0/b])==True:
                    return True
                # b/a
                if a!=0 and self.judgePoint24(reminder+[b*1.0/a])==True:
                    return True

        return False
        

### follow up, print out the expression ####
class Solution(object):
    def judgePoint24_with_expr(self, cards):
        eps = 1e-6

        # each item: (value, expression)
        items = [(float(x), str(x)) for x in cards]

        def dfs(arr):
            if len(arr) == 1:
                val, expr = arr[0]
                if abs(val - 24.0) < eps:
                    return expr
                return None

            n = len(arr)
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue

                    a_val, a_expr = arr[i]
                    b_val, b_expr = arr[j]

                    # remaining elements except i and j
                    rest = []
                    for k in range(n):
                        if k != i and k != j:
                            rest.append(arr[k])

                    # candidate results
                    candidates = []

                    # + and * (commutative) → only once
                    if i < j:
                        candidates.append((a_val + b_val, "(" + a_expr + "+" + b_expr + ")"))
                        candidates.append((a_val * b_val, "(" + a_expr + "*" + b_expr + ")"))

                    # - (non-commutative)
                    candidates.append((a_val - b_val, "(" + a_expr + "-" + b_expr + ")"))

                    # / (non-commutative, avoid divide by zero)
                    if abs(b_val) > eps:
                        candidates.append((a_val / b_val, "(" + a_expr + "/" + b_expr + ")"))

                    for val, expr in candidates:
                        res = dfs(rest + [(val, expr)])
                        if res is not None:
                            return res

            return None

        return dfs(items)
