'''
一道题分成两个section 
给一个list of int, 返回组成的最小number in string。[7,1,8] -> "178", [0,4,0] -> "4" 给一个list of int, and a lower bound int, 
返回smallest number that is larger than the lower bound in string. ex. [7,1,8] , lower=200 -> "718" ex2. [7,1,8], lower=179 -> "187", could you provide me the solution?
'''

#problem_1
'''
Goal: Arrange the digits to form the smallest number.
Leading zeros are allowed during construction, but the final representation removes leading zeros (since "004" and "4" represent the same number).
'''

def _normalize(num_str: str) -> str:
    # remove leading zeros; keep "0" if all zeros
    s = num_str.lstrip("0")
    return s if s else "0"


def min_number_string(digits) -> str:
    s = "".join(str(d) for d in sorted(digits))
    return _normalize(s)

# problem 2
#Prefix comparison + backtracking + greedy once larger

# "greater" indicates whether the number prefix we have built so far
# is already strictly greater than the corresponding prefix of the lower bound.
# if greater == False, the prefix so far can only be equal, never smaller.



from typing import List, Optional
from collections import Counter

def _normalize(num_str: str) -> str:
    s = num_str.lstrip("0")
    return s if s else "0"

def smallest_number_strictly_greater(digits: List[int], lower: int) -> Optional[str]:
    """
    Use all digits exactly once (multiset), allow leading zeros.
    Return the smallest number (as string, normalized) that is > lower.
    If no solution exists, return None.
    """
    n = len(digits)
    if n == 0:
        return None

    lower_pad = str(lower).zfill(n)  # pad to length n
    cnt = Counter(digits)
    uniq = sorted(cnt.keys())
    path: List[str] = []

    def dfs(i: int, greater: bool) -> bool:
        if i == n:
            return greater  # must be strictly greater overall

        bound_digit = int(lower_pad[i])

        for d in uniq:
            if cnt[d] == 0:
                continue

            if not greater and d < bound_digit:
                continue  # would make prefix smaller => cannot recover later

            cnt[d] -= 1
            path.append(str(d))

            next_greater = greater or (d > bound_digit)
            if dfs(i + 1, next_greater):
                return True

            path.pop()
            cnt[d] += 1

        return False

    ok = dfs(0, False)
    if not ok:
        return None

    return _normalize("".join(path))


# test case
print(min_number_string([7,1,8])) # "178"
print(min_number_string([0,4,0])) # "4"


print(smallest_number_strictly_greater([7,1,8], 200)) # "718"
print(smallest_number_strictly_greater([7,1,8], 179)) # "187"