'''
Remove prefix strings in a list of string

Example 1: input  {"a", "ab", "abc"} -> output: {"abc"}.
Example 2: input {"a", "bc", "ab", "abc hello"} --> output: {"bc", "abc hello"}.
要求： output array 要保持String 在input 里的顺序。比如第二个例子: {"abc hello", "bc"} 顺序就不对了。。。

Step:
Loop through all strings.
For each string, check if it is a prefix of any other string in the list.
If it is not a prefix of any other string, keep it in the result.
Maintain original order by iterating over the input list directly.

Time: O(n² * m), where: n = number of strings, m = average length of the strings.
Space: O(n) for the output list.


follow up: 删除其他string的substring
Step:
遍历每个字符串 s。
检查它是否是其他字符串的子串（t != s 且 s in t）。
如果不是任何其他字符串的子串，则保留。
返回保留的结果，顺序和原始输入一致。

Time: O(n² * k) — n 是字符串数量，k 是平均字符串长度
Space: O(n) — 存储结果用的空间

'''

def remove_prefix_strings(strings):
    result = []

    # Iterate through the list in order
    for i in range(len(strings)):
        is_prefix = False
        
        # Compare with all other strings
        for j in range(len(strings)):
            if i == j:
                continue  # Skip comparing with itself

            # Check if current string is a prefix of another
            if strings[j].startswith(strings[i]):
                is_prefix = True
                break  # No need to check further

        # If it's not a prefix of any other string, add to result
        if not is_prefix:
            result.append(strings[i])

    return result


#### follow up #####

def remove_substring_strings(strings):
    result = []

    for i, s in enumerate(strings):
        is_substring = False

        # Compare with all other strings
        for j, t in enumerate(strings):
            if i != j and s in t:
                is_substring = True
                break

        if not is_substring:
            result.append(s)

    return result


