'''
Strategy: Vertical Scanning

Start comparing characters from left to right, one column at a time across all strings.
For each character index i:
Compare strs[0][i] with the i-th character of all other strings.
If any mismatch occurs or any string is shorter than i + 1, return the prefix found so far.
If we finish scanning the first string entirely with no mismatches, it is the LCP.


Time: O(S), where S is the total number of characters in all strings
(since we compare up to the shortest string's length)
Space: O(1) extra (excluding the output)
'''

class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        # Return empty string if input list is empty
        if not strs:
            return ""

        # Iterate through characters of the first string
        for i in range(len(strs[0])):
            char = strs[0][i]  # Character to compare

            # Check this character against all other strings
            for s in strs[1:]:
                # If we reach the end of any string OR characters don't match
                if i >= len(s) or s[i] != char:
                    # Return the prefix found so far (up to index i)
                    return strs[0][:i]

        # If loop completes, entire first string is a common prefix
        return strs[0]
