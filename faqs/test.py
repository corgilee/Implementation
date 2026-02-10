from collections import Counter

class Solution:
    def divisible_by_8_after_rearrange(self, X: str) -> bool:
        """
        Return True if we can rearrange digits of X to form a number divisible by 8.
        Assumption: final number is a normal integer representation (no leading zeros),
        except the number "0" itself is allowed.
        """
        n = len(X)
        cnt = Counter(X)  # counts of characters '0'..'9'

        # Helper: check if we have enough digits to build pattern string s (like "104")
        def can_build(s: str) -> bool:
            need = Counter(s)
            for ch, k in need.items():
                if cnt[ch] < k:
                    return False
            return True

        # ---- Case 1: length 1 ----
        if n == 1:
            return (int(X) % 8) == 0

        # ---- Case 2: length 2 ----
        # We must form a 2-digit multiple of 8, and the first digit can't be '0'
        if n == 2:
            for v in range(0, 100, 8):
                s = f"{v:02d}"  # 2-digit with leading zeros
                if s[0] == '0':  # disallow leading zero for the whole number
                    continue
                if can_build(s):
                    return True
            return False

        # ---- Case 3: length >= 3 ----
        # We only need to find ANY 3 digits that form a multiple of 8 as the ending.
        # The last 3 digits can have leading zeros (e.g., "...016"), that's fine.
        # But if n == 3, the first digit of the whole number can't be '0'.
        for v in range(0, 1000, 8):
            s = f"{v:03d}"  # 3-digit ending, leading zeros allowed in the ending

            # If the entire number length is exactly 3, disallow overall leading zero
            if n == 3 and s[0] == '0':
                continue

            if not can_build(s):
                continue

            # If n > 3, we also need to ensure we can build a full number with non-leading-zero
            # If all digits are zero, number is 0 which is divisible by 8.
            if n > 3:
                # subtract needed digits for the ending and see what's left
                need = Counter(s)
                remaining = cnt.copy()
                for ch, k in need.items():
                    remaining[ch] -= k

                # If everything left is zero digits, the whole number is some zeros -> "0" effectively
                # But since length > 3, it would be "0000..." which is numerically 0 and divisible by 8.
                # If you strictly disallow leading zeros for a fixed-length string, then you'd require a non-zero
                # remaining digit; in most interview settings, numeric value is what matters.
                if sum(remaining.values()) == remaining['0']:
                    return True

                # Otherwise, as long as there exists a non-zero remaining digit, we can place it first.
                for d in '123456789':
                    if remaining[d] > 0:
                        return True

                # If no non-zero remaining digit exists, it means remaining are only zeros,
                # which we already handled above.
                # So continue.
            else:
                # n == 3 case already handled by leading-digit check
                return True

        return False
