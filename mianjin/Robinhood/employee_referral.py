'''
Robinhood is famous for its referral program. It’s exciting to see our users spreading the word across their friends and family. One thing that is interesting about the program is the network effect it creates. We would like to build a dashboard to track the status of the program. Specifically, we would like to learn about how people refer others through the chain of referral.

For the purpose of this question, we consider that a person refers all other people down the referral chain. For example, A refers B, C, and D in a referral chain of A -> B -> C -> D. Please build a leaderboard for the top 3 users who have the most referred users along with the referral count.

Referral rules:

A user can only be referred once.
Once the user is on the RH platform, he/she cannot be referred by other users. For example: if A refers B, no other user can refer A or B since both of them are on the RH platform.
Referrals in the input will appear in the order they were made.
Leaderboard rules:

The user must have at least 1 referral count to be on the leaderboard.
The leaderboard contains at most 3 users.
The list should be sorted by the referral count in descending order.
If there are users with the same referral count, break the ties by the alphabetical order of the user name.
Input

rh_users –> array.string
new_users –> array.string
rh_users = ["A", "B", "C"]
| | |
v v v
new_users = ["B", "C", "D"]
Output

array.string
["A 3", "B 2", "C 1"]
** note: at the beginning of the interview, the output may be returning [“a”,”b”] as starter code.

Additional Details

[execution time limit] 4 seconds
[memory limit] 1GB
[input] array.string rh_users
A list of referring users
[input] array.string new_users
A list of users that was referred by the users in the referrers array in the same order
[output] array.string
An array of 3 users on the leaderboard. Each of the elements here would have the “[user] [referral count]” format. For example, “A 4”

'''
import collections

def leaderboard(rh_users, new_users):
    parent = {}
    count = collections.defaultdict(int)
    SENTINEL = "__RH__"

    for ref, new in zip(rh_users, new_users):
        # ensure referrer is on platform
        if ref not in parent:
            parent[ref] = SENTINEL

        # now enforce: user can only be referred once / once on platform can't be referred
        if new in parent:
            continue

        parent[new] = ref

        cur = ref
        while cur != SENTINEL:
            count[cur] += 1
            cur = parent[cur]

    filtered = [(u, c) for u, c in count.items() if c > 0]
    filtered.sort(key=lambda x: (-x[1], x[0]))
    ranked = filtered[:3]
    return [f"{u} {c}" for u, c in ranked]



####--------- test----------------
def run_case(i, rh_users, new_users, expected):
    got = leaderboard(rh_users, new_users)  # <-- your function name here
    assert got == expected, (
        f"\n[FAIL] Case {i}\n"
        f"rh_users  = {rh_users}\n"
        f"new_users = {new_users}\n"
        f"expected  = {expected}\n"
        f"got       = {got}\n"
    )
    print(f"[PASS] Case {i}")

# ========================
# Test Case 1 — Easy (given example: simple chain)
# A -> B -> C -> D
# ========================
run_case(1,
    rh_users=["A", "B", "C"],
    new_users=["B", "C", "D"],
    expected=["A 3", "B 2", "C 1"],
)

# ========================
# Test Case 2 — Easy (single referral)
# A -> B
# ========================
run_case(2,
    rh_users=["A"],
    new_users=["B"],
    expected=["A 1"],
)

# ========================
# Test Case 3 — Easy (no valid referrals at all)
# A refers A (invalid because A is already on platform)
# ========================
run_case(3,
    rh_users=["A"],
    new_users=["A"],
    expected=[],
)

# ========================
# Test Case 4 — Medium (two-step chain)
# A -> B -> C
# ========================
run_case(4,
    rh_users=["A", "B"],
    new_users=["B", "C"],
    expected=["A 2", "B 1"],
)

# ========================
# Test Case 5 — Medium (duplicate target referred twice; second ignored)
# A -> B (ok)
# C -> B (ignored; B already on platform)
# A -> D (ok)
# ========================
run_case(5,
    rh_users=["A", "C", "A"],
    new_users=["B", "B", "D"],
    expected=["A 2"],
)

# ========================
# Test Case 6 — Medium (branching + tie-break alphabetical)
# A -> B, A -> C, B -> D, C -> E
# A=4; B=1; C=1 => B before C
# ========================
run_case(6,
    rh_users=["A", "A", "B", "C"],
    new_users=["B", "C", "D", "E"],
    expected=["A 4", "B 1", "C 1"],
)

# ========================
# Test Case 7 — Hard (multiple roots + tie on counts -> alphabetical; top 3 only)
# A,X,B,Y each has 1; pick A,B,X
# ========================
run_case(7,
    rh_users=["A", "X", "B", "Y"],
    new_users=["C", "Z", "D", "W"],
    expected=["A 1", "B 1", "X 1"],
)

# ========================
# Test Case 8 — Hardest (two big trees, ignored referral to existing referrer, tie at top)
# P->Q->R->S => P=3, Q=2, R=1
# T->U->V->W => T=3, U=2, V=1
# P->T ignored (T already on platform)
# Top3: P3, T3, Q2
# ========================
run_case(8,
    rh_users=["P", "Q", "R", "T", "P", "U", "V"],
    new_users=["Q", "R", "S", "U", "T", "V", "W"],
    expected=["P 3", "T 3", "Q 2"],
)

print("\nAll tests passed!")
