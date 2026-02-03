'''
given a list of player's tennis rank, print winners for each round. higher rank always wins e.g.: given [1, 2, 3, 4, 5, 6, 7, 8] print: 1, 2, 3, 4, 5, 6, 7, 8 -> 1, 3, 5, 7 -> 1,5 ->1

'''

def print_round_winners(ranks):
    cur = ranks[:]   # copy to avoid modifying input

    while len(cur) > 1:
        print(", ".join(map(str, cur)), end=" -> ")

        nxt = []
        i = 0
        while i < len(cur) - 1:
            winner = min(cur[i], cur[i + 1])  # higher rank wins
            nxt.append(winner)
            i += 2

        #【optional】odd player gets a bye
        if i == len(cur) - 1:
            nxt.append(cur[i])

        cur = nxt

    print(cur[0])  # champion

print_round_winners([1,2,3,4,5,6,7,8])