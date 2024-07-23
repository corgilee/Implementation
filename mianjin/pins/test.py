'''
# 4 种 feature， 每个feature，3种choice

1. Define card representation (Tuple, etc) and a function to generate a random card
2. Generate n random cards
3. Generate n unique random cards
4. Generate a bunch of random lists of 3 cards, check if they are sets
5. Generate 12 unique random cards, and find all valid sets, and make more efficient

'''
import numpy as np
def gen_rd_card():
    new=np.random.choice([1,2,3],4)
    return tuple(new)

print(gen_rd_card())

def gen_n_cards(n):
    res=[]
    for _ in range(n):
        new=gen_rd_card()
        #print(new)
        res.append(new)
    return res



def gen_unique_n_cards(n):
    res=[]
    for _ in range(n):
        new=gen_rd_card()
        print(new)
        if new in res:
            continue
        res.append(new)
        print(res)
    return res

res=gen_unique_n_cards(3)
print(res)





