'''
Input:
A method getRandom01 Biased() that generates a random integer in [0, 1], where 0 is generated with probability p and 1 is generated
with probability (1-p)

Output:
A method getRandom06 Uniform() that generates a random integer in [0, 6] with uniform probability
'''

import random


import random

def getRandom01Biased():
    # This function simulates a biased random generator.
    p = 0.3  # Example probability for generating 0
    return 0 if random.random() < p else 1

def getUnbiasedBit():
    while True:
        bit1 = getRandom01Biased()
        bit2 = getRandom01Biased()
        if bit1 != bit2:
            # 如果bit1 ！=bit2 ,那么概率是p*(1-p) 或者 (1-p)*p, 所以在这个情况下 bit1 是随机的
            return bit1

def getRandom06Uniform():
    while True:
        # Generate a 3-bit number (0 to 7)
        num = (getUnbiasedBit() << 2) | (getUnbiasedBit() << 1) | getUnbiasedBit()
        if num <= 6:
            return num

# Example usage
results = [0] * 7
for _ in range(10000):
    num = getRandom06Uniform()
    results[num] += 1

print("Generated numbers distribution over 10000 trials:")
for i in range(7):
    print(f"{i}: {results[i]}")

