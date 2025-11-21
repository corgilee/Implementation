import random

def reservoir_sample(stream):
    chosen = None
    for i, x in enumerate(stream, 1):
        # 以 1/i 的概率选择当前元素
        if random.random() < 1 / i:
            chosen = x
    return chosen


'''

Given an infinitely long sequence, we want to select n random elements.
How can we ensure that if the sequence stops at any arbitrary position,
the n selected elements are completely uniformly random?

For the first k elements, we keep all of them.
For the i-th element (i > k), we keep the i-th element with probability k / i,
and with probability 1 / k we replace a randomly chosen one from the existing k selected elements.

When k = 1:

The first element is kept, with probability p(n1) = 1.

No.2:
Keep it with probability 1/2, so p(n1) = 1 * 1/2 = 1/2
p(n2) = 1/2

No.3:
p(n1) = 1/2 * (1 - 1/3) = 1/3
p(n2) = p(n1)
p(n3) = 1/3

No.4:
p(n1) = 1/3 * (1 - 1/4) = 1/4
...

No.i:
We have p(n1) = p(n2) = ... = p(n_{i-1})
= (1 / (i - 1)) * (1 - 1/i) = 1/i
p(n_i) = 1/i

'''