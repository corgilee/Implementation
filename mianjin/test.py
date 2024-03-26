import numpy.random as nd
import numpy as np
from collections import Counter

import numpy as np

# True CTRs for each email version (unknown to the algorithm)
true_ctrs = {'A': 0.1, 'B': 0.15, 'C': 0.2, 'D': 0.25}

# Initialize counters for sends and clicks
sends = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
clicks = {'A': 0, 'B': 0, 'C': 0, 'D': 0}

# Epsilon-Greedy parameters
epsilon = 0.1
n_iterations = 10000  # Total number of emails to send


for _ in range(n_iterations):
    # explore or exploit

    if nd.random()<epsilon: #explore
        n=len(sends)
        version=nd.choice(list(sends.keys()))

    else:
        #exploite,selected the version with highest ctr
        version_ctr={ key:clicks[key]*1.0/sends[key] if sends[key]>0 else 0 for key in sends.keys() }

        # find the version with highest ctr
        version, ctr=max(version_ctr.items(),key=lambda x: x[1])

        #print(version,ctr)

    #simulate the results, with true ctr
    sends[version]+=1
    if nd.random()<true_ctrs[version]:
        clicks[version]+=1

print(version_ctr) 




