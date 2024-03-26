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
    # Decide whether to explore or exploit
    if np.random.rand() < epsilon:
        # Exploration: choose a random version
        chosen_version = np.random.choice(list(sends.keys()))
    else:
        # Exploitation: choose the version with the highest estimated CTR
        estimated_ctrs = {version: (clicks[version] / sends[version] if sends[version] > 0 else 0) 
                          for version in sends}
        chosen_version = max(estimated_ctrs, key=estimated_ctrs.get)
    
    # Simulate sending the email and whether it was clicked
    sends[chosen_version] += 1
    if np.random.rand() < true_ctrs[chosen_version]:
        clicks[chosen_version] += 1

# Calculate final estimated CTRs
estimated_ctrs_final = {version: (clicks[version] / sends[version] if sends[version] > 0 else 0) 
                        for version in sends}

#print(estimated_ctrs_final, sends, clicks)
print(estimated_ctrs_final) #最后模拟出来的结构，estimated ctr应该是接近 true ctr的

