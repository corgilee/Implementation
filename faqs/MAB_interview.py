import math
import random
import numpy as np

'''
#Multi‑armed bandit

3种常见方法

Epsilon-Greedy:
- Track how many times each arm is pulled and its estimated mean reward.
- For each step, explore with probability epsilon; otherwise pick the best mean.
- Pull the arm, observe reward, and update the mean with an incremental average.

UCB1 (Upper Confidence Bound 1):
- Ensure every arm is tried once to avoid divide-by-zero.
- For each step, compute a score = mean + exploration bonus.
- Pick the arm with the highest score, pull it, and update its mean.

Thompson Sampling (Bernoulli rewards):
- Maintain Beta(alpha, beta) for each arm as the posterior over CTR.
- For each step, sample a theta from each Beta distribution.
- Pick the arm with the largest sample, pull it, and update alpha/beta.
'''



class Bandit:
    def __init__(self, n_arms):
        self.n = n_arms
        self.counts = [0] * n_arms
        self.values = [0.0] * n_arms

    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n

def run(bandit, true_probs, steps=10000):
    #random.seed(seed)
    for _ in range(steps):
        arm = bandit.select()
        
        if random.random() < true_probs[arm]:
            reward = 1 
        else:
            reward=0
            
        bandit.update(arm, reward)
    return bandit


class EpsilonGreedy(Bandit):
    def __init__(self, n_arms, epsilon=0.1):
        super().__init__(n_arms)
        self.epsilon = epsilon

    def select(self):
        if random.random() < self.epsilon:
            return random.randrange(self.n) #(0,n-1)
        
        best_arm = np.argmax(self.values)

        return best_arm


class UCB1(Bandit):
    def select(self):
        for i in range(self.n):
            if self.counts[i] == 0:
                return i
        total = sum(self.counts)

        best_arm = 0
        best_score = float("-inf")

        for i in range(self.n):
            bonus = math.sqrt(2 * math.log(total) / self.counts[i])
            score = self.values[i] + bonus
            if score > best_score:
                best_score = score
                best_arm = i

        return best_arm



class ThompsonBernoulli:
    def __init__(self, n_arms):
        self.n = n_arms
        self.alpha = [1] * n_arms
        self.beta = [1] * n_arms

    def select(self):
        best_arm = 0
        best_sample = -1.0

        for i in range(self.n):
            sample = random.betavariate(self.alpha[i], self.beta[i])
            if sample > best_sample:
                best_sample = sample
                best_arm = i

        return best_arm


    def update(self, arm, reward):
        if reward == 1:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1





def main():
    true_probs = [0.1, 0.15, 0.2, 0.25]


    bandit=EpsilonGreedy(len(true_probs), epsilon=0.1)
    eg = run(bandit, true_probs)
    print("EpsilonGreedy values:", [round(v, 4) for v in eg.values])
    print("EpsilonGreedy counts:", eg.counts)
    print("----------------------------------------------------")


    bandit= UCB1(len(true_probs))
    ucb = run(bandit, true_probs)
    print("UCB1 values:", [round(v, 4) for v in ucb.values])
    print("UCB1 counts:", ucb.counts)
    print("----------------------------------------------------")

    ts = run(ThompsonBernoulli(len(true_probs)), true_probs)
    ts_means = [ts.alpha[i] / (ts.alpha[i] + ts.beta[i]) for i in range(ts.n)]
    print("Thompson means:", [round(v, 4) for v in ts_means])
    print("Thompson counts:", [a + b - 2 for a, b in zip(ts.alpha, ts.beta)])


if __name__ == "__main__":
    main()
