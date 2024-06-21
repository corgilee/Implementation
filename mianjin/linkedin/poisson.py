'''
要计算在3分钟内只来一辆车的概率，我们可以使用泊松分布。
泊松分布用于描述单位时间内事件发生的次数，其中事件是独立发生的，并且事件发生的平均速率是已知的。

It gives the probability of an event happening a certain number of times (k) within a given interval of time or space


'''


import math

def poisson_probability(lmbda, k):
    """
    Calculate the Poisson probability of k events occurring
    given the average rate (lmbda) of events.
    
    :param lmbda: Average rate (λ)
    :param k: Number of events (k)
    :return: Poisson probability P(X = k)
    """
    return (lmbda ** k) * math.exp(-lmbda) / math.factorial(k)

# Given data
average_rate_per_minute = 8 / 6  # 8 cars in 6 minutes
time_interval = 3  # minutes
k = 1  # number of cars we want to calculate the probability for

# Calculate the average rate for the 3-minute interval
lambda_3_minutes = average_rate_per_minute * time_interval

# Calculate the probability
probability = poisson_probability(lambda_3_minutes, k)

print(f"The probability of exactly {k} car(s) arriving in {time_interval} minutes is approximately {probability:.4f}")
