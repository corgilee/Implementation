'''
https://leetcode.com/discuss/interview-question/1920662/Google-or-Phone-or-Calculate-Total-Wait-Time

Question:
Calculate the total wait time for a customer C to speak to an agent given N agents, M customers, and T[] time for an agent to serve a customer. T[i] represents the amount of time it takes for an agent i to serve one customer. One agent can serve one customer at a time. All N agents can serve concurrently. The customer chooses the agent with the lowest wait time.

Examples:
N = 2
M = 2
T = [4, 5]
First customer chooses agent 1. Second customer chooses agent 2.
Customer C will wait 4 minutes.

N = 2
M = 4
T = [4, 5]
First customer chooses agent 1. Second customer chooses agent 2.
Third customer chooses agent 1. Forth customer chooses agent 2.
Customer C will wait 8 minutes.

Initial questions:

Bounds on N and M - No bounds
Can N or M be zero - Both can be zero
Are the T values constant - Yes
Are the T values integers - Yes

'''
import heapq
def wait_time(N, M, T):
    """N agents with serving times T, M customers (在当前客户前面的人)
    # T is a list of serving time
    """
    # edge case, #agents > # customers
    if N>M:
        return 0 

    # use heap to save the queue with shortest time
    q_list=[]
    for i in range(N):
        q_list.append((T[i],i))

    heapq.heapify(q_list) #logN
    # M 是客户，N是agent
    for i in range(M-N):
        # Mlog(N)
        time, index=heapq.heappop(q_list)
        time+=T[index]
        heapq.heappush(q_list,(time,index))
    return q_list[0][0]

N = 2
M = 4
T = [5, 4]  
print(wait_time(N, M, T))

'''
[follow up]
[Q1] what if the client number is far greater than the agent number?
[可能的答案]Suppose the number of agents is 2 with times 4 and 5. Now let, the total number of clients is 100000.
Take LCM (least common multiple) of 4 and 5 = 20. So in 20 units of time agent1 will deal with 20/4 =5 clients and in 20 units of time, agent2 will deal with 20/5 = 4 clients. So 9(5+4) clients will get covered in 20 units of time.
Now 100000 = 99999 + 1 => 100000 = 111119 + 1
So 99999 clients can be dealt with in 1111120 units of time. Now we are left with only 1 client which will be picked by agent1. And then agent2 will deal with the customer c.


[Q2]what if there's a constraint on time[i] that 1 <= time[i] <= 10, how would you use this to your advantage?
'''
import math

def lcm(a, b):
    # Calculate the LCM using the formula
    return abs(a * b) // math.gcd(a, b)

# Test the function
num1 = 12
num2 = 18

result = lcm(num1, num2)
print(f"The LCM of {num1} and {num2} is {result}")