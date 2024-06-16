import random
import collections

'''
如何用random7 生成 random10 ？
Generate a number in the range 1 to 49:

(rand7() - 1) * 7 generates one of the values {0, 7, 14, 21, 28, 35, 42}.
Adding rand7() shifts these to {1, 2, ..., 49} with equal probability.

不能用rand7()*7 去余数, 因为
Numbers from 1 to 40: All 10 possible remainders (0 to 9) appear exactly 4 times.
Numbers from 41 to 49: The remainders 0 to 8 appear one extra time, but 9 appears only 4 times, not 5.

要用40作为 一个cut，>40 pass, =< 40 看reminder 


同理，如何用rand3 产生 rand7
num1: rand3() --> 【1,2,3】
num2: (rand3()-1)*3 {0, 3, 6} 
num1+num2: range [1,9], 然后也是 >7 pass, <=7 看余数
'''

def rand7():
    return random.randint(1, 7) #(inclusive, inclusive)

def rand10():
    while True:
        num=(rand7()-1)*7+ rand7() #1-49
        # use 40 as threshold
        if num>40:
            #print("pass")
            pass
        else:
            return num%10
            

i=0
list1=[]
while i<1000:
    num=rand10()
    list1.append(num)
    i+=1

#count=sorted(count.items(), key=lambda x:x[0])

# generated_numbers
generated_numbers = [random.randint(0, 9) for _ in range(1000)]

'''
use Kolmogorov-Smirnov (KS) Test to determin if 2 list are from the same distribution
'''

from scipy.stats import ks_2samp

# Perform KS test
ks_stat, p_value = ks_2samp(list1, generated_numbers)

print(f"KS statistic: {ks_stat}, p-value: {p_value}")

if p_value > 0.05:
    print("The two lists have the same distribution (fail to reject null hypothesis).")
else:
    print("The two lists do not have the same distribution (reject null hypothesis).")





