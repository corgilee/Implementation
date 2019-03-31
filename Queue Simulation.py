# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 12:52:08 2019


病人每天从早上九点开始以均值为10每小时的泊松分布访问诊所，诊所接受任何在早上九点到下午四点之间的预约，医生需要把所有预约了的病人看完才能关门，医生给每个病人看病的时间服从5~20分钟的均匀分布，写代码对任意一天的情况进行仿真从而回答以下问题
#1 医生在这一天当中一共会见多少病人？
#2 在这一天当中会等待的病人总数？
#3 平均等待时间
#4 诊所关门时间



"""
from numpy import random as rm
import sys
import pandas as pd
import numpy as np

#from 9am to 4pm, 7 hours, 420 mins
'''
t
Na: Total number of coming customers until t, Nd: Total number of leaving customers until t,
n: the number of customers in the system at Time t

'''
#https://www.r-bloggers.com/lang/uncategorized/1471
def simulation(arrival,T=420):
    

    t,Na,Nd,n=0,0,0,0
    # A表示客来时点，D表示客走时点，N表示客走时系统中还有几人，S表示此人服务时长   
    A,D,N,S=[],[],[],[]

    tA=rm.poisson(arrival) # the arrival time of customer after t
    tD=sys.maxsize  # leaving time of the customer who is under service
    
    #t: start from 0
    #before 4:00pm
    while True:
        # t is increasing every loop, t=min(tA,tD), so t increase with either tA or tD
        
        # the next customing coming before the current leaving      
        if (tA<=tD) and (tA <= T):
            t = tA # 更新当前时间
            Na += 1 # Total number of coming customers until t
            n +=1 # number of people in the system
            
            tA=t+rm.poisson(arrival) #the arrival timing of next customer
            
            if n==1: # only consider the first customer in the queue
                tS = rm.uniform(5,25)
                tD = t + tS # the leaving time of the current customer
                S.append(tS) # 保存每人服务时间长度
            A.append(t) # 保存客来时间序列      
        
        elif (tD <= tA) and (tD <=T): #
            t = tD 
            n= n-1
            Nd = Nd + 1
            if n==0:  #如果无人等待
                tD = sys.maxsize 
            else:  # if someone is in the queue
                tS = rm.uniform(5,25)
                tD = t + tS #update the leaving time of first people in the queue
                S.append(tS) # 保存每人服务时间长度
            D.append(t) #保存客走时间序列
            #N.append(n) #保存系统中人数序列
        
        elif (tA>T) and (tD>T):  
            break # 过点关门

#after 4:00pm 
    while True:
        if n <=0:  
            break
        t = tD
        n=n-1
        Nd=Nd+1 
        D.append(t) 
        N.append(n) 
        if n>0:
            tS = rm.uniform(5,25)
            tD = t +  tS
            S.append(tS) 
    Tp = max(t-T,0) #关门时间
    # A表示客来时点，D表示客走时点，S表示此人服务时长
    raw = {'A':A,'D':D,'S':S}
    data = pd.DataFrame(raw)
    data["wait_time"]=data["D"].shift(1)-data["A"]    
    data["if_wait"]=np.where(data["D"].shift(1)-data["A"]>0,1,0)

    
    return {'Num_of_cust': Na,"Delay": round(Tp,1),
            "Num_of_cust_wait": data.if_wait.sum(),"avgwait":round(np.mean(data["wait_time"]*data["if_wait"]),1)}


#if __name__ == '__main__':
     #print(simulation(arrival=15,T=420))
     
#simulate 100 times
     
res=[simulation(15,T=420) for i in range(100)]
res=pd.DataFrame(res)
res.head()

# 画出平均等待时间的直方图
import matplotlib.pyplot  as plt
plt.hist(res.avgwait)
