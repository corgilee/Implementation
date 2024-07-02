'''
#https://www.1point3acres.com/bbs/thread-998365-1-1.html
Input:
Store Open and close time , total capacity, List of existing reservations
Output every interval with the capacity available

# test case
Test Case example 1:
Open: 8 AM, Close 9 PM, Capacity: 5
Reservations { {
start time: 9 am
end time: 9:30 am
size : 3
}
}
Output: Map<Interval, Int>
8AM-9AM 5,
9-930AM 2,
930-9PM 5


Test case example 2:
Reservations { {
‍‌‍‍‍‌‍‍‌‌‌‌‌‌‍‌‌‍‌‍start time: 9 am
end time: 9:30 am
size: 3
}
{
start time: 9:15 am
end time : 9:45 am
size: 2
}
}
Output: Map<Interval, Int>
8:00AM-9:00A‍‌‌‌‍‍‌‍‌‌‌‍‍‍‌‌‌‌M 5,
9:00AM-9:15AM 2,
9:15AM-9:30AM 0,
9:30AM-9:45AM 3
9:45AM-9:00PM 5
'''

def time_to_number(s):
    new_s=s.split(":")

    sign=new_s[1][-2:]
    if sign=="AM":
        h=int(new_s[0])
    else:
        h=int(new_s[0])+12
    m=int(new_s[1][:2])
    return h*60+m

def number_to_time(num):
    h=num//60
    m=num%60
    if h>=12:
        sign="PM"
        h-=12
    else:
        sign="AM"
    time=str(h)+":"+str(m).zfill(2)+sign
    return time

#print(time_to_number("8:00AM"))
# print(number_to_time(480))
# print(number_to_time(1180))

capacity=5
business_hour=["8:00AM","9:00PM"]
reservations=[["9:00AM","9:30AM",3],["9:15AM","9:45AM",2]]

def helper(business_hour,capacity, reservations):
    # Collect all unique times to define intervals
    # times 就是把所有start和end 的时间点都存在一起，因为都是cut的点
    times = set()
    open_time=time_to_number(business_hour[0])
    close_time=time_to_number(business_hour[1])
    times.add(open_time)
    times.add(close_time)


    for start_time, end_time, _ in reservations:
        start_time=time_to_number(start_time)
        end_time=time_to_number(end_time)
        times.add(start_time)
        times.add(end_time)

    # Sort the times to create intervals
    sorted_times = sorted(times)
    #print(sorted_times)
    n=len(sorted_times)
    res=[]
    for i in range(n-1):
        start=sorted_times[i]
        end=sorted_times[i+1]
        current_capacity=capacity
        for start_time,end_time,book in reservations:
            start_time=time_to_number(start_time)
            end_time=time_to_number(end_time)
            if start_time<=start and end_time>=end:
                current_capacity-=book
        res.append([number_to_time(start),number_to_time(end),current_capacity])

    for a,b,c in res:
        print(a,"-",b,c)


helper(business_hour,capacity, reservations)
