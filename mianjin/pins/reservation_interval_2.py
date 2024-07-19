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



from datetime import datetime, timedelta
from collections import defaultdict

class Solution:
    def time_to_number(self,s):
        new_s=s.split(":")

        sign=new_s[1][-2:]
        if sign=="AM":
            h=int(new_s[0])
        else:
            h=int(new_s[0])+12
        m=int(new_s[1][:2])
        return h*60+m
    
    def number_to_time(self,num):
        h=num//60
        m=num%60
        if h>=12:
            sign="PM"
            h-=12
        else:
            sign="AM"
        time=str(h)+":"+str(m).zfill(2)+sign
        return time


    # number of reservation, rlog(r)
    def calculate_available_capacity(self, open_time, close_time, total_capacity, reservations):
        # Convert times to datetime objects
        # time_format = "%I:%M%p"
        # open_dt = datetime.strptime(open_time, time_format)
        # close_dt = datetime.strptime(close_time, time_format)
        open_dt = self.time_to_number(open_time)
        close_dt = self.time_to_number(close_time)

        # Initialize the changes map
        changes = defaultdict(int)
        changes[open_dt] = 0
        changes[close_dt] = 0

        # Process reservations
        for reservation in reservations:
            # start_dt = datetime.strptime(reservation[0], time_format)
            # end_dt = datetime.strptime(reservation[1], time_format)
            start_dt=self.time_to_number(reservation[0])
            end_dt=self.time_to_number(reservation[1])
            size = reservation[2]
            changes[start_dt] -= size #开始book就减少人数
            changes[end_dt] += size #结束book 就增加人数

        # Sort the time points
        time_points = sorted(changes.keys())

        # Calculate available capacity for each interval
        result = {}
        current_capacity = total_capacity
        for i in range(len(time_points) - 1):
            current_time = time_points[i]
            next_time = time_points[i + 1]
            
            current_capacity += changes[current_time]
            #result[(current_time, next_time)] = max(0, min(total_capacity, current_capacity))
            result[(current_time, next_time)] = current_capacity


        formatted_result = {}
        for (start, end), capacity in result.items():
            # start_str = start.strftime("%I:%M%p")
            # end_str = end.strftime("%I:%M%p")
            start_str = self.number_to_time(start)
            end_str = self.number_to_time(end)
            formatted_result[f"{start_str}-{end_str}"] = capacity

        return formatted_result

# Example usage
solution = Solution()

# Test case 1
open_time = "8:00AM"
close_time = "9:00PM"
total_capacity = 5
reservations = [("9:00AM", "9:30AM", 3)]

result = solution.calculate_available_capacity(open_time, close_time, total_capacity, reservations)
for interval, capacity in result.items():
    print(f"{interval}: {capacity}")

print("\n")

# Test case 2
reservations = [
    ("9:00AM", "9:30AM", 3),
    ("9:15AM", "9:45AM", 2)
]

result = solution.calculate_available_capacity(open_time, close_time, total_capacity, reservations)
for interval, capacity in result.items():
    print(f"{interval}: {capacity}")