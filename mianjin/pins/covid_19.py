'''
Question delivery
Given an array of visits(user_id, location_id, start_time, end_time) and an array of users
infected with Covid 19 users write a function that returns the total number of users that caught
Covid19.
When a user gets infected they will start infecting other users afterwards immediately
meaning the person becomes contagious right after being infected and there is no
incubation period.
Additional Data
Here is an example input each element of visits array is (userid, locationid, start_time, end_time)

visits = [
[0, 0, 1, 3],
[0, 1, 4, 5],
[0, 2, 8, 9],
[1, 1, 4, 6],
[2, 2, 7, 9],
[3, 2, 6, 8],
]
infected = [1]
infection_simulator(visits, infected) # returns 3
# first 1 infects 0 at location 1
# then 0 infects 2 at location 2
# 3 remains uninfected cause they leave as soon as 0 arrive at location 2


Follow-up Questions / Variations
● How would you change the solution if covid19 didn’t start spreading immediately but
rather had an incubation period of K time units before being contagious.
● How would you handle recovery if users would recover after X time from they got
infected.
○ Additional input: An array of (userid, test_time) which tells that this particular user
has tested at this time and the test result was negative.
● How would you solve the problem if data arrived in a streaming/online fashion rather
than everything upfront?

'''
# a=[[1,3],[-1,2]]
# a.sort()

a=[["12323"]+[]]

print(a)
