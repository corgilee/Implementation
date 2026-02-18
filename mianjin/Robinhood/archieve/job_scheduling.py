# https://www.1point3acres.com/bbs/thread-573007-1-1.html

class Job:
    def __init__(self, start, finish, profit):
        self.start = start
        self.finish = finish
        self.profit = profit

def latest_non_conflict(jobs, i):
    for j in range(i - 1, -1, -1):
        if jobs[j].finish <= jobs[i].start:
            return j
    return -1

# 0/1背包问题， 当前这个job 做还是不做
def find_max_profit_dp(jobs, n):
    table = [0] * n
    table[0] = jobs[0].profit
    
    for i in range(1, n):
        incl_prof = jobs[i].profit
        l = latest_non_conflict(jobs, i)
        if l != -1:
            incl_prof += table[l]
        table[i] = max(incl_prof, table[i - 1])
    
    return table[n - 1]

def find_max_profit(jobs, n):
    jobs.sort(key=lambda x: x.finish) # 根据job 的finish 来sort
    return find_max_profit_dp(jobs, n)

# Example usage
if __name__ == "__main__":
    jobs = [Job(1, 3, 5), Job(2, 5, 6), Job(4, 6, 5), Job(6, 7, 4), Job(5, 8, 11), Job(7, 9, 2)]
    n = len(jobs)
    print("Maximum profit that can be obtained = ", find_max_profit(jobs, n))