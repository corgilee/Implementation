
pins=[(1, 4, 'L'), (2, 3, 'R'), (4, 8, 'R'), (6, 10, 'L')] 
screen=5
#Expected: 2 Output: 2


pins=[(1, 3, 'L'), (2, 4, 'R'), (3, 6, 'L'), (4, 7, 'R'), (6, 9, 'L')] 
screen= 5
#Expected: 3 Output: 3


'''
add all pins into a list and sort by end time

use a heap list to save the pins in the same screen
iterate through each pins and first check if this pin can fit
into the screen, if so, push into the list, use this pin as standard and remove any pins that has start
less then this start
'''
import heapq

def max_pins(pins,screen):
    #sort by end time
    pins.sort(key=lambda x:x[1])
    res=0
    h=[]
    for pin in pins:
        start_cut=pin[1]-screen
        print(start_cut)

        if pin[0]>=start_cut:
            heapq.heappush(h,pin[0])
        while h[0]<start_cut:
            heapq.heappop(h)
        res=max(res,len(h))

    return res

print(max_pins(pins,screen))

