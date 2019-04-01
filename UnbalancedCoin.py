# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 17:39:43 2019

@author: WZ009830
"""

import random
import numpy

class UnbalancedCoin:
    
    def __init__(self, p):
        #assert 0.0 < p < 1.0, 'invalid p'
        self._p = p
    
    def Flip(self):
        return random.uniform(0,1) < self._p

    def MakeEqualProb(self):
      while True:
        a = self.Flip()
        if a != self.Flip():
          return a
      
if __name__ == '__main__':
    test=UnbalancedCoin(0.8)
    count=[]
    for i in range(1000):
        count.append(test.MakeEqualProb())
    print(numpy.mean(count))
