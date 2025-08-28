#!/usr/bin/env python
# coding: utf-8

# In[306]:


import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots


# In[380]:


class DataLoader:
    def __init__(self, data, stocks = None):
        # prices must be PER DAY     
        if stocks is None:
            self.data = data
            self.nt, self.nins = self.data.shape
            self.stocks = range(self.nins)
        else:
            self.data = data[:, stocks[0]:stocks[0]+1]

            for i in range(1, len(stocks)):
                if not all(isinstance(i, int) and 0 <= i < data.shape[1] for i in stocks):
                    raise ValueError("All elements in 'stocks' must be valid integer column indices.")
                self.data = np.hstack([self.data, data[:, stocks[i]:stocks[i]+1]])
            self.nt, self.nins = self.data.shape
            self.stocks = stocks
        self.t = 0 # set curret time step to 0

    def resetTime(self):
        '''
        resets time step back to start
        '''
        self.t = 0

    def goToTime(self, i):
        '''
        puts current time at certain time step
        '''
        if type(i) == int and i >= 0 and i < self.nt:
            self.t = i
        else:
            raise IndexError('Either i not an int or out of bounds')

    def currentTime(self):
        '''
        returns current time step
        '''
        return self.t

    def stepTime(self, step = 1):
        '''
        return current prices, then advances one step
        '''
        if self.t >= self.nt: # check to see if end of data
            raise IndexError('No more data')
        prices = self.data[self.t]
        self.t += step
        return prices

    def lookBack(self, t):
        '''
        returns the previous specified amount of time steps
        '''
        start = max(0, self.t - t) # start at t timesteps before
        return self.data[start:self.t]

    def currentPrices(self):
        '''
        returns current prices at this timestep
        '''
        if self.t >= self.nt:
            raise IndexError('No more data')
        return self.data[self.t]

    def getReturns(self, log = True):
        '''
        returns the returns for today from yesterday (today - yesterday)
        '''
        if self.t <= 0:
            return np.full(self.nins, np.nan, dtype=float)
        
        today = self.data[self.t]
        yesterday = self.data[self.t - 1]

        with np.errstate(divide = 'ignore', invalid = 'ignore'): # ignore all division and log errors and replace with inf or nan
            if log:
                returns = np.log(today/yesterday)
            else: returns = today/yesterday - 1
                
        return returns

    def lookBackReturns(self, t_input, log = True):
        '''
        computes returns of every day until t days ago, (t outputs)
        '''
        # error handling
        t = min(self.t, t_input) # ensuring t is not bigger than self.t
        if t == 0:
            raise ValueError('cannot lookback 0 timesteps')

        lookback = self.data[self.t - t:self.t + 1]

        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            if log:
                returns = np.log(lookback[1:]/lookback[:-1]) # iterates but numpy vector
            else:
                returns = lookback[1:]/lookback[:-1] - 1

        return returns

    def returnsToNow(self, log = True):
        '''
        computes returns of every day until today
        '''
        lookback = self.data[:self.t + 1]
        
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            if log:
                returns = np.log(lookback[1:]/lookback[:-1])
            else:
                returns = lookback[1:]/lookback[:-1] - 1
                
        return returns