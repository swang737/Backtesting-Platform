#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
from DataLoader import DataLoader
from statsmodels.tsa.stattools import acf
import math


# In[214]:


class EDA:
    def __init__(self, DataLoader):
        self.loader = DataLoader

    def getACF(self, graph = True, lag = 20, ncols = 3, conf = True):
        '''
        returns the ACFs of all stocks at this timestep for the last 20 lags and graphs all of them.
        '''
        # error handling
        if lag < 1 or isinstance(lag, int) == False:
            raise ValueError('lag must be integer greater than 1')

        # get data and making acfs array
        data = self.loader.returnsToNow() # all returns of required stocks for last lags + today
        acfs = []

        if graph:
            ncols = max(1, int(ncols))
            nrows = math.ceil(self.loader.nins / ncols)
            fig, axes = subplots(nrows, ncols, figsize = (4 * ncols, 3 * nrows), sharex = True, sharey = True) # same x/y axis
            axes = np.atleast_1d(axes).ravel() # atleast_1d --> if its 1d it makes it 2d by chucking brackets around it
            # ravel so can go axes[i] and put them one by one

        for i in range(self.loader.nins):
            stock = data[:, i] # stock data for one stock

            # getting acfs and confidence intervals
            acf_vals, conf_int = acf(stock, nlags = lag, fft=False, missing='conservative', alpha = 0.05) # dont take shortcut and take out NAs
            acfs.append(acf_vals)

            if graph:
                ax = axes[i]
                ax.axhline(0, c = 'k', lw = 0.5)
                ax.bar(np.arange(lag + 1), acf_vals, width = 0.3, align = 'center') # align puts it at the lag (center)
                ax.plot(acf_vals, lw = 0.5, c = 'r')
                if conf:
                    bound = 1.96 / np.sqrt(len(stock))
                    ax.fill_between(np.arange(lag + 1), -bound, bound, alpha = 0.1, color = 'm')
                ax.set_title(f'Stock {self.loader.stocks[i]}')
                ax.set_xlabel('Lags')
                ax.set_ylabel('ACF')
                ax.set_xlim(0, lag)
        if graph:
            for j in range(self.loader.nins, len(axes)):
                fig.delaxes(axes[j]) # removes the axes that weren't used
            fig.tight_layout() # adjust so titles dont collide and squeeze shit in
            
            return np.vstack(acfs)

