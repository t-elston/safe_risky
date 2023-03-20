# -*- coding: utf-8 -*-
"""
random helper functions

@author: Thomas Elston
"""
import numpy as np

def makehistbinwidths(binwidth,data):
    bins=np.arange(min(data), max(data) + binwidth, binwidth)
    return bins


def find_sequences(inarray):
        ''' 
        run length encoding. Partial credit to R rle function. 
        Multi datatype arrays catered for including non Numpy
        returns: tuple (runlengths, startpositions, values) 
        '''
        ia = np.asarray(inarray)                # force numpy
        n = len(ia)
        if n == 0: 
            return (None, None, None)
        else:
            y = ia[1:] != ia[:-1]                 # pairwise unequal (string safe)
            i = np.append(np.where(y), n - 1)     # must include last element 
            lens = np.diff(np.append(-1, i))      # run lengths
            pos = np.cumsum(np.append(0, lens))[:-1] # positions
            return(lens, pos, ia[i])

def moving_average(x, w):
    '''
    x - input array
    w - size of window to convolve array with (i.e. smoothness factor)
    '''
    
    # need to figure out how to work with NaN's
    x[np.isnan(x)] = 0
    
    return np.convolve(x, np.ones(w), 'valid') / w

def compute_softmax(val1, val2, beta):
    
    '''
    inputs:
        val1 = value of option 1
        val2 = value of option 2
        beta = inverse temperature 
        
    outputs:
        softmax = likelihood of choosing val1
    '''
    
    softmax = (np.exp(beta*val1)) / (np.exp(beta*val1) + np.exp(beta*val2))
    
    return softmax
