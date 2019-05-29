""" Assignment 5

COMPLETE THIS FILE

Your name here:

"""

from .assignment4 import *

import numpy.fft as npf

def kernel2fft(nu, n1, n2, separable=None):
    if separable == None:
        tmp = np.zeros((n1,n2))
        l1,l2 = nu.shape
        s1 = int((l1-1)/2)
        s2 = int((l2-1)/2)
        tmp[ :s1+1, :s2+1] = nu[s1:2*s1+1, s2:2*s2+1]
        tmp[ :s1+1,n2-s2:] = nu[s1:2*s1+1, :s2]
        tmp[n1-s1:, :s2+1] = nu[:s1, s2:2*s2+1]
        tmp[n1-s1:,n2-s2:] = nu[:s1, :s2]
        lbd = npf.fft2(tmp)
        
    elif separable == 'product':
        nu1,nu2 =nu
        lbd1 = kernel2fft(nu1, n1, n2)
        lbd2 = kernel2fft(nu2, n1, n2)
        lbd = lbd1 * lbd2
        
    elif separable == 'sum':
        nu1,nu2 =nu
        lbd1 = kernel2fft(nu1, n1, n2)
        lbd2 = kernel2fft(nu2, n1, n2)
        lbd = lbd1 + lbd2

    return lbd

def convolvefft(x,lbd):
    lbd3 = np.expand_dims(lbd,axis = 2)
    fr = npf.fft2(x,axes=(0,1))
    imgConvol = np.real(npf.ifft2(fr*lbd3,axes= (0,1)))
    return imgConvol

