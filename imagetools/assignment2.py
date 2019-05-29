""" Assignment 2
COMPLETE THIS FILE
Your name here:
Jiahao Zhu
"""

from .provided import *
import math

def shift(x, k, l, boundary='periodical'):
    n1, n2 = x.shape[:2]
    if boundary is 'periodical_naive':
        xshifted = np.zeros(x.shape)
        # Main part
        for i in range(max(-k, 0), min(n1-k, n1)):
            for j in range(max(-l, 0), min(n2-l, n2)):
                xshifted[i, j] = x[i + k, j + l]
        # Corners
        for i in range(n1 - k, n1):
            for j in range(n2 - l, n2):
                xshifted[i, j] = x[i + k - n1, j + l - n2]
        for i in range(n1 - k, n1):
            for j in range(0, -l):
                xshifted[i, j] = x[i + k - n1, j + l + n2]
        for i in range(0, -k):
            for j in range(n2 - l, n2):
                xshifted[i, j] = x[i + k + n1, j + l - n2]
    elif boundary is 'periodical':
        irange = np.mod(np.arange(n1) + k, n1)
        jrange = np.mod(np.arange(n2) + l, n2)
        xshifted = x[irange, :][:, jrange]
        return xshifted
    elif boundary is 'extension':
        irange = np.minimum(np.arange(n1) + k, n1-1)
        jrange = np.maximum(np.arange(n2) + l, 0)
        xshifted = x[irange, :][:, jrange]
        return xshifted

def kernel(name, tau=1, eps=1e-3): 
    if name is 'box':
        ni = np.arange(0, 2 * tau + 1, 1)
        nj = np.arange(0, 2 * tau + 1, 1)
        ii, jj = np.meshgrid(ni, nj, indexing = 'ij')
        z = (2*tau+1)**2
        f = np.zeros((len(ii), len(jj)))
        nu = np.zeros((len(ii), len(jj)))
        for i in ii:
            for j in jj:
                f[i, j] = 1
                nu[i - tau, j-tau] = 1/z * f[i, j]

    elif name is 'gaussian':
        s = int(math.sqrt((-2)*(tau**2)*(math.log(eps))))
        ni = np.arange(-s, s + 1, 1)
        nj = np.arange(-s, s + 1, 1)
        ii, jj = np.meshgrid(ni, nj, indexing = 'ij')
        f = np.exp((ii**2+jj**2)/(-2*(tau**2)))
        nu = f/np.sum(f)

    elif name is 'exponential':
        s = int((-tau)*(math.log(eps)))
        ni = np.arange(-s, s + 1, 1)
        nj = np.arange(-s, s + 1, 1)
        ii, jj = np.meshgrid(ni, nj, indexing = 'ij')
        f = np.exp(np.sqrt(ii**2+jj**2)/(-tau))
        nu = f/np.sum(f)
    
    return nu

def convolve_naive(x, nu):  
    n1, n2 = x.shape[:2]
    s1 = int((nu.shape[0] - 1) / 2)
    s2 = int((nu.shape[1] - 1) / 2)
    xconv = np.zeros(x.shape)
    for i in range(s1, n1-s1):
        for j in range(s2, n2-s2):
            for k in range(-s1, s1+1):
                for l in range(-s2, s2+1):
                    xconv[i,j] += nu[k][l] * x[i-k][j-l]
    return xconv

def convolve(x, nu, boundary):
    s1 = int((nu.shape[0] - 1) / 2)
    s2 = int((nu.shape[1] - 1) / 2)
    xconv = np.zeros(x.shape)
    for k in range(-s1, s1+1):
        for l in range(-s2, s2+1):
            xconv += nu[k+s1][l+s2] * shift(x, k, l, boundary)
    return xconv
