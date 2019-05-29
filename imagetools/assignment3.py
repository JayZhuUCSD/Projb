""" Assignment 3
COMPLETE THIS FILE
Your name here:
Jiahao Zhu
"""

from .assignment2 import *

def shift(x, k, l, boundary='periodical'):
    n1, n2 = x.shape[:2]
    if boundary == 'periodical':
        irange = np.mod(np.arange(n1) + k, n1)
        jrange = np.mod(np.arange(n2) + l, n2)
        xshifted = x[irange, :][:, jrange]
        return xshifted
    elif boundary == 'extension':
        irange = np.minimum(np.arange(n1) + k, n1-1)
        jrange = np.maximum(np.arange(n2) + l, 0)
        xshifted = x[irange, :][:, jrange]
        return xshifted
    elif boundary == 'mirror':
        irange = np.arange(n1)+k
        jrange = np.arange(n2)+l
        irange[irange >= n1] = 2*n1-irange[irange >= n1]-1
        irange[irange < 0] = -irange[irange < 0]-1
        jrange[jrange >= n2] = 2*n2-jrange[jrange >= n2]-1
        jrange[jrange < 0] = -jrange[jrange < 0]-1
        xshifted = x[irange,:][:,jrange]
        return xshifted
    elif boundary == 'zero-padding':
        irange = np.arange(n1)+k
        jrange = np.arange(n2)+l
        maski = (irange >= n1) | (irange < 0)
        maskj = (jrange >= n2) | (jrange < 0)
        irange = np.mod(np.arange(n1) + k, n1)
        jrange = np.mod(np.arange(n2) + l, n2)
        xshifted = x[irange, :][:, jrange]
        xshifted[maski, :] = 0
        xshifted[:, maskj] = 0
        return xshifted


def kernel(name, tau=1, eps=1e-3): 
    if name.startswith('box'):
        if name.endswith('1'):
            nu = np.ones((2*tau+1, 1))
            nu = nu/ nu.sum()
        elif name.endswith('2'):
            nu = np.ones((1, 2*tau+1))
            nu = nu/ nu.sum()
        else:
            nu = np.ones((2*tau+1, 2*tau+1))
            nu = nu/ nu.sum()

    elif name.startswith('gaussian'):
        s = int(math.sqrt((-2)*(tau**2)*(math.log(eps))))
        if name.endswith('1'):
            s1, s2 = s, 0
        elif name.endswith('2'):
            s1, s2 = 0, s
        else:
            s1, s2 = s, s
        ni = np.arange(-s1, s1 + 1, 1)
        nj = np.arange(-s2, s2 + 1, 1)
        ii, jj = np.meshgrid(ni, nj, indexing = 'ij')
        f = np.exp((ii**2+jj**2)/(-2*(tau**2)))
        nu = f/np.sum(f)

    elif name.startswith('exponential'):
        s = int((-tau)*(math.log(eps)))
        if name.endswith('1'):
            s1, s2 = s, 0
        elif name.endswith('2'):
            s1, s2 = 0, s
        else:
            s1, s2 = s, s
        ni = np.arange(-s1, s1 + 1, 1)
        nj = np.arange(-s2, s2 + 1, 1)
        ii, jj = np.meshgrid(ni, nj, indexing = 'ij')
        f = np.exp(np.sqrt(ii**2+jj**2)/(-tau))
        nu = f/np.sum(f)

    elif name == 'motion':
        nu = np.load('assets/motionblur.npy')

    elif name == 'grad1_forward':
        nu = np.zeros((3, 1))
        nu[1, 0] = -1
        nu[2, 0] = 1

    elif name == 'grad1_backward':
        nu = np.zeros((3, 1))
        nu[0, 0] = -1 
        nu[1, 0] = 1
        
    elif name == 'grad2_forward':
        nu = np.zeros((1, 3))
        nu[0, 1] = -1
        nu[0, 2] = 1
        
    elif name == 'grad2_backward':
        nu = np.zeros((1, 3))
        nu[0, 0] = -1
        nu[0, 1] = 1
        
    elif name == 'laplacian1':
        nu = np.zeros((3, 1))
        nu[0, 0] = 1
        nu[1, 0] = -2
        nu[2, 0] = 1
        
    elif name == 'laplacian2':
        nu = np.zeros((1, 3))
        nu[0, 0] = 1
        nu[0, 1] = -2
        nu[0, 2] = 1
    return nu


def convolve(x, nu, boundary = 'periodical', separable = None):
    if separable == None:
        s1 = int((nu.shape[0] - 1) / 2)
        s2 = int((nu.shape[1] - 1) / 2)
        xconv = np.zeros(x.shape)
        for k in range(-s1, s1+1):
            for l in range(-s2, s2+1):
                xconv += nu[k+s1][l+s2] * shift(x, -k, -l, boundary)
        return xconv

    elif separable == 'product':
        nu1 = nu[0]
        nu2 = nu[1]
        xconv1 = convolve(x, nu1, boundary, separable=None)
        xconv2 = convolve(xconv1, nu2, boundary, separable=None)
        return xconv2
    
    elif separable == 'sum':
        nu1 = nu[0]
        nu2 = nu[1]
        xconv = convolve(x, nu1, boundary, separable=None) + convolve(x, nu2, boundary, separable=None)
        return xconv


def laplacian(x, boundary='periodical'):
    nu1 = kernel('laplacian1')
    nu2 = kernel('laplacian2')
    nu = (nu1, nu2)
    return convolve(x, nu, boundary, separable = 'sum')


def grad(x, boundary='periodical'):
    nu1 = kernel('grad1_forward')
    g1 = convolve(x, nu1, boundary, separable=None)
    nu2 = kernel('grad2_forward')
    g2 = convolve(x, nu2, boundary, separable=None)
    g = np.stack((g1, g2), axis = 2)
    return g


def div(f, boundary='periodical'):
    nu1 = kernel('grad1_backward')
    g1 = convolve(f[:, :, 0], nu1, boundary, separable=None)
    nu2 = kernel('grad2_backward')
    g2 = convolve(f[:, :, 1], nu2, boundary, separable=None)
    d = g1 + g2
    return d
