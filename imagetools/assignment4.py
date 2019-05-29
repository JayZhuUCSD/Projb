""" Assignment 4
COMPLETE THIS FILE
Your name here: Jiahao Zhu
"""

from .assignment3 import *

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

def bilateral_naive(y, sig, s1=2, s2=2, h=1):
    n1, n2 = y.shape[:2]
    c = y.shape[2] if y.ndim == 3 else 1
    x = np.zeros(y.shape)
    Z = np.zeros((n1, n2, *[1] * (y.ndim - 2)))
    
    for i in range(s1, n1-s1):
        for j in range(s2, n2-s2):
            for k in range(-s1, s1 + 1):
                for l in range(-s2, s2 + 1):
                    # complete
                    dist2 = np.mean((y[i + k, j + l] - y[i, j])**2)
                    num = (dist2-2*h*sig**2)*((dist2-2*h*sig**2)>0)
                    den = (2*np.sqrt(2)*h*sig**2)/np.sqrt(c)
                    phi = np.exp(-num/den)
                    x[i,j] += phi*y[i+k,j+l]
                    Z[i,j] += phi
    Z[Z == 0] = 1
    x = x/Z
    return x


def bilateral(y, sig, s1=10, s2=10, h=1, boundary='mirror'):
    n1, n2 = y.shape[:2]
    c = y.shape[2] if y.ndim == 3 else 1
    x = np.zeros(y.shape)
    Z = np.zeros((n1, n2, *[1] * (y.ndim - 2)))

    for k in range(-s1, s1+1):
        for l in range(-s2, s2+1):
            dist = np.mean((im.shift(y,k,l, boundary)-y)**2,axis =2,keepdims =True)
            num = (dist-2*h*sig**2)*((dist-2*h*sig**2)>0)
            den = (2*np.sqrt(2)*h*sig**2)/np.sqrt(c)
            phi = np.exp(-num/den)
            x += phi * im.shift(y, k, l, boundary)
            Z += phi
    Z[Z == 0] = 1
    x = x/Z
    return x


def nlmeans_naive(y, sig, s1=2, s2=2, p1=1, p2=1, h=1):
    n1, n2 = y.shape[:2]
    c = y.shape[2] if y.ndim == 3 else 1
    p = (2*p1+1)*(2*p2+1)
    x = np.zeros(y.shape)
    Z = np.zeros((n1, n2, *[1] * (y.ndim - 2)))
    
    for i in range(s1, n1-s1-p1):
        for j in range(s2, n2-s2-p2):
            for k in range(-s1, s1 + 1):
                for l in range(-s2, s2 + 1):
                    dist2 = 0
                    for u in range(-p1, p1 + 1):
                        for v in range(-p2, p2 + 1):
                            dist2 += ((y[i + k + u, j + l + v] - y[i + u, j + v])**2).mean()
                    dist2 = dist2/p
                    num = (dist2-2*h*sig**2)*((dist2-2*h*sig**2)>0)
                    den = (2*np.sqrt(2)*h*sig**2) / np.sqrt(c*p)
                    phi = np.exp(-num/den)
                    x[i,j] += phi*y[i+k,j+l]
                    Z[i,j] += phi
    Z[Z == 0] = 1
    x = x/Z
    return x


def nlmeans(y, sig, s1=7, s2=7, p1=None, p2=None, h=1, boundary='mirror'):
    p1 = (1 if y.ndim == 3 else 2) if p1 is None else p1
    p2 = (1 if y.ndim == 3 else 2) if p2 is None else p2
    n1, n2 = y.shape[:2]
    c = y.shape[2] if y.ndim == 3 else 1
    P = (2*p1+1)*(2*p2+1)
    x = np.zeros(y.shape)
    Z = np.zeros((n1, n2, *[1] * (y.ndim - 2)))
    nu1 = im.kernel('box1', p1)
    nu2 = im.kernel('box2', p1)
    nu12 = (nu1, nu2)
    for k in range(-s1, s1+1):
        for l in range(-s2, s2+1):
            dist2 = im.convolve( ((im.shift(y,k,l,boundary)) - y)**2, nu12,boundary='periodical',separable='product')
            dist = np.mean(dist2,axis =2,keepdims =True)
            num = (dist-2*h*sig**2)*((dist-2*h*sig**2)>0)
            den = (2*np.sqrt(2)*h*sig**2) / np.sqrt(c*p)
            phi = np.exp(-num/den)
            x += phi * im.shift(y, k, l, boundary)
            Z += phi
    Z[Z == 0] = 1
    x = x/Z
    return x


def psnr(x,x0):
    R = x.max()
    n = x.size
    return 10*np.log10(R**2*n/np.linalg.norm(x-x0)**2 )