""" Project B

COMPLETE THIS FILE

Your names here:

"""

from .assignment6 import *
from .provided import *

class Identity(LinearOperator):
    def __init__(self, ishape, oshape=None):
        LinearOperator.__init__(self, ishape, oshape=None)
        self._ishape = ishape
        self.nu = kernel("box",tau=0)
        self.mu = np.flip(self.nu,1)[::-1]
        self.H = kernel2fft(self.nu, ishape[0], ishape[1])
        self.H_star = kernel2fft(self.mu, ishape[0], ishape[1])
    def __call__(self, x):
        return convolvefft(x, self.H)
    def adjoint(self, x):
        return convolvefft(x, self.H_star)
    def gram(self, x):
        return convolvefft(x, self.H*self.H_star)
    def gram_resolvent(self, x, tau):
        res_lbd = 1 / (1 + tau * self.H*self.H_star)
        return convolvefft(x, res_lbd)

class Convolution(LinearOperator):
    def __init__(self, ishape, nu, separable=None, oshape=None):
        LinearOperator.__init__(self, ishape, oshape=None)
        self._ishape = ishape
        self.nu = nu
        self._separable = separable
        self.mu = np.flip(self.nu,1)[::-1]
        self.H = kernel2fft(self.nu, ishape[0], ishape[1], self._separable)
        self.H_star = kernel2fft(self.mu, ishape[0], ishape[1], self._separable)
    def __call__(self, x):
        return convolvefft(x, self.H)
    def adjoint(self, x):
        return convolvefft(x, self.H_star)
    def gram(self, x):
        return convolvefft(x, self.H*self.H_star)
    def gram_resolvent(self, x, tau):
        res_lbd = 1 / (1 + tau * self.H*self.H_star)
        return convolvefft(x, res_lbd)

class RandomMasking(LinearOperator):
    def __init__(self, ishape, p):
        LinearOperator.__init__(self, ishape, oshape=None)
        self._p = p
        self._mask = np.random.choice([0,1], size=ishape, p=[self._p, 1-self._p])
    def __call__(self, x):
        return np.multiply(self._mask, x)
    def adjoint(self, x):
        return np.multiply(self._mask, x)  # since conjugate of random mask is the same mask
    def gram(self, x):
        return np.multiply(self._mask, x)
    def gram_resolvent(self,x,tau):
        return x + (tau * self(x))
    
def total_variation(y,sig,H=None, m=400, scheme='gd', rho=1, return_energy=False):
    tau = rho * sig
    epsilon = sig**2 * 0.001
    if H is None:
        H = Identity(y.shape)
        
    laplacian = kernel('laplacian2')
    L = H.norm2()**2 + (tau/np.sqrt(epsilon)) * np.sqrt(np.sum(laplacian ** 2))
    gamma = 1 / L
    
    
    def energy(H, x, y, tau):
        op1 = 0.5 * np.sum((y-H(x))** 2)
        op2 = tau * np.sum(np.gradient(x))
        return op1 + op2
    
    def loss(H, x, y, tau, epsilon):
        op1 = H.gram(x) - H.adjoint(y)
        gradX = grad(x)
        op2 = tau * np.sum((gradX / np.sqrt(np.abs(gradX)**2 + epsilon)), axis=2)
        return op1 - op2
    
    x = H.adjoint(y)
    
    if scheme is 'gd':
        if return_energy:
            e = []        
            for i in range(m):
                e.append(energy(H,x,y,tau))
                x = x - gamma * loss(H,x,y,tau,epsilon)              
            return x, e

        else:
            for i in range(m):
                print("iteration %d"% i)
                x = x - gamma * loss(H,x,y,tau,epsilon)
            return x
        
    elif scheme is 'nesterov':
        
        xBar = H.adjoint(y)
        prevX = x
 
        def calcT1(t):
            return (1 + np.sqrt(1 + 4*t**2)) / 2
        
        def calcU(t, t1):
            return (t-1)/t1
        
        t = 1
        t1 = calcT1(t)
        
        if return_energy:
            e = []
            for i in range(m):
                e.append(energy(H,x,y,tau))
                prevX = x
                x = xBar - gamma * loss(H,xBar,y,tau,epsilon)
                xBar = x + calcU(t,t1)*(x-prevX)          
                t = t1
                t1 = calcT1(t)
            return x, e
        
        else:
            for i in range(m):
                print("iteration %d"% i)
                prevX = x
                x = xBar - gamma * loss(H,xBar,y,tau,epsilon)
                xBar = x + calcU(t,t1)*(x-prevX)          
                t = t1
                t1 = calcT1(t)
            return x
        
    elif scheme is 'admm':
        
        #initialize iteratable variables
        gamma = 1
        G = Grad(y.shape)
        xBar = H.adjoint(y)
        zBar = G(xBar)
        dX = np.zeros(xBar.shape)
        dZ = np.zeros(zBar.shape)
        x = None
        z = None
        
        if return_energy:
            e = []
            for i in range(m):
                e.append(energy(H,x,y,tau))
                x = H.gram_resolvent(xBar, gamma) + H.gram_resolvent(dX, gamma) + H.gram_resolvent((gamma * H.adjoint(y)), gamma)
                z = softthresh(zBar + dZ, gamma*tau)
                xBar = G.gram_resolvent(x, 1) - G.gram_resolvent(dX, 1) + G.gram_resolvent(G.adjoint(np.sum(z-dZ, axis=2)),1)
                zBar = G(xBar)
                dX = dX - x + xBar
                dZ = dZ - z + zBar
            return x, e
        else:
            for i in range(m):
                print('iteration %d' % i)
                x = H.gram_resolvent(xBar, gamma) + H.gram_resolvent(dX, gamma) + H.gram_resolvent(gamma * H.adjoint(y), gamma)
                z = softthresh(zBar + dZ, gamma*tau)
                xBar = G.gram_resolvent(x, 1) - G.gram_resolvent(dX, 1) + G.gram_resolvent(G.adjoint(np.sum(z-dZ, axis=2)),1)
                zBar = G(xBar)
                dX = dX - x + xBar
                dZ = dZ - z + zBar
            return x
        
def softthresh(z, t):
    z[np.abs(z) <= t] = 0
    return z - np.sign(z)*t