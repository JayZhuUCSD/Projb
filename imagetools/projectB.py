""" Project B

COMPLETE THIS FILE

Your names here:
Jiahao Zhu
Luke Wulf
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

def softthresh(z, t):
    z[np.abs(z) <= t] = 0
    return z - np.sign(z)*t

def total_variation(y, sig, H = None, m = 400, scheme = 'gd', rho = 1, return_energy = False):
    tau = rho * sig
    epsilon = sig**2 * 0.001
    if H is None:
        H = Identity(y.shape)
   
    L = H.norm2()**2 + (tau/np.sqrt(epsilon)) * Grad(y.shape).norm2()**2
    gamma = 1 / L
    
    def energy(H, x, y, tau):
        op1 = 0.5 * la.norm(y-H(x))**2
        op2 = tau * np.sum(np.abs(Grad(x.shape)(x)))
        return op1 + op2
    
    def loss(H, x, y, tau, epsilon):
        op1 = H.gram(x) - H.adjoint(y)
        Gx = Grad(x.shape)(x)
        op2 = tau * div(Gx / np.sqrt(np.abs(Gx)**2 + epsilon))
        return op1 - op2
    
    if scheme is 'gd':
        x = H.adjoint(y)
        e = []        
        for i in range(m):
            e.append(energy(H,x,y,tau))
            x = x - gamma * loss(H,x,y,tau,epsilon)   
        if return_energy:           
            return x, e
        else:
            return x
        
    elif scheme is 'nesterov':      
        xBar = H.adjoint(y)
        x = H.adjoint(y)
        t = 1
 
        def calcT(t):
            return (1 + np.sqrt(1 + 4*t**2)) / 2     

        e = []
        for i in range(m):
            e.append(energy(H,x,y,tau))
            prevX = np.copy(x)
            x = xBar - gamma * loss(H,xBar,y,tau,epsilon)
            xBar = x + (t-1)/calcT(t)*(x-prevX)          
            t = calcT(t)
        
        if return_energy:
            return x, e       
        else:
            return x
        
    elif scheme is 'admm':      
        #initialize iteratable variables
        gamma = 1
        xBar = H.adjoint(y)
        zBar = Grad(y.shape)(xBar)
        dX = np.zeros(y.shape)
        dZ = np.zeros(zBar.shape)

        e = []
        for i in range(m):          
            x = H.gram_resolvent(xBar + dX + gamma*H.adjoint(y), gamma)
            e.append(energy(H,x,y,tau))
            z = softthresh(zBar + dZ, gamma*tau)
            xBar = Grad(y.shape).gram_resolvent(x-dX+Grad(zBar.shape).adjoint(z-dZ), 1)
            zBar = Grad(y.shape)(xBar)
            dX = dX - x + xBar
            dZ = dZ - z + zBar
        
        if return_energy:
            return x, e
        else:
            return x
        
    elif scheme is 'cp':
        #initialize iteratable variables
        gamma = 1
        theta = 1
        k = 1/(Grad(y.shape).norm2()**2)
        x = H.adjoint(y)
        z = np.zeros(Grad(y.shape)(y).shape)
        v = np.zeros(y.shape)

        e = [] 
        for i in range(m):
            e.append(energy(H,x,y,tau))
            prevX = np.copy(x)
            zBar = z + k * Grad(y.shape)(v)
            z = zBar - softthresh(zBar, tau)
            xBar = x - gamma * Grad(z.shape).adjoint(z)
            x = H.gram_resolvent(xBar + gamma*H.adjoint(y), gamma)
            v = x + theta*(x-prevX)
                                 
        if return_energy:
            return x, e
        else:
            return x

def tgv(y, sig, H=None, zeta=.1, rho=1, m=400, return_energy=False):  
    
    def energyGen(H,x,y,Gamma,tau):
        op1 = 0.5 * la.norm(y-H(x[0]))**2
        t = Gamma(x)    
        op2 = tau * (np.sum(np.abs(t[0])) + np.sum(np.abs(t[1])))
        return op1 + op2
    
    if H is None:
        H = Identity(y.shape)
    
    gamma = 1
    G = Gamma(y.shape, zeta)
    tau = rho * sig
    gr = Grad(y.shape)
    zeroGrad = np.zeros(gr(y).shape)
    zeroImg = np.zeros(y.shape)
    
    X = [None, None]
    Z = [None, None]
    ZBar = [None, None]
    XBar = [None, None]
    DX = [None, None]
    DZ = [None, None] 
    XBar[0] = H.adjoint(y)
    XBar[1] = np.copy(zeroGrad)  
    ZBar[0] = gr(XBar[0])
    ZBar[1] = np.copy(zeroImg)   
    DX[0] = np.copy(zeroImg)
    DX[1] = np.copy(zeroGrad)
    DZ[0] = np.copy(zeroGrad)
    DZ[1] = np.copy(zeroImg)

    e = []
    for i in range(m):
        star = XBar[0] + DX[0] + gamma * H.adjoint(y)
        X[0] = star + gamma * H.gram(star)
        X[1] = XBar[1] + DX[1]     
        Z[0] = softthresh(ZBar[0] + DZ[0], gamma*tau)
        Z[1] = softthresh(ZBar[1] + DZ[1], gamma*tau)       
        star2 = [None, None]
        inner = [None, None]
        inner[0] = Z[0] - DZ[0]
        inner[1] = Z[1] - DZ[1]
        t = G.adjoint(inner)
        star2[0] = X[0] - DX[0] + t[0]
        star2[1] = X[1] - DX[1] + t[1]
        t2 = G.gram_resolvent(star2, 1)
        XBar[0] = t2[0]
        XBar[1] = t2[1]
        
        t = G(XBar)
        ZBar[0] = t[0]
        ZBar[1] = t[1]
        DX[0] = DX[0] - X[0] + XBar[0]
        DX[1] = DX[1] - X[1] + XBar[1]
        DZ[0] = DZ[0] - Z[0] + ZBar[0]
        DZ[1] = DZ[1] - Z[1] + ZBar[1]
        e.append(energyGen(H,X,y,G,tau))
    
    if return_energy:
        return X, e
    else:      
        return X


class Gamma(LinearOperator):
    def __init__(self, ishape, zeta):
        LinearOperator.__init__(self, ishape, oshape=None)
        self._zeta = zeta
        self._norm2 = [2.0298192177803034, 1.9256455963809336]
        
    def __call__(self, X):
        x = X[0]
        z = X[1]
        Gx = Grad(x.shape)
        Gz = Grad(z.shape)
        term1 = Gx(x) - self._zeta * z
        term2 = np.sum(z, axis=2)
        return [term1,term2]
    
    def adjoint(self, X):
        z = X[0]
        x = X[1]
        Gx = Grad(x.shape)
        term1 = -np.sum(z, axis=2)
        term2 = -Gx(x) - self._zeta * z
        return [term1, term2]
    
    def gram(self, X):
        x = X[0]
        z = X[1]
        Gx = Grad(x.shape)
        term1 = np.sum(Gx(x), axis=2) + self._zeta * np.sum(z, axis=2)
        term2 = -self._zeta*Gx(x) + (self._zeta**2) * z - Gx(np.sum(z, axis=2))
        return [term1, term2]
    
    def gram_resolvent(self, X, tau):
        inter = self.gram(X)    
        return X + [inter[0] * tau, inter[1] * tau]
    
    def norm_2(self):
        if self._norm2 is not None:
            return self._norm2
        K = 100
        x = np.random.randn(*self.ishape)
        z = np.zeros(Grad(x.shape)(x).shape)
        X = [x,z]
        for k in range(K):
            y = self.gram(X)
            x = y[0] / np.sqrt((y[0]**2).sum())
            z = y[1] / np.sqrt((y[1]**2).sum())
            X = [x,z]
        self._norm2 = [np.sqrt(np.sqrt((y[0]**2).sum())), np.sqrt(np.sqrt((y[1]**2).sum()))]
        return self._norm2