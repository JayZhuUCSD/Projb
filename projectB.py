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

class Gamma(LinearOperator):
    def __init__(self, ishape, zeta):
        LinearOperator.__init__(self, ishape, oshape=None)
        self._zeta = zeta
        self._norm2 = [2.0298192177803034, 1.9256455963809336]     #dirty way to skip calculations for ball
        
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
        #laplacian = kernel('laplacian2')
        #lbd = kernel2fft(laplacian, x.shape[0], x.shape[1])
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

class HBar():
    def __init__(self, H):
        self._H = H
    def __call__(self, X):
        return self._H(X[0])
    def adjoint(self, X):
        return self._H.adjoint(X[0])
    def gram(self,X):
        return [self._H.gram(X[0]),0]
    def gram_resolvent(self, X, tau):
        return [X[0] + tau * self.gram(X)[0], X[1] + tau * self.gram(X)[1]]

def total_variation(y,sig,H=None, m=400, scheme='gd', rho=1, return_energy=False):
    tau = rho * sig
    epsilon = sig**2 * 0.001
    if H is None:
        H = Identity(y.shape)
        
    laplacian = kernel('laplacian2')
    L = H.norm2()**2 + (tau/np.sqrt(epsilon)) * np.sqrt(np.sum(laplacian ** 2))
    gamma = 1 / L
    
    
    def energy(H, x, y, tau):
        G = Grad(x.shape)
        op1 = 0.5 * np.sum(np.square(y-H(x)))
        op2 = tau * np.sum(np.abs(G(x)))
        print('Energy: %f' % (op1+op2))
        return op1 + op2
    
    def energyGen(H,x,y,Gamma,tau):
        op1 = 0.5 * np.sum(np.square(y-H(x)))
        op2 = tau * np.sum(np.abs(Gamma(x)))
        return op1 + op2
    
    def loss(H, x, y, tau, epsilon):
        op1 = H.gram(x) - H.adjoint(y)
        Gx = Grad(x.shape)
        op2 = tau * np.sum((Gx(x) / np.sqrt(np.abs(Gx(x))**2 + epsilon)), axis=2)
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
                print("iteration %d" %i)
                e.append(energy(H,x,y,tau))
                prevX = np.copy(x)
                x = xBar - gamma * loss(H,xBar,y,tau,epsilon)
                xBar = x + calcU(t,t1)*(x-prevX)          
                t = t1
                t1 = calcT1(t)
                if(e[i] > e[i-1]):
                    return prevX, e
            return x, e
        
        else:
            e = []
            for i in range(m):
                print("iteration %d"% i)
                prevX = np.copy(x)
                x = xBar - gamma * loss(H,xBar,y,tau,epsilon)
                xBar = x + calcU(t,t1)*(x-prevX)          
                t = t1
                t1 = calcT1(t)
                if(e[i] > e[i-1]):
                    return prevX
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
                
                x = H.gram_resolvent(xBar, gamma) + H.gram_resolvent(dX, gamma) + H.gram_resolvent((gamma * H.adjoint(y)), gamma)
                e.append(energy(H,x,y,tau))
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
        
    elif scheme is 'cp':
        
        tau = rho * sig
        
        #init
        G = Grad(y.shape)
        gamma = 1
        theta = 1
        k = 1/(G.norm2()**2)
        x = H.adjoint(y)
        z = np.zeros(G(x).shape)
        v = np.zeros(x.shape)
        e = []
        
        for i in range(m):
            e.append(energy(H,x,y,tau))
            print('iteration %d' % i)
            zBar = z + k * G(v)
            z = zBar - softthresh(zBar, tau)
            xBar = x - gamma*G.adjoint(z)
            prevX = np.copy(x)
            x = H.gram_resolvent((xBar + (gamma*H.adjoint(y))), gamma)
            v = x + theta*(x-prevX)
                                 
        if return_energy:
            return x, e
        return x
        
def softthresh(z, t):
    z[np.abs(z) <= t] = 0
    return z - np.sign(z)*t

def tgv(y, sig, H=None, zeta=.1, rho=1, m=400, return_energy=False):
    
    
    def energyGen(H,x,y,Gamma,tau):
        op1 = 0.5 * np.sum(np.square(y-H(x[0])))
        t = Gamma(x)
        
        op2 = tau * (np.sum(np.abs(t[0])) + np.sum(np.abs(t[1])))
        print("Energy: %f" % (op1+op2))
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
    
    if return_energy:
        e = []
        for i in range(m):
            
            print('iteration %d' %i)
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
            
        return X, e
    else:
        for i in range(m):
            print('iteration %d' %i)
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
            
        return X
    
    '''
            # zBar = z + k*Gamma(V)
            t = G(V)
            ZBar[0] = Z[0] + k[0]*t[0]
            ZBar[1] = Z[1] + k[1]*t[1]
            
            # z = zBar - softthresh(zBar, tau)
            Z[0] = ZBar[0] - softthresh(ZBar[0], tau)
            Z[1] = ZBar[1] - softthresh(ZBar[1], tau)
            
            # xBar = x - gamma * G.adjoint(Z)
            t = G.adjoint(Z)
            XBar[0] = X[0] - gamma * t[0]
            XBar[1] = X[1] - gamma * t[1]
            
            #storing previous version of x
            prevX[0] = np.copy(X[0])
            prevX[1] = np.copy(X[1])
            
            # x = HBar.gram_resolvent(XBar + gamma * HBar.adjoint(y), gamma)
            star = XBar[0] + gamma * H.adjoint(y)
            X[0] = star + gamma * H.gram(star)
            X[1] = XBar[1]
            
            # v = s + theta(x - prevX)
            V[0] = X[0] + theta * (X[0] - prevX[0])
            V[1] = X[1] + theta * (X[1] - prevX[1])
            if e[i] > e[i-1]:
                pass
                #return prevX, e
        return X, e
    else:
        e = []
        for i in range(m):
            e.append(energyGen(H,X,y,G,tau))
            # zBar = z + k*Gamma(V)
            t = G(V)
            ZBar[0] = Z[0] + k[0]*t[0]
            ZBar[1] = Z[1] + k[1]*t[1]
            
            # z = zBar - softthresh(zBar, tau)
            Z[0] = ZBar[0] - softthresh(ZBar[0], tau)
            Z[1] = ZBar[1] - softthresh(ZBar[1], tau)
            
            # xBar = x - gamma * G.adjoint(Z)
            t = G.adjoint(Z)
            XBar[0] = X[0] - gamma * t[0]
            XBar[1] = X[1] - gamma * t[1]
            
            #storing previous version of x
            prevX[0] = np.copy(X[0])
            prevX[1] = np.copy(X[1])
            
            # x = HBar.gram_resolvent(XBar + gamma * HBar.adjoint(y), gamma)
            star = XBar[0] + gamma * H.adjoint(y)
            X[0] = star + gamma * H.gram(star)
            X[1] = XBar[1]
            
            # v = s + theta(x - prevX)
            V[0] = X[0] + theta * (X[0] - prevX[0])
            V[1] = X[1] + theta * (X[1] - prevX[1])
            if e[i] > e[i-1]:
                pass
                #return prevX, e
        return X

    def energy(H, x, y, tau):
        Gr = Grad(x.shape)
        op1 = 0.5 * np.sum(np.square(y-H(x)))
        op2 = tau * np.sum(np.abs(Gr(x)))
        return op1 + op2
    

    tau = rho * sig
    if H is None:
        H = Identity(y.shape)  
    x = H.adjoint(y)
    
    G = Gamma(y.shape, zeta)
    gamma = 1
    XBar = [x, np.zeros(Grad(x.shape)(x).shape)]  
    ZBar = [Grad(x.shape)(x), np.zeros(x.shape)]
    DX = [np.zeros(x.shape), np.zeros(Grad(x.shape)(x).shape)]
    DZ = [np.zeros(Grad(x.shape)(x).shape), np.zeros(x.shape)]
    X = None
    Z = None
    
    if return_energy:
        e = []
        for i in range(m):
            if X is not None:
                e.append(energy(H, X[0], y, tau))
            X = H.gram_resolvent(XBar[0], gamma) + H.gram_resolvent(DX[0], gamma) + H.gram_resolvent(gamma * H.adjoint(y), gamma)
            Z = softthresh(ZBar + DZ, gamma*tau)
            XBar = G.gram_resolvent(X, 1) - G.gram_resolvent(DX, 1) + G.gram_resolvent(G.adjoint(Z - DZ), 1)
            ZBar = G(XBar)
            DX = DX - X + XBar
            DZ = DZ - Z + ZBar
        return X
    else:
        pass
    

    tau = rho * sig
    if H is None:
        H = Identity(y.shape)  
    x = H.adjoint(y)
    
    H = HBar(H)
    
    G = Gamma(y.shape, zeta)
    gamma = 1
    theta = 1
    print(G.norm_2())
    k = [1 / (G.norm_2()[0]**2), 1 / (G.norm_2()[1]**2)]
    v = np.zeros(x.shape)
    
    # update param's img -> (img, vec) vec -> (vec, img)
    X = [x, np.zeros(Grad(x.shape)(x).shape)]  
    V = [v, np.zeros(Grad(x.shape)(x).shape)]
    Z = [np.zeros(Grad(x.shape)(x).shape), np.zeros(x.shape)]

    ZBar = None
    XBar = None
    PrevX = None

    if return_energy:
        e = []
        for i in range(m):
            print('iteration %d' %i)
            e.append(energy(H,X[0],y,tau))
            ZBar = Z + k*G(V)
            Z = ZBar - softthresh(ZBar, tau)
            XBar = X - gamma * G.adjoint(Z)
            PrevX = np.copy(X)
            X = H.gram_resolvent(XBar[0], gamma) + H.gram_resolvent(gamma*H.adjoint(y), gamma)
            V = X + theta*(X - PrevX)
        return X, e
    else:
        for i in range(m):
            print('iteration %d' % i)
            ZBar = Z + k*G(V)
            Z = ZBar - softthresh(ZBar, tau)
            XBar = X - gamma * G.adjoint(Z)
            PrevX = np.copy(X)
            X = H.gram_resolvent(XBar[0], gamma) + H.gram_resolvent(gamma*H.adjoint(y), gamma)
            V = X + theta*(X - PrevX)
        return X
    '''