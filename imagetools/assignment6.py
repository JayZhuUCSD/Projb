""" Assignment 6

COMPLETE THIS FILE

Your name here:

"""

from .assignment5 import *

def average_power_spectral_density(x):
    K = len(x)
    apsd = 0
    for k in range(K):
        f = npf.fft2(x[k],axes=(0,1))
        S = np.abs(f)**2
        if x[k].ndim==3:
            apsd += np.mean(S, axis=2)
        else:
            apsd += S
    return apsd/K


def mean_power_spectrum_density(apsd):
    n1,n2 = apsd.shape
    w = np.zeros((n1,n2))
    t = np.zeros((n1,n2))
    u_range, v_range = im.fftgrid(n1,n2)
    for u in range(n1):
        for v in range(n2):
            if u!=0 or v!=0:
                w[u,v] = np.sqrt((u_range[u,v]/n1)**2+(v_range[u,v]/n2)**2)
                t[u,v] = np.log(w[u,v])

    s = np.log(apsd)-np.log(n1)-np.log(n2)
    s[0,0] = 0
    
    s_avg = np.sum(s)/(n1*n2-1)
    t_avg = np.sum(t)/(n1*n2-1)
    alpha = np.sum(s*t-s_avg*t)/np.sum(t*t-t_avg*t)
    beta = s_avg - alpha*t_avg
    mpsd = np.zeros((n1,n2))
    for u in range(n1):
        for v in range(n2):
            if u!=0 or v!=0:
                mpsd[u,v] = n1*n2*np.exp(beta)*w[u,v]**alpha
    
    mpsd[0,0] =np.inf
    return mpsd, alpha, beta