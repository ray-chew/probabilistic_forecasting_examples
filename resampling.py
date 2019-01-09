import numpy as np

# define residual resampling function
def resampling(X,w,M,J):
    wm = M*w
    wi = np.floor(wm)
    nc = np.sum(wi).astype(int)

    A = np.zeros((J,nc))
    X = X.reshape(J,-1)
    
    if (nc > 0):
        k = 0
        for i in range(M):
            for j in range(int(wi[i])):
                #if j > 0:
                A[:,k] = X[:,i]
                k += 1
    else:
        A = np.zeros((J,0))
        
    wmd = wm - wi
    wmd /= np.sum(wmd)
    Nk = M - nc
    
    B = np.zeros((J,Nk))
    
    if (Nk > 0):
        exponent = 1./(np.arange(1,Nk+1)[::-1]).astype(float)
        u = np.cumprod(np.power(np.random.rand(Nk),exponent))
        u = u[::-1]

        wcu = np.cumsum(wmd)
        
        ind = np.zeros((Nk)).astype(int)
        kk = 0
        
        for ll in range(Nk):
            while (wcu[kk] < u[ll]):
                kk += 1
                
            ind[ll] = kk
        B = X[:,ind]
        
    else:
        B = np.zeros((J,0))

    return A, B 
