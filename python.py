import numpy as np

def gaus(r, A, s): 
    rAr = (r.T*A*r)
    sr = s.T*r
    print(rAr)
    print(sr)
    return np.exp(-rAr + sr)

A = np.array([[1,2], [3,4]])
s = np.array([[0,0], [0,0]])
r = np.array([[1,0], [0,1]])

print(gaus(r, A, s))