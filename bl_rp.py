# -*- coding: utf-8 -*-
from numpy.linalg import inv
def bl_rp(l_u, l_sigma, budget = [[1, 1, 1, 1, 1, 1]]):
    lmd = 2.5
    tau = 0.8   
    l_w = []
    l_w_mkt =  list(map(rb, l_sigma, budget * len(l_sigma)))
    
    for i in range(len(l_u)):
        if i >= 3:
            w_mkt = l_w_mkt[i]
            P = np.array([0, 0, 0, 0, 0, 0])
            mnt = sum(l_u[i-2:i+1]) / 3.0
            u = mnt.argmax()
            d = mnt.argmin()
            P[u] = 1
            P[d] = -1
            Q = mnt[u] - mnt[d]
            Omega = np.eye(6) * Q
            
            
            
            u = l_u[i]
            sigma = l_sigma[i]
            pai = lmd * (sigma.dot(w_mkt))
            er_l = inv(inv(tau * sigma) + P.dot(inv(Omega)).dot(P))
            er_r = inv(tau * sigma).dot(pai) + P.dot(inv(Omega)).dot(Q)
            ER = er_l.dot(er_r)
            Nsigma = inv(inv(tau * sigma) + P.dot(Omega).dot(P))
        
            l_w.append(mv(ER, Nsigma+sigma, lmd))
        else:
            u = l_u[i]
            sigma = l_sigma[i]
            l_w.append(mv(u, sigma, lmd))
    return l_w


# we need lmd tau Omega P Q
# w_mkt equal weight
# P Q Omega

