import numpy as np
import matplotlib.pyplot as plt
from  scipy.optimize import minimize


def mv(u, sigma, lmd=2.5):
    
    def func(x, u, sigma, lmd, sign=-1.0):
        N = len(sigma)
        L = 0
        for i in range(N):
            L = L + x[i] * u[i]
        R = 0
        for j in range(N):
        	for k in range(N):
        		R = R + x[j] * x[k] * sigma[j][k]

        return sign * (L - lmd / 2.0 * R)

    def func_deriv(x, u, sigma, lmd, sign=-1.0):
    	N = len(sigma)
    	dfdx = []
    	for i in range(N):
    		R = 0
    		for j in range(N):
    			R = R + sigma[i, j] * x[j]
    		
    		dfdxi = sign * (u[i] - lmd * R)
    		dfdx.append(dfdxi)
    	return np.array(dfdx)
    
    N = len(sigma)
    cons = ({'type': 'eq', 
    	     'fun': lambda x: np.array(x.sum() - 1.0),
    	     'jac': lambda x: np.ones(N)})

    #res = minimize(func, [1/N]*N, args=(u, sigma, lmd), jac=func_deriv, \
    res = minimize(func, [1/N]*N, args=(u, sigma, lmd), \
    	constraints=cons, bounds=[(0, 1)]*N, method='SLSQP', tol=1e-12, options={'disp':False, 'maxiter':100})
    return res.x

def mv_ub(u, sigma, rtn=3):
    N = len(sigma)
    e = np.ones(N)
    
    sigma_inv = np.linalg.inv(sigma)
    k1 = e.dot(sigma_inv).dot(u)
    k2 = e.dot(sigma_inv).dot(e)
    k3 = u.dot(sigma_inv).dot(u)
    k4 = u.dot(sigma_inv).dot(e)

    L1 = 1 / k1 - (rtn / k3 - 1 / k1) / (k1 / k3 - k2 / k4) * k2 / k4
    L2 = (rtn / k3 - 1 / k1) / (k1 / k3 - k2 / k4)
    
    w = L1 * (sigma_inv.dot(u)) + L2 * (sigma_inv.dot(e))

    return w

def mv_ub_m(sigma):
    N = len(sigma)
    e = np.ones(N)
    sigma_inv = np.linalg.inv(sigma)
    w = sigma_inv.dot(e) / (e.dot(sigma_inv).dot(e))
    
    return w

def totalrisk(r, u, sigma):
    w = mv_ub(u, sigma, r)
    return w.dot(sigma).dot(w) ** 0.5
    
#d=0
#u = l_u[d]
#sigma = l_sigma[d]
'''
rtn = np.linspace(1, 8)
risk = np.array(list(map(totalrisk, rtn, [u]*50, [sigma]*50)))
w_m = mv_ub_m(sigma)
rtn_m = w_m.dot(u)
risk_m = w_m.dot(sigma).dot(w_m) ** 0.5
plt.figure(dpi=300)
plt.plot(risk*4, rtn)
plt.plot(risk_m*4, rtn_m, 'ro')
plt.xlabel('风险')
plt.ylabel('收益')
plt.title('有效边界曲线')
'''





    
    