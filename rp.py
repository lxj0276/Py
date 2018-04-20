import numpy as np
from  scipy.optimize import minimize

def rp(sigma):

    def func(x, sigma, sign=1.0):
        N = len(sigma)
        R = sigma.dot(x)
        O = 0
        for i in range(N):
            for j in range(N):
                O = O + ((x[i] * R[i] - x[j] * R[j])) ** 2 / R.dot(x)
        return sign * O

    def func_deriv(x, sigma, sign=1.0):
        N = len(sigma)
        R = sigma.dot(x)
        U = R.dot(x)

        def drv(i, k, r=R, s=sigma, w=x):
            if i != k:
                return s[i][k] * w[i]
            else:
                return r.sum() + r[i]
        
        def drv_sigma(k, s=sigma, w=x):
            N = len(sigma)
            D = 0
            for i in range(N):
                D = D + s[k, j] * w[j]
            return D
        
        def drv_total(i, k):
            return  (drv(i, k) * U ** 0.5 - (drv_sigma(k) * x[i] * R[i]) / (2 * U ** 0.5)) / U

        dfdx = []
        for k in range(N):
            S = 0
            for i in range(N):
                for j in range(N):
                    S = S + 2 * ((x[i] * R[i] - x[j] * R[j]) / R.dot(x)) \
                                 * (drv_total(i, k) - drv_total(j, k))
            dfdxk = sign * S
            dfdx.append(dfdxk)

        return np.array(dfdx)

    N = len(sigma)
    cons = ({'type': 'eq', 
            'fun': lambda x: np.array(x.sum() - 1.0),
            'jac': lambda x: np.ones(N)})

    #res = minimize(func, [1/N]*N, args=(sigma), jac=func_deriv, \
    res = minimize(func, [1/N]*N, args=(sigma), \
        constraints=cons, bounds=[(0, 1)]*N, method='SLSQP', tol=1e-18, options={'disp':False, 'maxiter':1000})
    
    return res.x


def rb(sigma, budget=None):
    if budget:
        coef = np.divide(1, budget)
    else:
        coef = 1 / np.ones(len(sigma))

    def func(x, sigma, sign=1.0):
        N = len(sigma)
        R = sigma.dot(x) * (coef)
        S = sigma.dot(x).dot(x)
        O = 0
        for i in range(N):
            for j in range(N):
                O = O + ((x[i] * R[i] - x[j] * R[j])) ** 2 / S
        return sign * O

    N = len(sigma)
    cons = ({'type': 'eq', 
            'fun': lambda x: np.array(x.sum() - 1.0),
            'jac': lambda x: np.ones(N)})

    #res = minimize(func, [1/N]*N, args=(sigma), jac=func_deriv, \
    res = minimize(func, [1/N]*N, args=(sigma), \
        constraints=cons, bounds=[(0, 1)]*N, method='SLSQP', tol=1e-18, options={'disp':False, 'maxiter':1000})
    
    return res.x


def rp_s(arr):
    brr = arr.diagonal()
    return 1 / brr / (sum(1 / brr))