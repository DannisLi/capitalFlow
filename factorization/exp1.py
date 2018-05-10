#-*- coding:utf8 -*-

from skopt import gp_minimize
import numpy as np
import pickle
import math

I,J,K = 10,10,4

with open('../merge/whole_month.data','rb') as f:
    M = pickle.load(f)
M = np.log(M + np.ones(M.shape))

P,_,Q = M.shape

def loss(X, lam=[2.,2.,2.,2.]):
    def L1(mat):
        return np.sum(np.abs(mat))
    X = np.array(X)
    O = X[: P*I].reshape((P,I))
    D = X[P*I : P*I+P*J].reshape((P,J))
    T = X[P*I+P*J : P*I+P*J+Q*K].reshape((Q,K))
    C = X[P*I+P*J+Q*K :].reshape((I,J,K))
    
    M_hat = np.einsum('ijk,pi,qj,rk->pqr', C, O, D, T)
    return np.sum((M - M_hat)**2) + lam[0]*L1(O) + lam[1]*L1(D) + lam[2]*L1(T) + lam[3]*L1(C)


def initial():
    tmp = np.max(M)
    tmp = math.pow(tmp, 0.25)
    return np.random.uniform(0, 6*tmp, (P*I+P*J+Q*K+I*J*K,))


bnds = [(0,100) for i in range(P*I + P*J + Q*K + I*J*K)]

print (gp_minimize(loss, bnds))

