#-*- coding:utf8 -*-

import math
import pickle
import numpy as np
from pymatbridge import Matlab
import scipy.optimize as sciopt


items = [
    'a', 'ag', 'al', 'au', 'b', 'bb', 'bu', 'c', 'cf', 'cu', 'fb', 'fg', 'fu', 'i', 'j', 'jd', 
    'jm', 'jr', 'l', 'm', 'oi', 'p', 'pb', 'pm', 'rb', 'ri', 'rm', 'rs', 'ru', 'sr', 'ta', 'v', 
    'wh', 'wr', 'y', 'zn', 'fund'
]

item2name = {
    'a': '豆一', 'ag': '沪银', 'al': '沪铝', 'au': '沪金', 'b': '豆二', 'bb': '胶合板', 'bu': '沥青', 'c': '玉米', 'cf': '郑棉', 
    'cu': '沪铜', 'fb': '纤维板', 'fg': '玻璃', 'fu': '燃油', 'i': '铁矿石', 'j': '焦炭', 'jd': '鸡蛋', 'jm': '焦煤', 'jr': '粳稻', 
    'l': '塑料', 'm': '豆粕', 'oi': '菜油', 'p': '棕榈油', 'pb': '沪铅', 'pm': '普麦', 'rb': '螺纹钢', 'ri': '早籼稻', 'rm': '菜粕', 
    'rs': '菜籽', 'ru': '橡胶', 'sr': '白糖', 'ta': 'PTA', 'v': 'PVC', 'wh': '强麦', 'wr': '线材', 'y': '豆油', 'zn': '沪锌', 'fund': 'fund'
}

def _SP(mat):
    # 计算矩阵mat的稀疏度: [0,1]
    def L1(mat):
        return np.sum(np.abs(mat))
    def L2(mat):
        return math.sqrt(np.sum(mat*mat))
    n = mat.size
    l1 = L1(mat)
    l2 = L2(mat)
    return (math.sqrt(n) - l1/l2) / (math.sqrt(n)-1)

def col_norm(mat, inplace=False):
    # 矩阵列中心化
    R = np.copy(mat)
    col_sum = np.sum(mat, axis=0)
    for i in range(len(mat)):
        R[i] = R[i] / col_sum
    return R

class CF(object):
    def __init__(self, datapath, preprocess=['log10']):
        '''
        preprocess: 
            'log'         加一取对数
            'sqrt'        平方根
            'symmetric'   对称元消除 
        '''
        # 读取数据
        with open(datapath, 'rb') as f:
            self.data = pickle.load(f)
        M,_,N = self.data.shape
        # 去掉部分资金流动
        '''
        tmp = np.delete(self.data, delete, axis=0)
        tmp = np.delete(tmp, delete, axis=1)
        for i in delete:
            for j in range(M):
                if j not in delete:
                    cnt = 0
                    for k in delete:
                        if k<j:
                           cnt += 1
                    # 将i转移到其他商品 改为 fund转移到其他商品
                    tmp[-1,j-cnt] += self.data[i,j]
                    # 将其他商品转移到i 改为 其他商品转移到fund
                    tmp[j-cnt,-1] += self.data[j,i]
        self.data = tmp
        print (self.data.shape)
        '''
        # 预处理
        for pro in preprocess:
            if pro == 'log':
                self.data = np.log(self.data + np.ones(self.data.shape))
            elif pro == 'log10':
                self.data = np.log10(self.data + np.ones(self.data.shape))
            elif pro == 'sqrt':
                self.data = np.sqrt(self.data)
            elif pro == 'symmetric':
                M,_,N = self.data.shape
                for k in range(N):
                    for i in range(M):
                        for j in range(i+1,M):
                            if self.data[i,j,k] > self.data[j,i,k]:
                                self.data[i,j,k] -= self.data[j,i,k]
                                self.data[j,i,k] = 0
                            else:
                                self.data[j,i,k] -= self.data[i,j,k]
                                self.data[i,j,k] = 0
    
    def MDS(self, link=True, val=30):
        N = self.data.shape[2]
        dis = np.zeros((n,n), np.float32)
        for i in range(N):
            for j in range(i+1,N):
                # 使用L1范数引导的距离
                dis[i,j] = np.sum(np.abs(self.data[:,:,i] - self.data[:,:,j]))
                dis[j,i] = dis[i,j]
        mds = MDS(n_components=2, dissimilarity='precomputed', n_jobs=-1)
        coor = mds.fit_transform(dis).T
        plt.figure(figsize=(12,10))
        plt.scatter(coor[0], coor[1], c=[i/N for i in range(0,N)], cmap=plt.cm.get_cmap('cool'))
        if link:
            plt.plot(coor[0][::val], coor[1][::val], c='g', marker='+', linestyle='solid')
        plt.colorbar()
        plt.show()
    
    def Isomap(self, link=True, val=30):
        N = self.data.shape[2]
        isomap = Isomap(n_neighbors=4, n_components=2, n_jobs=-1)
        coor = isomap.fit_transform(self.data.reshape((-1,N)).T).T
        plt.figure(figsize=(12,10))
        plt.scatter(coor[0], coor[1], c=[1.*i/N for i in range(0,N)], cmap=plt.cm.get_cmap('cool'))
        if link:
            plt.plot(coor[0][::val], coor[1][::val], c='g', marker='+', linestyle='solid')
        plt.colorbar()
        plt.show()
    
    def tucker(self, coreNway, lam):
        '''
        tucker factorization
        '''
        # 随机初始化
        A0,C0 = self.__tucker_initial(coreNway)
        mlab = Matlab()
        mlab.start()
        res = mlab.run_func('tucker.m', self.data, coreNway, {'hosvd':1, 'lam':lam, 'maxit':5000, 'A0':A0, 'C0':C0, 'tol':1e-5})
        mlab.stop()
        O,D,T,C = res['result']
        return TuckerResult(self.data, O, D, T, C)
    
    def tucker_with_col_sum_constraint(self, coreNway, lam):
        # 加入行和约束的tucker分解
        R = self.data
        def obj_func_for_C(C):
            R_hat = np.einsum('ijk,pi,qj,rk->pqr', C, O, D, T)
            return np.sum((R-R_hat)**2)
        
        def obj_func_for_O(O):
            R_hat = np.einsum('ijk,pi,qj,rk->pqr', C, O, D, T)
            return np.sum((R-R_hat)**2)
        
        def obj_func_for_D(D):
            R_hat = np.einsum('ijk,pi,qj,rk->pqr', C, O, D, T)
            return np.sum((R-R_hat)**2)
        
        def obj_func_for_T(T):
            R_hat = np.einsum('ijk,pi,qj,rk->pqr', C, O, D, T)
            return np.sum((R-R_hat)**2)
        
        
        
    def nmf_with_square_loss(self, r, max_iter=8000, tol=0.0001):
        V = self.data.reshape((-1,self.data.shape[-1]))
        W,H = self.__nmf_initial(r)
        last_loss = self.__nmf_square_loss(W, H)
        for i in range(max_iter):
            if i>2000 and i%300==0:
                loss = self.__nmf_square_loss(W, H)
                if last_loss - loss <= tol:
                    break
                else:
                    last_loss = loss
            W,H = W * np.matmul(V,H.T) / np.maximum(np.matmul(np.matmul(W,H),H.T), 1e-3),\
            H * np.matmul(W.T,V)/ np.maximum(np.matmul(np.matmul(W.T,W),H), 1e-3)
        return NMFResult(V, W, H)
    
    def nmf_with_KL_loss(self, r, max_iter=8000, tol=0.0001):
        V = self.data.reshape((-1,self.data.shape[-1]))
        W,H = self.__nmf_initial(r)
        last_loss = self.__nmf_KL_loss(W, H)
        for i in range(max_iter):
            if i>2000 and i%300==0:
                loss = self.__nmf_KL_loss(W, H)
                if last_loss - loss <= tol:
                    break
                else:
                    last_loss = loss
            tmp = V / np.maximum(np.matmul(W,H), 1e-3)
            W,H = W * np.matmul(tmp, col_norm(H.T)), H * np.matmul(col_norm(W).T, tmp)
        return NMFResult(V, W, H)
    
    def __tucker_initial(self, coreNway):
        M,_,N = self.data.shape
        I,J,K = coreNway
        # 随机初始化
        _max = np.max(self.data)
        _max = math.pow(_max, 0.25) * 6
        A0 = [
            np.random.uniform(0, _max/math.sqrt(I), (M,I)), 
            np.random.uniform(0, _max/math.sqrt(J), (M,J)), 
            np.random.uniform(0, _max/math.sqrt(K), (N,K))
        ]
        C0 = np.random.uniform(0, _max/math.sqrt(I*J*K), (I,J,K))
        return A0,C0
    
    def __nmf_initial(self, r):
        V = self.data.reshape((-1,self.data.shape[-1]))
        m,n = V.shape
        _max = math.sqrt(np.max(V)/r) * 6
        W0 = np.random.uniform(0, _max, (m,r))
        H0 = np.random.uniform(0, _max, (r,n))
        return W0,H0
    
    def __tucker_loss(self, O, D, T, C, lam):
        R = self.data
        R_hat = np.einsum('ijk,pi,qj,rk->pqr', C, O, D, T)
        return np.sum((R-R_hat)**2) + lam[0]*np.sum(O) + lam[1]*np.sum(D) + \
            lam[2]*np.sum(T) + lam[3]*np.sum(C)
    
    def __nmf_square_loss(self, W, H):
        V = self.data.reshape((-1,self.data.shape[-1]))
        return np.sum((V-np.matmul(W,H))**2)
    
    def __nmf_KL_loss(self, W, H):
        V = self.data.reshape((-1,self.data.shape[-1]))
        V_hat = np.matmul(W,H)
        return np.sum(V*np.log(V/np.maximum(V_hat, 1e-3) + 1e-3) - V + V_hat)


class TuckerResult(object):
    def __init__(self, R, O, D, T, C):
        '''
        R: 被分解张量
        O,D,T: 模式矩阵
        C: 核张量
        '''
        self.R = R
        self.O = O
        self.D = D
        self.T = T
        self.C = C
    
    @property
    def R_hat(self):
        return np.einsum('ijk,pi,qj,rk->pqr', self.C, self.O, self.D, self.T)
    
    @property
    def RMSE(self):
        return np.sqrt(np.mean((self.R - self.R_hat)**2))
    
    @property
    def RRMSE(self):
        return self.RMSE / np.mean(self.R)
    
    @property
    def sparseness(self):
        return _SP(self.O), _SP(self.D), _SP(self.T), _SP(self.C)
    
    def save(self, path):
        pass


class NMFResult(object):
    def __init__(self, V, W, H):
        self.V = V
        self.W = W
        self.H = H
    
    @property
    def RMSE(self):
        V_hat = np.matmul(self.W, self.H)
        return np.sqrt(np.mean((self.V-V_hat)**2))
    
    @property
    def RRMSE(self):
        return self.RMSE / np.mean(self.V)
    
    @property
    def sparseness(self):
        return _SP(self.W), _SP(self.H)
