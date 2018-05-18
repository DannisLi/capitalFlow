#-*- coding:utf8 -*-

import math
import pickle
import numpy as np
import pandas as pd
from pymatbridge import Matlab
import scipy.optimize as sciopt
from sklearn.manifold import MDS, Isomap


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

month_days = np.array([21, 16, 21, 21, 20, 20, 23, 21, 21, 18, 20, 23, 20, 15, 22, 21, 20, 
                       21, 23, 21, 20, 17, 21, 23, 20, 16, 23, 20, 21, 20, 21, 23, 20, 16, 
                       22, 22])


class Tensor(object):
    # 目前仅限三阶张量，且前两阶的维数相同
    def __init__(self, R):
        self.R = R
        self.M,_,self.N = R.shape
        assert self.M == _
    
    def tucker(self, coreNway, lam, init=None):
        # 初始化
        if init is None:
            A0,C0 = self.tucker_init(*coreNway)
        else:
            assert len(init) == 4
            A0 = init[:3]
            C0 = init[-1]
        mlab = Matlab()
        mlab.start()
        while True:
            res = mlab.run_func('tucker.m', self.R, coreNway, 
                                {'hosvd':1, 'lam':lam, 'maxit':4000, 'A0':A0, 'C0':C0, 'tol':1e-5})
            if res['success']:
                break
        mlab.stop()
        O,D,T,C = res['result']
        return TuckerResult(self.R, O, D, T, C)
    
    def tucker_init(self, I, J, K):
        _max = np.max(self.R)
        _max = math.pow(_max, 0.25) * 6
        A0 = [
            np.random.uniform(0, _max/math.sqrt(I), (self.M,I)), 
            np.random.uniform(0, _max/math.sqrt(J), (self.M,J)), 
            np.random.uniform(0, _max/math.sqrt(K), (self.N,K))
        ]
        C0 = np.random.uniform(0, _max/math.sqrt(I*J*K), (I,J,K))
        return A0,C0


def _SP(mat):
    '''
    : 稀疏性度量方法
    '''
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
            'log'         取对数 (以e为底)
            'log10'       取对数 (以10为底)
            'sqrt'        平方根
            'symmetric'   对称元消除 
            'mean'        除以该月的交易日天数(仅用于时间维度以月为单位时)
        '''
        # 读取数据
        with open(datapath, 'rb') as f:
            self.data = pickle.load(f)
        self.M,_,self.N = self.data.shape
        # 预处理
        for pro in preprocess:
            if pro == 'log':
                self.data = np.log(self.data + np.ones(self.data.shape))
            elif pro == 'log10':
                self.data = np.log10(self.data + np.ones(self.data.shape))
            elif pro == 'sqrt':
                self.data = np.sqrt(self.data)
            elif pro == 'symmetric':
                for k in range(self.N):
                    for i in range(self.M):
                        for j in range(i+1,self.M):
                            if self.data[i,j,k] > self.data[j,i,k]:
                                self.data[i,j,k] -= self.data[j,i,k]
                                self.data[j,i,k] = 0
                            else:
                                self.data[j,i,k] -= self.data[i,j,k]
                                self.data[i,j,k] = 0
            elif pro == 'mean':
                for k in range(self.N):
                    self.data[:,:,k] /= month_days[k]
    
    def MDS(self, link=True, val=30):
        import matplotlib.pyplot as plt
        dis = np.zeros((self.N,self.N))
        for i in range(self.N):
            for j in range(i+1,self.N):
                # 使用L1范数引导的距离
                dis[i,j] = np.sum(np.abs(self.data[:,:,i] - self.data[:,:,j]))
                dis[j,i] = dis[i,j]
        mds = MDS(n_components=2, dissimilarity='precomputed', n_jobs=-1)
        coor = mds.fit_transform(dis).T
        plt.figure(figsize=(12,10))
        plt.scatter(coor[0], coor[1], c=[i/self.N for i in range(0,self.N)], cmap=plt.cm.get_cmap('cool'))
        if link:
            plt.plot(coor[0][::val], coor[1][::val], c='g', marker='+', linestyle='solid')
        plt.colorbar()
        plt.show()
    
    def Isomap(self, link=True, val=30):
        import matplotlib.pyplot as plt
        isomap = Isomap(n_neighbors=4, n_components=2, n_jobs=-1)
        coor = isomap.fit_transform(self.data.reshape((-1,self.N)).T).T
        plt.figure(figsize=(12,10))
        plt.scatter(coor[0], coor[1], c=[i/self.N for i in range(0,self.N)], cmap=plt.cm.get_cmap('cool'))
        if link:
            plt.plot(coor[0][::val], coor[1][::val], c='g', marker='+', linestyle='solid')
        plt.colorbar()
        plt.show()
    
    def tucker(self, coreNway, lam):
        R = Tensor(self.data)
        return R.tucker(coreNway, lam)
    
    def series_tucker(self, coreNway, lam, interval=12):
        '''
        : 资金流动按interval分段，分别进行tucker分解。每段使用上一段的结果初始化。
        : 要求：interval必须能够被时间维度除尽
        '''
        assert self.N % interval == 0
        n = int(self.N / interval)
        tensor_list = [Tensor(self.data[:,:,i*interval:(i+1)*interval]) for i in range(n)]
        result_list = []
        init = None
        for R in tensor_list:
            res = R.tucker(coreNway, lam, init)
            init = [res.O, res.D, res.T, res.C]
            result_list.append(res)
        return result_list
    
    def tucker_with_col_sum_constraint(self, coreNway, lam):
        # 带分解张量和尺寸
        R = self.data
        M,_,N = R.shape
        # 核张量的尺寸
        I,J,K = coreNway
        # 初始化
        O = np.random.uniform(size=(M,I))
        D = np.random.uniform(size=(M,J))
        for i in range(M):
            O[i] /= np.sum(O[i])
            D[i] /= np.sum(D[i])
        T = np.random.uniform(size=(N,K))
        for i in range(N):
            T[i] /= np.sum(T[i])
        C = np.random.uniform(0, np.max(R), size=(I,J,K))
        # 各自的优化函数
        def obj_func_for_C(_C):
            _C = _C.reshape((I,J,K))
            R_hat = np.einsum('ijk,pi,qj,rk->pqr', _C, O, D, T)
            return np.sum((R-R_hat)**2) + 0.1*np.sum((_C-C)**2)
        def obj_func_for_O(_O):
            _O = _O.reshape((M,I))
            R_hat = np.einsum('ijk,pi,qj,rk->pqr', C, _O, D, T)
            print (np.sum((R-R_hat)**2) + 0.1*np.sum((_O-O)**2))
            return np.sum((R-R_hat)**2) + 0.1*np.sum((_O-O)**2)
        def obj_func_for_D(_D):
            _D = _D.reshape((M,J))
            R_hat = np.einsum('ijk,pi,qj,rk->pqr', C, O, _D, T)
            return np.sum((R-R_hat)**2) + 0.1*np.sum((_D-D)**2)
        def obj_func_for_T(_T):
            _T = _T.reshape((N,K))
            R_hat = np.einsum('ijk,pi,qj,rk->pqr', C, O, D, _T)
            return np.sum((R-R_hat)**2) + 0.1*np.sum((_T-T)**2)
        
        for i in range(400):
            # optimize matrix O
            O = sciopt.minimize(obj_func_for_O, O.reshape((-1,)), method='SLSQP', bounds=[(0,None)]*(M*I), tol=1e-3,
                                constraints={'type': 'eq', 'fun': lambda _O: np.matmul(_O.reshape((M,I)), np.ones(I))-np.ones(M)}).x.reshape((M,I))
            # optimize matrix D
            D = sciopt.minimize(obj_func_for_D, D.reshape((-1,)), method='SLSQP', bounds=[(0,None)]*(M*J), tol=1e-3,
                                constraints={'type': 'eq', 'fun': lambda _D: np.matmul(_D.reshape((M,J)), np.ones(J))-np.ones(M)}).x.reshape((M,J))
            # optimize matrix T
            T = sciopt.minimize(obj_func_for_T, T.reshape((-1,)), method='SLSQP', bounds=[(0,None)]*(N*K), tol=1e-3,
                                constraints={'type': 'eq', 'fun': lambda _T: np.matmul(_T.reshape((N,K)), np.ones(K))-np.ones(N)}).x.reshape((N,K))
            # optimize matrix C
            C = sciopt.minimize(obj_func_for_C, C.reshape((-1,)), bounds=[(0,None)]*(I*J*K), tol=1e-4).x.reshape((I,J,K))
            print (i)
        return TuckerResult(self.data, O, D, T, C)
    
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
    
    def energy_norm(self):
        # 更改读写模式
        self.O.flags.writeable = True
        self.D.flags.writeable = True
        self.T.flags.writeable = True
        self.C.flags.writeable = True
        # 模式数
        I,J,K = self.C.shape
        # 原始矩阵尺寸
        M,_,N = self.R.shape
        # O
        for i in range(I):
            mask = np.zeros((M,I))
            mask[:,i] += np.ones((M,))
            u = np.mean(np.einsum('ijk,pi,qj,rk->pqr', self.C, self.O*mask, self.D, self.T))/np.sum(self.O[:,i])
            self.O[:,i] = self.O[:,i] * u
            self.C[i,:,:] = self.C[i,:,:] / u
        # D
        for i in range(J):
            mask = np.zeros((M,J))
            mask[:,i] += np.ones((M,))
            u = np.mean(np.einsum('ijk,pi,qj,rk->pqr', self.C, self.O, self.D*mask, self.T))/np.sum(self.D[:,i])
            self.D[:,i] = self.D[:,i] * u
            self.C[:,i,:] = self.C[:,i,:] / u
        # T
        for i in range(K):
            mask = np.zeros((N,K))
            mask[:,i] += np.ones((N,))
            u = np.mean(np.einsum('ijk,pi,qj,rk->pqr', self.C, self.O, self.D, self.T*mask))/np.sum(self.T[:,i])
            self.T[:,i] = self.T[:,i] * u
            self.C[:,:,i] = self.C[:,:,i] / u
        # 恢复写入保护
        self.O.flags.writeable = False
        self.D.flags.writeable = False
        self.T.flags.writeable = False
        self.C.flags.writeable = False
    
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
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    def O_analysis(self, how='row', k=5):
        M,I = self.O.shape
        ptn2items = [[] for i in range(I)]
        if how=='row':
            # 按行确定模式
            item2ptn = np.argmax(self.O, axis=1)
            for i in range(M):
                ptn2items[item2ptn[i]].append(item2name[items[i]])
        elif how=='col':
            # 按列确定模式
            for i in range(I):
                s = pd.Series(self.O[:,i], index=items)
                s.sort_values(ascending=False, inplace=True)
                ptn2items[i] = [item2name[x] for x in s.index.tolist()[:k]]
        return ptn2items
    
    def D_analysis(self, how='row', k=5):
        M,J = self.D.shape
        ptn2items = [[] for i in range(J)]
        if how=='row':
            # 按行确定模式
            item2ptn = np.argmax(self.D, axis=1)
            for i in range(M):
                ptn2items[item2ptn[i]].append(item2name[items[i]])
        elif how=='col':
            # 按列确定模式
            for i in range(J):
                s = pd.Series(self.D[:,i], index=items)
                s.sort_values(ascending=False, inplace=True)
                ptn2items[i] = [item2name[x] for x in s.index.tolist()[:k]]
        return ptn2items
        
    
    def T_analysis(self):
        # 分析T矩阵
        K = self.T.shape[1]    # 时间模式数目
        import matplotlib.pyplot as plt
        plt.figure(figsize=(18,8))
        for k in range(K):
            plt.plot(self.T[:,k], label='pattern %d' % k)
        plt.legend()
        plt.show()


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
