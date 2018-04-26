#-*- coding:utf8 -*-
import numpy as np
import scipy.io as scio

def get_CF(time='month', population='whole', mode='M', log=True):
    '''
    time: 资金流动的时间尺度，有两种取值：'month','day'
    pupulation: 资金流动的人群范围，有五种取值：'whole','institution','individual','monentum','reverse'
    mode: 资金流动的表达形式（矩阵 or 三阶张量），要两种取值：'M','T'
    log: 是否对资金流动取对数
    '''
    fname = '/home/lizimeng/workspace/capitalFlow/data/%s_%s_%s.mat' % (mode, population, time)
    CF = scio.loadmat(fname)['M']
    if log:
        CF = np.log(CF + np.ones(CF.shape))
    return CF
