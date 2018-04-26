#!/usr/bin/env python
#-*- coding:utf8 -*-

import datetime, math, pandas as pd, numpy as np
from db import DB
from tradingday import Tradingday

# 2014年到2016年的所有品种
vari_list = [
    'a', 'ag', 'al', 'ap', 'au', 'b', 'bb', 'bu', 'c', 'cf', 'cs', 'cu', 
    'cy', 'er', 'fb', 'fg', 'fu', 'hc', 'i', 'j', 'jd', 'jm', 'jr', 'l', 
    'lr', 'm', 'ma', 'me', 'ni', 'oi', 'p', 'pb', 'pm', 'pp', 'rb', 'ri', 
    'rm', 'ro', 'rs', 'ru', 'sf', 'sm', 'sn', 'sr', 'ta', 'tc', 'v', 'wh', 
    'wr', 'ws', 'y', 'zc', 'zn'
]

vari_num = 53

class CapitalFlow(object):
    @classmethod
    def mat(cls, account, day):
        '''
        返回account账户在day日的资金流动矩阵
        '''
        # 链接数据库
        db = DB("investor")
        # 上个交易日日期
        lastday = Tradingday.lastday(day)
        # 初始化起始保证金、终止保证金、持仓收益、平仓收益
        start_margin = [0.]*vari_num
        end_margin = [0.]*vari_num
        profit = [0.]*vari_num
        # 查询end_margin、hold_profit
        results = db.execute_sql("select vari,sum(margin),sum(hold_profit_d) from chicang \
        where account=%s and tradedate=%s group by vari", (account, day))
        for row in results:
            try:
                i = vari_list.index(row[0])
            except:
                continue
            end_margin[i] += row[1]
            profit[i] += row[2]
        # 查询start_margin
        results = db.execute_sql(
        "select vari,sum(margin) from chicang where account=%s and tradedate=%s group by vari",
        (account, lastday))
        for row in results:
            try:
                i = vari_list.index(row[0])
            except:
                continue
            start_margin[i] += row[1]
        # 查询drop_profit
        results = db.execute_sql("select vari,sum(drop_profit_d) from pingcang where account=%s and tradedate=%s group by vari",
        (account, day))
        for row in results:
            try:
                i = vari_list.index(row[0])
            except:
                continue
            profit[i] += row[1]
        # 关闭数据库
        db.close()
        # 整理数据
        df = pd.DataFrame({"start_margin":start_margin, "end_margin":end_margin, 
        "profit":profit}, index=vari_list)
        # 计算保证金变化和收益
        change = (df.start_margin-df.end_margin).tolist() + df.profit.tolist()
        # 区分正负
        pos = []
        neg = []
        a = 0
        b = 0
        for i in range(len(change)):
            if change[i]>0:
                a += change[i]
                pos.append(i)
            elif change[i]<0:
                b += change[i]
                neg.append(i)
        b = abs(b)
        change = [abs(x) for x in change]
        # 初始化转移矩阵
        mat = np.zeros((2*vari_num+1,2*vari_num+1), dtype=np.float32)
        # 比较正负流动总额
        if a==b==0:
            pass
        elif a>=b:
            # 计算流出各部分的比例
            ratio = [change[i]/a for i in pos]
            # 将a-b补偿到账户上
            t = a-b
            for i in range(len(pos)):
                mat[pos[i],2*vari_num] += t * ratio[i]
            # 各保证金和收益相互补偿
            for i in range(len(neg)):
                for j in range(len(pos)):
                    mat[pos[j],neg[i]] += change[neg[i]] * ratio[j]
        else:
            # 计算流入各部分的比例
            ratio = [change[i]/b for i in neg]
            # 将b-a补偿到各流入上
            t = b-a
            for i in range(len(neg)):
                mat[2*vari_num,neg[i]] += t * ratio[i]
            # 各保证及和收益相互补偿
            for i in range(len(pos)):
                for j in range(len(neg)):
                    mat[pos[i],neg[j]] += change[pos[i]] * ratio[j]
        # 返回结果
        return mat
    
    @classmethod
    def triple(cls, mat):
        '''
        转移矩阵转换为三元组
        '''
        triple = []
        for i in range(2*vari_num+1):
            for j in range(2*vari_num+1):
                if mat[i,j]!=0:
                    if i < vari_num:
                        x = vari_list[i] + 'm'
                    elif vari_num <= i < 2*vari_num:
                        x = vari_list[i-vari_num] + 'p'
                    else:
                        x = 'fund'
                    if j < vari_num:
                        y = vari_list[j] + 'm'
                    elif vari_num <= j < 2*vari_num:
                        y = vari_list[j-vari_num] + 'p'
                    else:
                        y = 'fund'
                    triple.append([x,y,int(round(mat[i,j]))])
        return triple

if __name__=='__main__':
    d1 = datetime.date(2015,5,25)
    mat = CapitalFlow.mat("488854682519", d1)
    triple = CapitalFlow.triple(mat)
    print (triple)
