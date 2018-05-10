#-*- coding:utf8 -*-
import pickle
import datetime
import pymysql
import numpy as np

# 2014年到2016年的一直在市场上交易的品种
vari_list = [
    'a', 'ag', 'al', 'au', 'b', 'bb', 'bu', 'c', 'cf', 'cu', 'fb', 'fg', 'fu', 'i', 'j', 'jd', 
    'jm', 'jr', 'l', 'm', 'oi', 'p', 'pb', 'pm', 'rb', 'ri', 'rm', 'rs', 'ru', 'sr', 'ta', 'v', 
    'wh', 'wr', 'y', 'zn'
]
vari_num = len(vari_list)

conn = pymysql.connect(
    host = '219.224.169.45',
    user = 'lizimeng',
    password = 'codegeass',
    db = 'market',
    charset = 'utf8'
)
with conn.cursor() as cursor:
    cursor.execute("select day from trade_day where day between '20140101' and '20161231' order by day asc")
    trading_day_list = [x for x, in cursor.fetchall()]


# 仅考虑保证金，不考虑收益
def get_cf_only_margin(account, day):
    # 上一个交易日的日期
    i = trading_day_list.index(day)
    lastday = trading_day_list[i-1]
    
    # 初始化保证金列表
    start_margin = np.zeros(vari_num)
    end_margin = np.zeros(vari_num)
    
    conn = pymysql.connect(
        host = '219.224.169.45',
        user = 'lizimeng',
        password = 'codegeass',
        db = 'investor',
        charset = 'utf8'
    )
    with conn.cursor() as cursor:
        # 查询昨天的保证金
        cursor.execute("select vari,sum(margin) from chicang \
        where account=%s and tradedate=%s group by vari", (account, lastday))
        for row in cursor.fetchall():
            try:
                i = vari_list.index(row[0])
            except:
                continue
            start_margin[i] += row[1]
        # 查询今天的保证金
        cursor.execute("select vari,sum(margin) from chicang \
        where account=%s and tradedate=%s group by vari", (account, day))
        for row in cursor.fetchall():
            try:
                i = vari_list.index(row[0])
            except:
                continue
            end_margin[i] += row[1]
    # 各品种上变化的保证金    
    delta_margin = end_margin - start_margin
    
    # 没有改变的保证金
    fixed_margin = np.minimum(start_margin, end_margin)
    
    # 流入总资金与流出总资金
    in_flow = np.maximum(delta_margin, 0)
    in_all = np.sum(in_flow)
    out_flow = np.abs(np.minimum(delta_margin, 0))
    out_all = np.sum(out_flow)
    
    cf = np.diag(fixed_margin.tolist() + [0])
    if in_all > out_all:
        ratio = in_flow / in_all
        cf[vari_num,:vari_num] += (in_all-out_all) * ratio
        for i in range(vari_num):
            cf[i,:vari_num] += ratio * out_flow[i]
    elif out_all>0:
        ratio = out_flow / out_all
        cf[:vari_num,vari_num] += (out_all-in_all) * ratio
        for i in range(vari_num):
            cf[:vari_num,i] += ratio * in_flow[i]
    
    return cf
    
# 仅考虑保证金，但是将保证金分为持仓保证金和平仓保证金
def get_cf_only_margin2(account, day):
    # 上一个交易日的日期
    i = trading_day_list.index(day)
    lastday = trading_day_list[i-1]
    
    # 初始化保证金列表，前半部分为多头，后半部分为空头
    start_margin = np.zeros(vari_num*2)
    end_margin = np.zeros(vari_num*2)
    
    conn = pymysql.connect(
        host = '219.224.169.45',
        user = 'lizimeng',
        password = 'codegeass',
        db = 'investor',
        charset = 'utf8'
    )
    with conn.cursor() as cursor:
        # 查询昨天多头保证金
        cursor.execute("select vari,sum(margin) from chicang \
        where account=%s and tradedate=%s and bs_flag='0' group by vari", (account, lastday))
        for row in cursor.fetchall():
            try:
                i = vari_list.index(row[0])
                start_margin[i] += row[1]
            except:
                pass
        # 查询昨天空头保证金
        cursor.execute("select vari,sum(margin) from chicang \
        where account=%s and tradedate=%s and bs_flag='1' group by vari", (account, lastday))
        for row in cursor.fetchall():
            try:
                i = vari_list.index(row[0]) + vari_num
                start_margin[i] += row[1]
            except:
                pass
        # 查询今天多头保证金
        cursor.execute("select vari,sum(margin) from chicang \
        where account=%s and tradedate=%s and bs_flag='0' group by vari", (account, day))
        for row in cursor.fetchall():
            try:
                i = vari_list.index(row[0])
                end_margin[i] += row[1]
            except:
                pass
        # 查询今天空头保证金
        cursor.execute("select vari,sum(margin) from chicang \
        where account=%s and tradedate=%s and bs_flag='1' group by vari", (account, day))
        for row in cursor.fetchall():
            try:
                i = vari_list.index(row[0]) + vari_num
                end_margin[i] += row[1]
            except:
                pass
    # 各品种上变化的保证金
    delta_margin = end_margin - start_margin
    
    # 没有改变的保证金
    fixed_margin = np.minimum(start_margin, end_margin)
    
    # 流入总资金与流出总资金
    in_flow = np.maximum(delta_margin, 0)
    in_all = np.sum(in_flow)
    out_flow = np.abs(np.minimum(delta_margin, 0))
    out_all = np.sum(out_flow)
    
    cf = np.diag(fixed_margin.tolist() + [0])
    if in_all > out_all:
        ratio = in_flow / in_all
        cf[vari_num*2,:vari_num*2] += (in_all-out_all) * ratio
        for i in range(vari_num*2):
            cf[i,:vari_num*2] += ratio * out_flow[i]
    elif out_all>0:
        ratio = out_flow / out_all
        cf[:vari_num*2,vari_num*2] += (out_all-in_all) * ratio
        for i in range(vari_num*2):
            cf[:vari_num*2,i] += ratio * in_flow[i]
    
    return cf

if __name__=='__main__':
    cf = get_cf_only_margin2("488854682519", datetime.date(2015,7,8))
    print (np.sum(cf))
    
