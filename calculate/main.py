#-*- coding:utf8 -*-

import datetime
import pymysql
import pickle
import numpy as np
from multiprocessing import Pool
from capitalFlow import get_cf_only_margin as get_cf

# 线程总数量
n = 60

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
    cursor.execute("select distinct account from investor.chicang")
    accounts = [x for x, in cursor.fetchall()]
conn.close()


def solve(accounts):
    conn = pymysql.connect(
        host = '219.224.169.45',
        user = 'lizimeng',
        password = 'codegeass',
        db = 'investor',
        charset = 'utf8'
    )
    with conn.cursor() as cursor:
        for account in accounts:
            cursor.execute("select distinct tradedate from chicang where account=%s order by tradedate asc",
                           (account,))
            for day, in cursor.fetchall():
                if day == datetime.date(2014,1,2):
                    # 这一天找不到前日持仓
                    continue
                try:
                    cf = get_cf(account, day)
                except:
                    print ('get_cf() error!')
                if np.sum(cf)==0:
                    continue
                # cf中所有元素都应大于等于0
                try:
                    assert np.sum(cf>=0) == cf.size
                except:
                    print ('cf < 0 !')
                    continue
                cursor.execute("insert into capital_flow.only_margin values (%s,%s,%s)",
                               (account, day, pickle.dumps(cf)))
            conn.commit()
    conn.close()
    print ('finish!')



# 进程池
pool = Pool(20)

# 每个进程处理的用户数量
seg = int(len(accounts) / n)

# 开启进程
for i in range(n):
    pool.apply_async(solve, args=(accounts[i*seg:(i+1)*seg],))
if n*seg<len(accounts):
    pool.apply_async(solve, args=(accounts[n*seg:],))

# 关闭进程池
pool.close()

# 等待所有进程结束
pool.join()

# 输出结束提示信息
print ("finish!")
