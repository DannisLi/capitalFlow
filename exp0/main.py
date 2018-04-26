#-*- coding:utf8 -*-

import json, datetime
from db import DB
from multiprocessing import Pool
from capitalFlow import CapitalFlow
from tradingday import Tradingday

# 工作线程总数量
n = 100

def solve(accounts):
    db = DB("investor")
    d = datetime.date(2014,1,3)    # 第二个交易日
    end = datetime.date(2016,12,31)
    step = datetime.timedelta(1)
    try:
        while d<=end:
            if Tradingday.is_tradingday(d):
                for account in accounts:
                    mat = CapitalFlow.mat(account, d)
                    triple = CapitalFlow.triple(mat)
                    if len(triple)>0:
                        db.insert_sql("insert into capital_flow.basic values (%s,%s,%s)", (account,d,json.dumps(triple)))
                db.commit()
            d += step
    except Exception as e:
        print (e)
    finally:
        db.close()
        print ('A process has finished!')


# 查找所有账户
db = DB("investor")
accounts = db.execute_sql("(select distinct account from chicang) union (select distinct account from pingcang)")
db.close()

# 进程池
pool = Pool()

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
