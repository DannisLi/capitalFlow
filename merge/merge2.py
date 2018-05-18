#-*- coding:utf8 -*-

import pickle
import numpy as np
import pymysql
from multiprocessing import Pool

db_config = {
    'host' : '219.224.169.45',
    'user' : 'lizimeng',
    'password' : 'codegeass',
    'charset' : 'utf8',
    'db' : 'capital_flow'
}

vari_list = [
    'a', 'ag', 'al', 'au', 'b', 'bb', 'bu', 'c', 'cf', 'cu', 'fb', 'fg', 'fu', 'i', 'j', 'jd', 
    'jm', 'jr', 'l', 'm', 'oi', 'p', 'pb', 'pm', 'rb', 'ri', 'rm', 'rs', 'ru', 'sr', 'ta', 'v', 
    'wh', 'wr', 'y', 'zn'
]
vari_num = len(vari_list)


def solve(days):
    # 初始化结果
    result = [np.zeros((vari_num*2+1, vari_num*2+1)) for i in range(len(days))]
    # 链接数据库
    conn = pymysql.connect(**db_config)
    with conn.cursor() as cursor:
        i = 0
        for day in days:
            cursor.execute("select cf from only_margin2 where date=%s", (day,))
            for cf, in cursor.fetchall():
                cf = pickle.loads(cf)
                result[i] += cf
            i += 1
    conn.close()
    return result


conn = pymysql.connect(**db_config)
with conn.cursor() as cursor:
    cursor.execute("select day from market.trade_day where day between \
    '20140101' and '20161231' order by day asc")
    day_list = [x for x, in cursor.fetchall()][1:]

day_num = len(day_list)
val = 100
seg = int(day_num/val) + 1

result = []

pool = Pool(processes=5)
for r in pool.imap(solve, [day_list[i*val:min((i+1)*val,day_num)] for i in range(seg)]):
    result.extend(r)


whole_day = np.zeros((vari_num*2+1, vari_num*2+1, 732))
whole_month = np.zeros((vari_num*2+1, vari_num*2+1, 36))

i = 0
for day in day_list:
    whole_day[:,:,i] += result[i]
    whole_month[:,:,(day.year-2014)*12+day.month-1] += result[i]
    i += 1


with open('whole_day2.data','wb') as f:
    pickle.dump(whole_day, f)

with open('whole_month2.data','wb') as f:
    pickle.dump(whole_month, f)

print ('finish!')
