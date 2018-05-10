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

conn = pymysql.connect(**db_config)

with conn.cursor() as cursor:
    cursor.execute("select day from market.trade_day where day between \
    '20140101' and '20161231' order by day asc")
    day_list = [x for x, in cursor.fetchall()][1:]

whole_day = np.zeros((vari_num+1, vari_num+1, 732))
whole_day_mean = np.zeros((vari_num+1, vari_num+1, 732))
whole_month = np.zeros((vari_num+1, vari_num+1, 36))

with conn.cursor() as cursor:
    i = 0
    for day in day_list:
        tmp = np.zeros((vari_num+1, vari_num+1))
        n = cursor.execute("select cf from only_margin where date=%s", (day,))
        for cf, in cursor.fetchall():
            cf = pickle.loads(cf)
            tmp += cf
        whole_day_mean[:,:,i] += tmp / n
        whole_day[:,:,i] += tmp
        whole_month[:,:,(day.year-2014)*12+day.month-1] += tmp
        i += 1

conn.close()

with open('whole_day.data','wb') as f:
    pickle.dump(whole_day, f)

with open('whole_day_mean.data','wb') as f:
    pickle.dump(whole_day_mean, f)

with open('whole_month.data','wb') as f:
    pickle.dump(whole_month, f)

print ('finish!')
