#-*- coding:utf8 -*-

from db import DB
import pandas as pd

db = DB("market")
days = db.execute_sql("select day from trade_day where day between '20140101' and '20161231' order by day asc")
n = len(days)
db.close()

class Tradingday(object):
    @classmethod
    def is_tradingday(cls, day):
        '''
        判断day是否为交易日
        '''
        if day in days:
            return True
        else:
            return False
    
    @classmethod
    def shift(cls, day, val):
        '''
        day后第val个交易日
        '''
        i = days.index(day)
        assert 0<= i+val < n
        return days[i+val]
        
    @classmethod
    def lastday(cls, day):
        return Tradingday.shift(day, -1)
        
    @classmethod
    def nextday(cls, day):
        return Tradingday.shift(day, 1)
    
    
