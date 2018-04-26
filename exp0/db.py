#-*- coding:utf8 -*-

import pymysql

class DB(object):
    def __init__(self, db_name=None):
        self.conn = pymysql.connect(
            host = '219.224.169.45',
            user = 'lizimeng',
            password = 'codegeass',
            db = db_name,
            charset = 'utf8'
        )
        self.cursor = self.conn.cursor()
    
    def execute_sql(self, sql, params=()):
        self.cursor.execute(sql, params)
        result = self.cursor.fetchall()
        if len(result)==0:
            return []
        else:
            if len(result[0])==1:
                return [row[0] for row in result]
            else:
                return list(result)
    
    def insert_sql(self, sql, params=()):
        self.cursor.execute(sql, params)
    
    def commit(self):
        self.conn.commit()
    
    def close(self):
        self.cursor.close()
        self.conn.close()
