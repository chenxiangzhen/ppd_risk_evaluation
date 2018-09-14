import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import arrow
import seaborn as sns
sns.set(style='whitegrid')

def parse_date(date_str, str_format='YYYY/MM/DD'):
    d = arrow.get(date_str, str_format)
    # 月初，月中，月末
    month_stage = int((d.day-1) / 10) + 1
    return (d.timestamp, d.year, d.month, d.day, d.week, d.isoweekday(), month_stage)

def show_cols(df):
    for c in df.columns:
        print(c)

# 加载数据
path = './PPD-First-Round-Data-Update/Training Set'
# 每一行代表一个样本（一笔成功成交借款），每个样本包含200多个各类字段
train_master = pd.read_csv(path + '/PPD_Training_Master_GBK_3_1_Training_Set.csv', encoding='gbk')
# 借款人的登陆信息
train_loginfo = pd.read_csv(path + '/PPD_LogInfo_3_1_Training_Set.csv', encoding='gbk')
# 借款人修改信息
train_userinfo = pd.read_csv(path + '/PPD_Userupdate_Info_3_1_Training_Set.csv', encoding='gbk')

# 数据清洗
null_sum = train_master.isnull().sum()
null_sum = null_sum[null_sum!=0]
null_sum_df = DataFrame(null_sum, columns=['num'])
null_sum_df['ratio'] = null_sum_df['num'] / 30000.0
null_sum_df.sort_values(by='ratio', ascending=False, inplace=True)

# 删除缺失严重的列
train_master.drop(['WeblogInfo_3', 'WeblogInfo_1', 'UserInfo_11', 'UserInfo_13', 'UserInfo_12', 'WeblogInfo_20'],
                  axis=1, inplace=True)