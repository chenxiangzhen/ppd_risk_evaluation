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
null_sum_df.sort_values(by='ratio', ascending=False, inplace=True)  # 找出确实严重的列

# 删除缺失严重的列
train_master.drop(['WeblogInfo_3', 'WeblogInfo_1', 'UserInfo_11', 'UserInfo_13', 'UserInfo_12', 'WeblogInfo_20'],
                  axis=1, inplace=True)

# 删除缺失严重的行
record_nan = train_master.isnull().sum(axis=1).sort_values(ascending=False)
drop_record_index = [i for i in record_nan.loc[(record_nan>=5)].index]
print('before train_master shape {}'.format(train_master.shape))
train_master.drop(drop_record_index, inplace=True)
print('after train_master shape {}'.format(train_master.shape))
# len(drop_record_index)

# 填补缺失值
print('before all nan num: {}'.format(train_master.isnull().sum().sum()))

train_master.loc[train_master['UserInfo_2'].isnull(), 'UserInfo_2'] = '位置地点'
train_master.loc[train_master['UserInfo_4'].isnull(), 'UserInfo_4'] = '位置地点'

def fill_nan(f, method):
    if method == 'most':
        common_value = pd.value_counts(train_master[f], ascending=False).index[0]
    else:
        common_value = train_master[f].mean()
    train_master.loc[train_master[f].isnull(), f] = common_value

# 通过pd.value_counts(train_master[f])的观察得到经验
fill_nan('UserInfo_1', 'most')
fill_nan('UserInfo_3', 'most')
fill_nan('WeblogInfo_2', 'most')
fill_nan('WeblogInfo_4', 'mean')
fill_nan('WeblogInfo_5', 'mean')
fill_nan('WeblogInfo_6', 'mean')
fill_nan('WeblogInfo_19', 'most')
fill_nan('WeblogInfo_21', 'most')

print('after all nan num: {}'.format(train_master.isnull().sum().sum()))