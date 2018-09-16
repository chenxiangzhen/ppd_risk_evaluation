import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import arrow
import seaborn as sns
sns.set(style='whitegrid')


def parse_date(date_str, str_format='YYYY/MM/DD'):
    '''
    日期格式
    :param date_str:
    :param str_format:
    :return:
    '''
    d = arrow.get(date_str, str_format)
    # 月初，月中，月末
    month_stage = int((d.day-1) / 10) + 1
    return (d.timestamp, d.year, d.month, d.day, d.week, d.isoweekday(), month_stage)


def show_cols(df):
    '''
    输出各列
    :param df:
    :return:
    '''
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
null_sum = train_master.isnull().sum()  # 找出各列为空的计数
null_sum = null_sum[null_sum!=0]  # 找出有缺失的列
null_sum_df = DataFrame(null_sum, columns=['num'])  # 转成DF
null_sum_df['ratio'] = null_sum_df['num'] / 30000.0  # 新增一列ratio
null_sum_df.sort_values(by='ratio', ascending=False, inplace=True)  # 找出确实严重的列，按缺失率排序
print(null_sum_df[null_sum_df['ratio'] > 0.2].index[0])
print(null_sum_df[null_sum_df['ratio'] > 0.2].index[1])

# 删除缺失超过20%的列
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

# Feature分类
ratio_threshold = 0.5
binarized_features = []
binarized_features_most_freq_value = []

for f in train_master.columns:
    if f in ['target']:
        continue

    not_null_sum = (train_master[f].notnull()).sum()
    most_count = pd.value_counts(train_master[f], ascending=False).iloc[0]
    most_value = pd.value_counts(train_master[f], ascending=False).index[0]
    ratio = most_count / not_null_sum

    if ratio > ratio_threshold:
        binarized_features.append(f)
        binarized_features_most_freq_value.append(most_value)

numerical_features = [f for f in train_master.select_dtypes(exclude=['object']).columns
                      if f not in (['Idx', 'target'])
                      and f not in binarized_features]

categorical_features = [f for f in train_master.select_dtypes(include=["object"]).columns
                        if f not in (['Idx', 'target'])
                        and f not in binarized_features]

for i in range(len(binarized_features)):
    f = binarized_features[i]
    most_value = binarized_features_most_freq_value[i]
    train_master['b_' + f] = 1
    train_master.loc[train_master[f] == most_value, 'b_' + f] = 0
    train_master.drop([f], axis=1, inplace=True)

feature_unique_count = []
for f in numerical_features:
    feature_unique_count.append((np.count_nonzero(train_master[f].unique()), f))

# print(sorted(feature_unique_count))

for c, f in feature_unique_count:
    if c <= 10:
        print('{} moved from numerical to categorical'.format(f))
        numerical_features.remove(f)
        categorical_features.append(f)

# 特征工程
melt = pd.melt(train_master, id_vars=['target'], value_vars=[f for f in numerical_features])
g = sns.FacetGrid(data=melt, col="variable", col_wrap=4, sharex=False, sharey=False)
g.map(sns.stripplot, 'target', 'value', jitter=True, palette="muted")

# hard work, but helps a lot

print('{} lines before drop'.format(train_master.shape[0]))

train_master.drop(train_master[(train_master.ThirdParty_Info_Period6_1 > 250) & (train_master.target == 1)].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period6_2 > 400].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period6_2 > 250) & (train_master.target == 1)].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period6_3 > 2000].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period6_3 > 1250) & (train_master.target == 1)].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period6_4 > 1500].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period6_4 > 1250) & (train_master.target == 1)].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period6_5 > 400)].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period6_7 > 2000)].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period6_6 > 1500)].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period6_5 > 1000) & (train_master.target == 0)].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period6_8 > 1500)].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period6_8 > 1000) & (train_master.target == 1)].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period6_16 > 2000000) & (train_master.target == 0)].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period6_14 > 1000000) & (train_master.target == 0)].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period6_12 > 60)].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period6_11 > 120) & (train_master.target == 0)].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period6_11 > 20) & (train_master.target == 1)].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period6_13 > 200000)].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period6_13 > 150000) & (train_master.target == 1)].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period6_15 > 40000) & (train_master.target == 1)].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period6_17 > 130000) & (train_master.target == 0)].index, inplace=True)


train_master.drop(train_master[train_master.ThirdParty_Info_Period5_1 > 500].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period5_2 > 500].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period5_3 > 3000) & (train_master.target == 0)].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period5_3 > 2000)].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period5_5 > 500].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period5_4 > 2000) & (train_master.target == 1)].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period5_6 > 700].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period5_6 > 300) & (train_master.target == 1)].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period5_7 > 4000)].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period5_8 > 800)].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period5_11 > 200)].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period5_13 > 200000].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period5_14 > 150000].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period5_15 > 75000].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period5_16 > 180000].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period5_17 > 150000].index, inplace=True)

# go above

train_master.drop(train_master[(train_master.ThirdParty_Info_Period4_1 > 400)].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period4_2 > 350)].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period4_3 > 1500)].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period4_4 > 1600].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period4_4 > 1250) & (train_master.target == 1)].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period4_5 > 500].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period4_6 > 800].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period4_6 > 400) & (train_master.target == 1)].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period4_8 > 1000].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period4_13 > 250000].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period4_14 > 200000].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period4_15 > 70000].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period4_16 > 210000].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period4_17 > 160000].index, inplace=True)


train_master.drop(train_master[train_master.ThirdParty_Info_Period3_1 > 400].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period3_2 > 380].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period3_3 > 1750].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period3_4 > 1750].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period3_4 > 1250) & (train_master.target == 1)].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period3_5 > 600].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period3_6 > 800].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period3_6 > 400) & (train_master.target == 1)].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period3_7 > 1600) & (train_master.target == 1)].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period3_8 > 1000].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period3_13 > 300000].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period3_14 > 200000].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period3_15 > 80000].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period3_16 > 300000].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period3_17 > 150000].index, inplace=True)


train_master.drop(train_master[train_master.ThirdParty_Info_Period2_1 > 400].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period2_1 > 300) & (train_master.target == 1)].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period2_2 > 400].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period2_2 > 300) & (train_master.target == 1)].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period2_3 > 1800].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period2_3 > 1500) & (train_master.target == 1)].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period2_4 > 1500].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period2_5 > 580].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period2_6 > 800].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period2_6 > 400) & (train_master.target == 1)].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period2_7 > 2100].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period2_8 > 700) & (train_master.target == 1)].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period2_11 > 120].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period2_13 > 300000].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period2_14 > 170000].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period2_15 > 80000].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period2_15 > 50000) & (train_master.target == 1)].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period2_16 > 300000].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period2_17 > 150000].index, inplace=True)


train_master.drop(train_master[train_master.ThirdParty_Info_Period1_1 > 350].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period1_1 > 200) & (train_master.target == 1)].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period1_2 > 300].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period1_2 > 190) & (train_master.target == 1)].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period1_3 > 1500].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period1_4 > 1250].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period1_5 > 400].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period1_6 > 500].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period1_6 > 250) & (train_master.target == 1)].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period1_7 > 1800].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period1_8 > 720].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period1_8 > 600) & (train_master.target == 1)].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period1_11 > 100].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period1_13 > 200000].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period1_13 > 140000) & (train_master.target == 1)].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period1_14 > 150000].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period1_15 > 70000].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period1_15 > 30000) & (train_master.target == 1)].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period1_16 > 200000].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period1_16 > 100000) & (train_master.target == 1)].index, inplace=True)
train_master.drop(train_master[train_master.ThirdParty_Info_Period1_17 > 100000].index, inplace=True)
train_master.drop(train_master[(train_master.ThirdParty_Info_Period1_17 > 80000) & (train_master.target == 1)].index, inplace=True)

train_master.drop(train_master[train_master.WeblogInfo_4 > 40].index, inplace=True)
train_master.drop(train_master[train_master.WeblogInfo_6 > 40].index, inplace=True)
train_master.drop(train_master[train_master.WeblogInfo_7 > 150].index, inplace=True)
train_master.drop(train_master[train_master.WeblogInfo_16 > 50].index, inplace=True)
train_master.drop(train_master[(train_master.WeblogInfo_16 > 25) & (train_master.target == 1)].index, inplace=True)
train_master.drop(train_master[train_master.WeblogInfo_17 > 100].index, inplace=True)
train_master.drop(train_master[(train_master.WeblogInfo_17 > 80) & (train_master.target == 1)].index, inplace=True)
train_master.drop(train_master[train_master.UserInfo_18 < 10].index, inplace=True)

print('{} lines after drop'.format(train_master.shape[0]))