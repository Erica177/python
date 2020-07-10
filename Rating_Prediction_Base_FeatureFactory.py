# coding: utf-8
"""
Created on Wed Jul  8 14:40:36 2020

@author: ShuaiLiu
"""
import numpy as np 
import pandas as pd     
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb
#%%
rr_df = pd.read_json('D:\\bisai\\clothes\\data\\renttherunway_final_data.json', lines=True)
mc_df = pd.read_json('D:\\bisai\\clothes\\data\\modcloth_final_data.json', lines=True)
#%%
#重命名数据集 方便合并
mc_df.rename(columns={'quality':'rating',
                      'bra size':'bra_size',
                      'cup size':'cup_size'},inplace=True)


rr_df.rename(columns={'body type':'body_type',
                      'rented for':'rented_for',
                      'bust size':'bra_size'},inplace=True)

print(rr_df.columns)
print(mc_df.columns)
#%%
# 分割 rr_df 中的 bust_size --> 36 c
rr_df['cup_size'] = rr_df['bra_size']

bust = rr_df['bra_size']
cup = []
bra = []
for i in range(len(bust)):
    if isinstance(bust[i], float) or bust[i] == 'NaN' or bust[i] == 'nan':
        cup.append(float('NaN'))
        bra.append(float('NaN'))
        continue
#    print(bust[i][:2])
    bra.append(bust[i][:2])#胸围大小 数字
    cup.append(bust[i][2:])#胸围大小 字母abcdefg

rr_df['cup_size'] = cup
rr_df['bra_size'] = bra

rr_df['bra_size'] = [float(j) for j in rr_df['bra_size']]

print(rr_df.head())
#%%
# 把raiting 变到 1~5
rr_df['rating'] = [rr_df['rating'][i]/2 for i in range(len(rr_df['rating']))]

print(rr_df['rating'].value_counts())
#%%
# 英尺换算为厘米
def get_cms(x):

    if type(x) == type(1.0):
        return
    try: 
        return (int(x[0])*30.48) + (int(x[4:-2])*2.54)
    except:
        return (int(x[0])*30.48)
mc_df['height'] = mc_df['height'].apply(get_cms)
print(mc_df['height'])

def get_cms_r(x):
    if type(x) == type(1.0):
        return
    try: 
        return (int(x[0])*30.48) + (int(x[3])*2.54)
    except:
        return (int(x[0])*30.48)
    
rr_df['height'] = rr_df['height'].apply(get_cms_r)
print(rr_df['height'])

#%%
data = pd.concat([rr_df, mc_df], ignore_index=True)

df_ = pd.DataFrame(columns=['user_id','item_id'])
df_['user_id'] = data['user_id']
df_['item_id'] = data['item_id']

# print(df_)
#%%
missing_data = pd.DataFrame({'total_missing': data.isnull().sum(), 'perce_missing(%)': (data.isnull().sum()/len(data))*100})
print(missing_data)

#%%
#删除缺失率80%的数据
data.drop(['waist'], axis=1, inplace=True)
data.drop(['bust'], axis=1, inplace=True)
data.drop(['shoe size'], axis=1, inplace=True)
data.drop(['shoe width'], axis=1, inplace=True)

data.drop(['user_id','user_name','review_date','review_text', 'review_summary', 'item_id'], axis=1, inplace=True)
#%%
#去掉体重的单位lbs
weight = data['weight']
new_weight = []
for i in range(len(weight)):
    if isinstance(weight[i],float) or weight[i] == 'NaN' or weight[i] == 'nan':
        new_weight.append(float('NaN'))
        continue
    new_weight.append(weight[i][:-3])
data['weight'] = new_weight
#%%
# 通过箱型图删除异常点
print(data.columns)
num_cols = ['bra_size','hips','size', 'age', 'height']
plt.figure(figsize=(15,9))
sns.boxplot(data = data[num_cols])
plt.title("Numerical variables boxplot", fontsize=20)
plt.show()
#%%
# data = data.drop(data[(data.age>10) & (data.age <70)].index)
# data = data.drop(data[(data.height>100) & (data.height <200)].index)
# data = data[data['age']>10 and data['age']<80]
# data = data[data['height']>100 and data['height']<200]
#%%
#画出特征的分布图
def plot_dist(col, ax):   
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
    tmp = data[col][data[col].notnull()].value_counts().reset_index()
    tmp.columns = [col,'counts']
    sns.barplot(x=tmp[col],y=tmp['counts'],ax=ax)
    ax.set_xlabel('{}'.format(col), fontsize=15)
    return ax

f1, ax1 = plt.subplots(2,2, figsize = (20,10))
f2, ax2 = plt.subplots(2,2, figsize = (20,10))
f3, ax3 = plt.subplots(2,2, figsize = (20,10))

f1.tight_layout(h_pad=9, w_pad=5, rect=[0, 0.03, 1, 0.93])
f2.tight_layout(h_pad=9, w_pad=5, rect=[0, 0.03, 1, 0.93])
f3.tight_layout(h_pad=9, w_pad=5, rect=[0, 0.03, 1, 0.93])

cols1 = ['age', 'body_type', 'bra_size', 'category']
cols2 = ['cup_size', 'fit', 'height','hips']
cols3 = ['length','rented_for','size','weight']
k = 0

for i in range(2):
    for j in range(2):
        plot_dist(cols1[k], ax1[i][j])
        k += 1
k = 0     
for m in range(2):
    for n in range(2):
        plot_dist(cols2[k], ax2[m][n])
        k += 1
k = 0     
for o in range(2):
    for p in range(2):
        plot_dist(cols3[k], ax3[o][p])
        k += 1
        
plt.suptitle("Distributions of some features", fontsize= 15)
plt.show()
#%%
#丢弃 raiting 为空的样本
missing_rows =data[data['rating'].isnull()].index
data.drop(missing_rows, axis = 0, inplace=True)
print(data.columns)

#%%
#类别特征作类别编码
cate_fea = ['body_type','cup_size','fit','category',
            'rented_for','bra_size','length']
data[cate_fea] = data[cate_fea].astype(str)
data[cate_fea] = data[cate_fea].apply(LabelEncoder().fit_transform)
print(data[cate_fea])

#数值特征归一化
num_fea = ['age','height','hips','size','weight']
data[num_fea] = MinMaxScaler().fit_transform(data[num_fea])
print(data[num_fea])
#%%
#做特征关系热力图 为下面做交叉特征提供依据
corr = data.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(15, 15))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()
print(data)
#%%
#根据类别特征做一些数值特征的衍生统计特征
def agg_fea(tdf,main_col,cols,method):
    for c in cols:
        new_col_name = main_col + '_' + c + '_' + method
        tdf[new_col_name]=tdf.groupby([main_col])[c].transform(method)

agg_fea(data,'rented_for',['age'],'mean')
agg_fea(data,'rented_for',['age'],'std')
agg_fea(data,'rented_for',['age'],'min')
agg_fea(data,'rented_for',['age'],'max')

agg_fea(data,'cup_size',['height','hips','size','weight'],'mean')
agg_fea(data,'cup_size',['height','hips','size','weight'],'std')
agg_fea(data,'cup_size',['height','hips','size','weight'],'min')
agg_fea(data,'cup_size',['height','hips','size','weight'],'max')

agg_fea(data,'bra_size',['height','hips','size','weight'],'mean')
agg_fea(data,'bra_size',['height','hips','size','weight'],'std')
agg_fea(data,'bra_size',['height','hips','size','weight'],'min')
agg_fea(data,'bra_size',['height','hips','size','weight'],'max')

agg_fea(data,'body_type',['height','hips','size','weight'],'mean')
agg_fea(data,'body_type',['height','hips','size','weight'],'std')
agg_fea(data,'body_type',['height','hips','size','weight'],'min')
agg_fea(data,'body_type',['height','hips','size','weight'],'max')

#%%
#做一些类别特征的统计特征
#unique
for feat_i in cate_fea:
    for feat_j in cate_fea:
        if feat_i != feat_j:
            col_name = "unique_of_"+feat_i+"_and_"+feat_j
            data[col_name] = data[feat_i].map(data.groupby([feat_i])[feat_j].nunique())
#%%   
#二维count 即共现次数
cate_fea_ = ['body_type','cup_size','fit','bra_size','length']
data['cnt']=1
for feat_i in cate_fea_:
    for feat_j in cate_fea_:
        if feat_i != feat_j:
            col_name = "count_of_"+feat_i+"_and_"+feat_j
            se = data.groupby([feat_i,feat_j])['cnt'].sum()
            dt = data[[feat_i,feat_j]]            
            data[col_name] = (pd.merge(dt,se.reset_index(),how='left',
                            on=[feat_i,feat_j]).sort_index()['cnt'].fillna(value=0)).astype(int)
#%%
print(data.columns)

# 分离出评价
y = data['rating'].values
data.drop(['rating'], axis=1, inplace=True)

#归一化
y_norm = (y - min(y)) / (max(y) - min(y))

#%%
# 数据分割为训练集测试集
X_train, X_test, y_train, y_test = train_test_split(data, y_norm, test_size=0.1)

#%%
#XGboot回归
params = {
    'min_child_weight': 10.0,
    'learning_rate': 0.02,
    'objective': 'reg:squarederror',
    'eval_metric': 'mae',
    'max_depth': 7,
    'max_delta_step': 1.8,
    'colsample_bytree': 0.4,
    'subsample': 0.8,
    'eta': 0.025,
    'gamma': 0.65,
    'nthread': -1,
    'seed': 2020,
}

folds = KFold(n_splits=5, shuffle=True, random_state=2020)
oof = np.zeros(X_train.shape[0])
predictions = np.zeros(X_test.shape[0])


y_train = np.log1p(y_train)
feature_importance_df = pd.DataFrame()
for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train)):
    print("fold {}".format(fold_ + 1))
    trn_data = xgb.DMatrix(X_train.iloc[trn_idx], label=y_train[trn_idx])
    val_data = xgb.DMatrix(X_train.iloc[val_idx], label=y_train[val_idx])

    watchlist = [(trn_data, 'train'), (val_data, 'valid')]
    clf = xgb.train(params, 
                    trn_data, 
                    1000,
                    watchlist, 
                    verbose_eval=200, 
                    early_stopping_rounds=200)

    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = clf.get_fscore().keys() 
    fold_importance_df["importance"] = clf.get_fscore().values()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    predictions += clf.predict(xgb.DMatrix(X_test), ntree_limit=clf.best_ntree_limit) / folds.n_splits

print('MSE of XGboost: ' + str(mean_squared_error(predictions, y_test)))
print('MAE of Xgboost: ' + str(mean_absolute_error(predictions, y_test)))

#特征重要性

cols = (feature_importance_df[["Feature", "importance"]].groupby("Feature").mean().sort_values(by="importance", ascending=False)[:35].index)
best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)].sort_values(by='importance',ascending=False)
plt.figure(figsize=(8, 10))
sns.barplot(y="Feature",
            x="importance",
            data=best_features.sort_values(by="importance", ascending=False))
plt.title('XGboost Features (avg over folds)')
plt.tight_layout()
#%%
#LGBM回归
params = {'num_leaves': 38,
          'min_data_in_leaf': 50,
          'objective': 'regression',
          'max_depth': -1,
          'learning_rate': 0.02,
          "min_sum_hessian_in_leaf": 6,
          "boosting": "gbdt",
          "feature_fraction": 0.9,
          "bagging_freq": 1,
          "bagging_fraction": 0.7,
          "bagging_seed": 11,   
          "lambda_l1": 0.1,
          "verbosity": -1,
          "nthread": 4,
          'metric': 'mae',
          "random_state": 2020,
          }

folds = KFold(n_splits=5, shuffle=True, random_state=2020)
oof = np.zeros(data.shape[0])
predictions = np.zeros(X_test.shape[0])

features = data.columns
feature_importance_df = pd.DataFrame()
y_train = np.log1p(y_train)
for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train)):
    print("fold {}".format(fold_ + 1))

    trn_data = lgb.Dataset(X_train.iloc[trn_idx], label=y_train[trn_idx])
    val_data = lgb.Dataset(X_train.iloc[val_idx], label=y_train[val_idx])


    clf = lgb.train(params,
                    trn_data,
                    3000,
                    valid_sets=[trn_data, val_data],
                    verbose_eval=200,
                    early_stopping_rounds=200)

    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    predictions += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits


print('MSE of LGBM: ' + str(mean_squared_error(predictions, y_test)))
print('MAE of LGBM: ' + str(mean_absolute_error(predictions, y_test)))
#特征重要性
cols = (feature_importance_df[["Feature", "importance"]].groupby("Feature").mean().sort_values(by="importance", ascending=False)[:35].index)
best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)].sort_values(by='importance',ascending=False)
plt.figure(figsize=(8, 10))
sns.barplot(y="Feature",
            x="importance",
            data=best_features.sort_values(by="importance", ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()