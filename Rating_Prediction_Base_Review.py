# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 14:40:36 2020

@author: ShuaiLiu
"""
import pandas as pd
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb
import re
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

#%%
df = pd.read_json('D:\\bisai\\clothes\\data\\renttherunway_final_data.json', lines=True)
# df = df[:50000]
#只保留文本数据
df = df[['review_text','review_summary','rating']]
#将text和summary合并
df['review'] = df['review_text'] + df['review_summary']

#丢弃无用行和空值的行
df = df.drop(['review_text','review_summary'],axis=1) 
df = df.dropna(axis=0,how='any').reset_index(drop=True)
# print(df)
# print(df['rating'].isnull().value_counts()) 

df['rating'] = [df['rating'][i]/2 for i in range(len(df['rating']))]
# print(df)
#%%

def preprocess(text):
    #去除标点符号
    text = re.sub(r'[{}]+'.format('.!,/;:?"\''),"",text)
    #分词
    token_words = word_tokenize(text)
    #去除停用词
    stop_words = stopwords.words('english')
    filter_words = [word for word in token_words if word not in stop_words]
    
    stemmer = PorterStemmer()
    final_word = [stemmer.stem(word) for word in filter_words]
    
    return " ".join(final_word)

df['review'] = df['review'].apply(preprocess)
print(df['review'][:10])
#%%
#将DataFrame转换为矩阵 
dataset = df.values
# print(dataset)

# 提取特征(第2列 处理后的文本)
features = dataset[:,1]
# print(features)

# 提取评价(第1列)
target = dataset[:,0]
# print(target)

#对输入的文本转化为词向量 
tfidf = TfidfVectorizer()
X_processed = tfidf.fit_transform(features)

print(X_processed.shape) #每行代表一个文本的特征向量
print(X_processed[0].shape)
print(X_processed[0]) #稀疏向量
#%%
#分割数据集
X_train, X_test, y_train, y_test = train_test_split(X_processed, target, test_size=0.25, random_state=42)
#%%
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

predictions = np.zeros(X_test.shape[0])

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train)):
    print("fold {}".format(fold_ + 1))
    trn_data = xgb.DMatrix(X_train[trn_idx], label=y_train[trn_idx])
    val_data = xgb.DMatrix(X_train[val_idx], label=y_train[val_idx])

    watchlist = [(trn_data, 'train'), (val_data, 'valid')]
    clf = xgb.train(params, 
                    trn_data, 
                    3000,
                    watchlist, 
                    verbose_eval=200, 
                    early_stopping_rounds=200)

    predictions += clf.predict(xgb.DMatrix(X_test), ntree_limit=clf.best_ntree_limit) / folds.n_splits

print('MSE of XGboost Based Review: ' + str(mean_squared_error(predictions, y_test)))
print('MAE of Xgboost Based Review: ' + str(mean_absolute_error(predictions, y_test)))

#%%
# #LGBM回归
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
predictions = np.zeros(X_test.shape[0])

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train)):
    print("fold {}".format(fold_ + 1))

    trn_data = lgb.Dataset(X_train[trn_idx], label=y_train[trn_idx])
    val_data = lgb.Dataset(X_train[val_idx], label=y_train[val_idx])


    clf = lgb.train(params,
                    trn_data,
                    3000,
                    valid_sets=[trn_data, val_data],
                    verbose_eval=200,
                    early_stopping_rounds=200)

    predictions += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits


print('MSE of LGBM Based Review: ' + str(mean_squared_error(predictions, y_test)))
print('MAE of LGBM Based Review: ' + str(mean_absolute_error(predictions, y_test)))