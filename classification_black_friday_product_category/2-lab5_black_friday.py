#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 18:19:19 2019

@author: yingxin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import metrics
from sklearn.metrics import accuracy_score

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


#Read Data
data = pd.read_csv("black_friday_data_kaggle.csv",encoding='utf-8')
data.head(10)
data.dtypes
data.info()
data['User_ID'].nunique()
data.loc[1,['User_ID','Product_Category_2']]
data[['User_ID','Product_Category_2']].head()
df1 = data.drop(['Unnamed: 0', 'User_ID','Product_Category_2', 'Product_Category_3'],axis=1)


#Preprocessing
df_group = data.groupby(by='Product_ID', as_index=False).agg({'User_ID': pd.Series.nunique})
df1=pd.merge(df1, df_group, how='inner', on='Product_ID')
df1.dtypes
df1['Occupation'] = df1['Occupation'].astype("object")
df1['Marital_Status'] = df1['Marital_Status'].astype("object")
df1['Product_Category_1'] = df1['Product_Category_1'].astype("int")
df1.dtypes
# =============================================================================
# #factor
# le = preprocessing.LabelEncoder()
# df1['Gender'] = le.fit_transform(df1['Gender'])
# df1['Age'] = le.fit_transform(df1['Age'])
# df1['City_Category'] = le.fit_transform(df1['City_Category'])
# df1['Stay_In_Current_City_Years'] = le.fit_transform(df1['Stay_In_Current_City_Years'])
# =============================================================================
#Dummy
df1 = pd.get_dummies(data=df1, columns=['Gender','Age','Occupation',
 'City_Category',
 'Stay_In_Current_City_Years',
 'Marital_Status'] )


# Data Exploration
#find correlation
corr = df_train.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(df_train.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(df_train.columns)
ax.set_yticklabels(df_train.columns)
plt.show()

#Split train and test data set
# Train Dataset
df_train = df1[df1.Product_Category_1 != -1]
df_train = df_train.drop(['Product_ID'],axis=1)
y = df_train['Product_Category_1']
X = df_train.drop('Product_Category_1', axis=1)
# Test Dataset
df_test = df1[df1.Product_Category_1 == -1] # Not to be touched until the model is trained and ready
Product_ID_test = df_test['Product_ID']
df_test = df_test.drop(['Product_ID'],axis=1)
df_test = df_test.drop('Product_Category_1', axis=1)

# Split Train Dataset into Traning and Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X_train=X_train.astype(int)
y_train=y_train.astype(int)
X_test=X_test.astype(int)
df_test=df_test.astype(int)


#model
#Lasso
# =============================================================================
# from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC
# import time
# X = X_train
# y = y_train
# 
# model_bic = LassoLarsIC(criterion='bic')
# t1 = time.time()
# model_bic.fit(X, y)
# t_bic = time.time() - t1
# alpha_bic_ = model_bic.alpha_
# 
# model_aic = LassoLarsIC(criterion='aic')
# model_aic.fit(X, y)
# alpha_aic_ = model_aic.alpha_
# 
# 
# def plot_ic_criterion(model, name, color):
#     alpha_ = model.alpha_
#     alphas_ = model.alphas_
#     criterion_ = model.criterion_
#     plt.plot(-np.log10(alphas_), criterion_, '--', color=color,
#              linewidth=3, label='%s criterion' % name)
#     plt.axvline(-np.log10(alpha_), color=color, linewidth=3,
#                 label='alpha: %s estimate' % name)
#     plt.xlabel('-log(alpha)')
#     plt.ylabel('criterion')
# 
# plt.figure()
# plot_ic_criterion(model_aic, 'AIC', 'b')
# plot_ic_criterion(model_bic, 'BIC', 'r')
# plt.legend()
# plt.title('Information-criterion for model selection (training time %.3fs)'
#           % t_bic)
# =============================================================================
# =============================================================================
# from sklearn import linear_model
# la = linear_model.Lasso(alpha=0.000002)
# la = la.fit(X_train, y_train)
# y_pred = la.predict(X_train)
# y_pred = np.where(y_pred > 0.5, 1, 0)
# print('Training data accuracy', float(accuracy_score(y_pred, y_train))*100, '%')
# y_pred = la.predict(X_test)
# y_pred = np.where(y_pred > 0.5, 1, 0)
# print('Testing data accuracy', float(accuracy_score(y_pred, y_test))*100, '%')
# 
# =============================================================================

#neutral network
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=1, activation='relu', solver='lbfgs', 
                    learning_rate = 'adaptive', random_state=10,
                    early_stopping = True, alpha = 0.0001)
mlp = mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_train)
#print(mlp.out_activation_)
print('Training data accuracy', float(accuracy_score(y_pred, y_train))*100, '%')
y_pred = mlp.predict(X_test)
print('Testing data accuracy', float(accuracy_score(y_pred, y_test))*100, '%')

# =============================================================================
# #Logistic
# from sklearn.linear_model import LogisticRegression
# lr = LogisticRegression(solver='lbfgs', multi_class='multinomial',random_state=10)
# lr = lr.fit(X_train, y_train)
s# y_pred = lr.predict(X_train)
# print('Training data accuracy', float(accuracy_score(y_pred, y_train))*100, '%')
# y_pred = lr.predict(X_test)
# print('Testing data accuracy', float(accuracy_score(y_pred, y_test))*100, '%')
# =============================================================================

#Decision Tree
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(max_depth=20)
dt = dt.fit(X_train, y_train)
y_pred = dt.predict(X_train)
print('Training data accuracy', float(accuracy_score(y_pred, y_train))*100, '%')
y_pred = dt.predict(X_test)
print('Testing data accuracy', float(accuracy_score(y_pred, y_test))*100, '%')

# =============================================================================
# #SVM
# from sklearn import svm
# svmm = svm.SVC(kernel='linear')
# svmm = svmm.fit(X_train, y_train)
# y_pred = svmm.predict(X_train)
# print('Training data accuracy', float(accuracy_score(y_pred, y_train))*100, '%')
# y_pred = svmm.predict(X_test)
# print('Testing data accuracy', float(accuracy_score(y_pred, y_test))*100, '%')
# =============================================================================

#RF
clf = RandomForestClassifier(n_jobs=2, random_state=0,n_estimators=100)
clf.fit(X_train,y_train)
clf.predict(X_test)
y_pred = clf.predict(X_train)
print('Training data accuracy', float(accuracy_score(y_pred, y_train))*100, '%')
y_pred = clf.predict(X_test)
print('Testing data accuracy', float(accuracy_score(y_pred, y_test))*100, '%')


#Max_Voting
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
maxV = VotingClassifier(estimators=[('dt', dt),('rf',clf),('nt',xgb)], voting='hard')
maxV = maxV.fit(X_train,y_train)
y_pred = maxV.predict(X_train)
print('Training data accuracy', float(accuracy_score(y_pred, y_train))*100, '%')
y_pred = maxV.predict(X_test)
print('Testing data accuracy', float(accuracy_score(y_pred, y_test))*100, '%')

#Stacking
from sklearn.linear_model import LogisticRegression
dt = DecisionTreeClassifier(max_depth=6)
lr = LogisticRegression()
from mlxtend.classifier import StackingClassifier
import numpy as np
sclf = StackingClassifier(classifiers=[dt, lr, clf], 
                          meta_classifier=lr)
sclf = sclf.fit(X_train, y_train)
y_pred = sclf.predict(X_train)
print('Training data accuracy', float(accuracy_score(y_pred, y_train))*100, '%')
y_pred = sclf.predict(X_test)
print('Testing data accuracy', float(accuracy_score(y_pred, y_test))*100, '%')

#Xgboost
from xgboost import XGBClassifier

xgb = XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=1,
       learning_rate=0.1, max_delta_step=0, 
       min_child_weight=1, missing=None, n_estimators=100, nthread=-1,
       objective='multiclass', reg_alpha=0, reg_lambda=1,
       scale_pos_weight=1, silent=True, subsample=1,
       max_depth=6,eta=0.05,gamma= 0.1)
xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_train)
print('Training data accuracy', float(accuracy_score(y_pred, y_train))*100, '%')
y_pred = xgb.predict(X_test)
print('Testing data accuracy', float(accuracy_score(y_pred, y_test))*100, '%')


#Test
y_pred_test = xgb.predict(df_test)
# write to submission file
df = pd.DataFrame({'Product_ID': Product_ID_test,
                   'Product_Category_1': y_pred_test
                   })
tem=df.groupby(['Product_ID']).agg(lambda x:x.value_counts().index[0])
tem['Product_ID'] = tem.index
tem.to_csv('submission_file.csv', index=False)
