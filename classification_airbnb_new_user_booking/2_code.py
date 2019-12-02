from pandas import read_csv  
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import preprocessing
from datetime import datetime,date

# ～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～
# ～～～～～～～～～～～～～～～～～～Read Data～～～～～～～～～～～～～～～～～～～
#age_genderr_ = read_csv("age_gender_bkts.csv",encoding='utf-8')
#countries = read_csv("countries.csv",encoding='utf-8')
#age_gender_countries = age_gender.merge(countries, left_on='country_destination', right_on='country_destination')
#age_gender_countries.to_csv('age_gender_countries.csv',encoding='utf-8')
age_gender_countries = read_csv("age_gender_countries.csv",encoding='utf-8')
age_gender_countries.language.unique()
#age_gender_countries.drop(['Unnamed: 0','Unnamed: 0.1'], axis=1, inplace=True)
#age_gender_countries.to_csv('age_gender_countries.csv',encoding='utf-8')
#session = read_csv("sessions.csv",encoding='utf-8')
session_pre = read_csv("df_session_1.csv",encoding='utf-8')
train_users = read_csv("train_users_2.csv",encoding='utf-8')
test_users = read_csv("test_users.csv",encoding='utf-8')


# ～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～
# ～～～～～～～～～～～～～～～～～～Data Preprocessing～～～～～～～～～～～～～～～～～～～
### +++++++++++++++++++++++Dummy All Features++++++++++++++++++++++++++++++
## ===================== Explore the dataset =======================
train_users.dtypes # age and signup_flow are int or float 64 others are object and date(three columns) are datetime
train_users.describe() # find incorrect minimum and max age values
# store id
train_id = train_users['id']
test_id = test_users['id']
## store country names
train_label = train_users['country_destination'].values
train_users_1 = train_users.drop(['country_destination'], axis = 1)
# combine 2 dataset to do data cleaning
users = pd.concat((train_users_1, test_users), axis=0, ignore_index=True)
# drop id which are saved separately and date of first booking which is completely absent in the test data
drop_list = ['date_first_booking','id']
users.drop(drop_list, axis=1, inplace=True)
users.head()


## ==========================Data Cleaning================================
# Date: date_account_created and timestamp_first_active
# split datetime to year month and day
users['date_account_created'] = pd.to_datetime(users['date_account_created'])
users['year_account_created'] = users.date_account_created.dt.year
users['month_account_created'] = users.date_account_created.dt.month
users['day_account_created'] = users.date_account_created.dt.day
# split datetime to year month and day
users['timestamp_first_active'] = pd.to_datetime(users['timestamp_first_active'])
users['year_first_active'] = users.timestamp_first_active.dt.year
users['month_first_active'] = users.timestamp_first_active.dt.month
users['day_first_active'] = users.timestamp_first_active.dt.day
# Delete columns
drop_list = ['date_account_created','timestamp_first_active']
users.drop(drop_list, axis=1, inplace=True)
users.head()
# Gender
users.gender.unique()
users.gender.replace('-unknown-',np.nan, inplace = True)
# Age
# Remove weird age values and add missing value
users.age[users['age'] >= 100] = 100
users.age[users['age'] <= 13] = 13
age_mean = users.age.mean() 
users.age[users['age'].isnull()] = int(age_mean)
# signup_method
users.signup_method.unique()
users.signup_method.replace('-unknown-',np.nan, inplace = True)
# signup_flow
users.signup_flow.unique()
users.signup_flow.replace('-unknown-',np.nan, inplace = True)
# language
users.language.unique()
#users.language.replace('-unknown-',np.nan, inplace = True)
users.language.replace(['zh', 'ko', 'ja', 'ru', 'pl','el', 'sv',  
                        'hu', 'da', 'id', 'fi', 'no', 'tr', 'th', 
                        'cs','hr', 'ca', 'is','-unknown-'],'en', inplace = True)
# affiliate_channel
users.affiliate_channel.unique()
users.affiliate_channel.replace('-unknown-',np.nan, inplace = True)
# affiliate_provider
users.affiliate_provider.unique()
users.affiliate_provider.replace('-unknown-',np.nan, inplace = True)
# first_affiliate_tracked
users.first_affiliate_tracked.unique()
users.first_affiliate_tracked.replace('-unknown-',np.nan, inplace = True)
# signup_app
users.signup_app.unique()
users.signup_app.replace('-unknown-',np.nan, inplace = True)
# first_device_type
users.first_device_type.unique()
users.first_device_type.replace('-unknown-',np.nan, inplace = True)
# first_browser
users.first_browser.unique()
users.first_browser.replace('-unknown-',np.nan, inplace = True)

#======================= Explore the Data=======================
users.dtypes
users.describe()

# Gender
pd.value_counts(users['gender'])
users.groupby('gender').age.agg(['min','max','mean','count'])
users.gender.value_counts(dropna = False).plot(kind = 'bar')
users.groupby('gender').age.mean().plot(kind = 'bar')
# Age
users.age.plot(kind = 'hist')
# signup_method
users.signup_method.value_counts(dropna = False).plot(kind = 'bar')
# signup_flow
users.signup_flow.plot(kind = 'hist')
# language
users.language.value_counts(dropna = False).plot(kind = 'bar')
train_users.language.value_counts(dropna = False).plot(kind = 'bar')
# affiliate_channel
users.affiliate_channel.value_counts(dropna = False).plot(kind = 'bar')
# affiliate_provider
users.affiliate_provider.value_counts(dropna = False).plot(kind = 'bar')
# first_affiliate_tracked
users.first_affiliate_tracked.value_counts(dropna = False).plot(kind = 'bar')
# signup_app
users.signup_app.value_counts(dropna = False).plot(kind = 'bar')
# first_device_type
users.first_device_type.value_counts(dropna = False).plot(kind = 'bar')
# first_browser
users.first_browser.value_counts(dropna = False).plot(kind = 'bar')

# ======================= Data Preprocessing =============================
users.dtypes
## add age，gender，countries
users_age_gender_countries = users.merge(age_gender_countries, 
                                         left_on='language', right_on='language',
                                         how ='left')
drop_list = ['Unnamed: 0','country_destination']
users_age_gender_countries.drop(drop_list, axis=1, inplace=True)

users = users_age_gender_countries

#users_age_gender_countries.to_csv('users_age_gender_countries.csv',encoding='utf-8')
## numeric features
## Normalize age_date

## Initialize a MinMax scaler, then apply it to the numerical features
#from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import MinMaxScaler
#scaler = MinMaxScaler()
#numerical = ['age','signup_flow','year_account_created','month_account_created','day_account_created',
#             'year_first_active','month_first_active','day_first_active']
#users[numerical] = scaler.fit_transform(users[numerical])

from sklearn.preprocessing import Normalizer
transformer = Normalizer()
numerical = ['age','signup_flow','year_account_created','month_account_created','day_account_created',
             'year_first_active','month_first_active','day_first_active']
users[numerical] = transformer.transform(users[numerical])

## object features
## Dummy
objects = ['gender','signup_method','language','affiliate_channel','affiliate_provider',
           'first_affiliate_tracked','signup_app','first_device_type','first_browser']
users = pd.get_dummies(users,columns=objects)


### add cluster value
#from sklearn.cluster import KMeans
#kmeans = KMeans(n_clusters=7, random_state=0).fit(users)
#label = kmeans.labels_
#label = pd.DataFrame(data=label,columns=['label'])
#users_test = pd.concat([users,label],axis=1)
#users = users_test



#～～～～～～～～～～～～～～～～～～Training Model～～～～～～～～～～～～～～～～～～～～～～
## Divide the training data into test and train data to measure accuracy
train_label = train_users['country_destination'].values
Nrows_train = train_users.shape[0] 

users_c = users
#users.fillna(users.mean())
#pd.isnull(users).sum() > 0
train = users_c[:Nrows_train]
test = users_c[Nrows_train:]

# Label
from sklearn.preprocessing import LabelEncoder
labler = LabelEncoder()
y = labler.fit_transform(train_label)
X = train

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#find correlation
#corr = train.corr()
#fig = plt.figure()
#ax = fig.add_subplot(111)
#cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
#fig.colorbar(cax)
#ticks = np.arange(0,len(train.columns),1)
#ax.set_xticks(ticks)
#plt.xticks(rotation=90)
#ax.set_yticks(ticks)
#ax.set_xticklabels(train.columns)
#ax.set_yticklabels(train.columns)
#plt.show()
## +++++++++++++++++++++++ Dummy All Features +++++++++++++++++++++++++++++

### +++++++++++++++++++++++ Encoder All Features +++++++++++++++++++++++++++++
### ==================== Explore Dataset ==============================
#train = train_user
#test = test_users
#train.info()
#test.info()
#
#train_id = train_users['id']
#test_id = test_users['id']
#
### ========================= Data Cleaning ========================
#train = train.drop('id', 1)
#test = test.drop('id', 1)
## drop id which are saved separately and date of first booking which is completely absent in the test data
#train = train.drop('date_first_booking', 1)
#test = test.drop('date_first_booking', 1)
#train = train.drop('timestamp_first_active', 1)
#test = test.drop('timestamp_first_active', 1)
##Replace nan as class none in f_a_B
#train['first_affiliate_tracked'] = train['first_affiliate_tracked'].replace(np.NaN,'None')
#test['first_affiliate_tracked'] = test['first_affiliate_tracked'].replace(np.NaN,'None')
##age=>mean
#age_mean = train.age.mean()     
#train.age[train['age'].isnull()] = age_mean 
#age_mean = test.age.mean()     
#test.age[test['age'].isnull()] = age_mean 
## ===================== Data Preprocessing ==========================
## Factors to LabelEncoder
#le = preprocessing.LabelEncoder()
#
#train.gender.unique()
#le.fit(['FEMALE', '-unknown-', 'MALE', 'OTHER'])
#train['gender'] = le.transform(train['gender'])
#test['gender'] = le.transform(test['gender'])
#
#train.signup_method.unique()
#le.fit(['basic', 'facebook', 'google','weibo'])
#train['signup_method'] = le.transform(train['signup_method'])
#test['signup_method'] = le.transform(test['signup_method'])
#
#len(train.language.unique().tolist())
#test.language.unique().tolist()
#le.fit(['en','fr','es','de','zh','ko','ja','it','pt','ru','sv','nl','pl','hu','da','no','fi',
# 'tr','cs','th','ca','is','el','-unknown-','id','hr'])
#train['language'] = le.transform(train['language'])
#test['language'] = le.transform(test['language'])
#
#train.affiliate_channel.unique().tolist()  #8
#test.affiliate_channel.unique().tolist()  #7
#le.fit(['direct', 'other', 'seo', 'sem-non-brand', 'content', 'sem-brand',
#       'remarketing', 'api'])
#train['affiliate_channel'] = le.transform(train['affiliate_channel'])
#test['affiliate_channel'] = le.transform(test['affiliate_channel'])
#
#train.affiliate_provider.unique()#17
#len(test.affiliate_provider.unique().tolist())
#le.fit(['direct','google','bing','facebook','other','craigslist','padmapper',
# 'email-marketing','yahoo','baidu','naver','gsp','facebook-open-graph',
# 'meetup','vast','d`aum','yandex','daum','wayn'])
#train['affiliate_provider'] = le.transform(train['affiliate_provider'])
#test['affiliate_provider'] = le.transform(test['affiliate_provider'])
#
#train.first_affiliate_tracked.unique()
#test.first_affiliate_tracked.unique()
#le.fit(['untracked', 'omg', 'None', 'linked', 'tracked-other', 'product',
#       'marketing', 'local ops'])
#train['first_affiliate_tracked'] = le.transform(train['first_affiliate_tracked'])
#test['first_affiliate_tracked'] = le.transform(test['first_affiliate_tracked'])
#
#train.signup_app.unique()
#test.signup_app.unique()
#le.fit(['Moweb', 'Web', 'iOS', 'Android'])
#train['signup_app'] = le.transform(train['signup_app'])
#test['signup_app'] = le.transform(test['signup_app'])
#
#len(train.first_device_type.unique())
#len(test.first_device_type.unique())
#le.fit(['iPhone', 'Windows Desktop', 'Mac Desktop', 'iPad',
#       'Android Tablet', 'Android Phone', 'Desktop (Other)',
#       'Other/Unknown', 'SmartPhone (Other)'])
#train['first_device_type'] = le.transform(train['first_device_type'])
#test['first_device_type'] = le.transform(test['first_device_type'])
#
#len(train.first_browser.unique())
#len(test.first_browser.unique())
#le.fit(['IE','Firefox','Chrome','Safari','-unknown-','Mobile Safari','RockMelt','Chromium',
# 'Android Browser','Chrome Mobile','Palm Pre web browser','AOL Explorer','Mobile Firefox',
# 'TenFourFox','Opera','Apple Mail','Silk','Camino','BlackBerry Browser','SeaMonkey','Iron',
# 'IE Mobile','Opera Mini','Kindle Browser','CoolNovo','Maxthon','IceWeasel','wOSBrowser',
# 'Sogou Explorer','Mozilla','NetNewsWire','CometBird','Avant Browser','Pale Moon',
# 'TheWorld Browser','SlimBrowser','SiteKiosk','Stainless','Googlebot','Yandex.Browser',
# 'IBrowse' ,'Nintendo Browser','Opera Mobile' ,'UC Browser','Arora','Conkeror',
# 'Google Earth','Crazy Browser','OmniWeb','PS Vita browser','Comodo Dragon','Flock',
# 'Epic','Outlook 2007','IceDragon'])
#train['first_browser'] = le.transform(train['first_browser'])
#test['first_browser'] = le.transform(test['first_browser'])
#
##Date
## convert date to datetime
#train['date_account_created'] = pd.to_datetime(train['date_account_created'])
#test['date_account_created'] = pd.to_datetime(test['date_account_created'])
## split datetime to year month and day
#train['year'] = train['date_account_created'].dt.year
#train['month'] = train['date_account_created'].dt.month
#train['daytime'] = train['date_account_created'].dt.day
#train = train.drop(columns = ['year'])
#test['year'] = test['date_account_created'].dt.year
#test['month'] = test['date_account_created'].dt.month
#test['daytime'] = test['date_account_created'].dt.day
#test = test.drop(columns = ['year'])
## convert date to weekday or weekend
#train['day'] = train.date_account_created.dt.weekday
#test['day'] = test.date_account_created.dt.weekday
## drop date_account_created date, substitue it by weekday or weekend
#train = train.drop(columns = ['date_account_created'])
#test = test.drop(columns = ['date_account_created'])
## replace date by weekday or weekend
#train['day'] = train['day'].replace([0,1,2,3,4],'weekday')
#train['day'] = train['day'].replace([5,6],'weekend')
#test['day'] = test['day'].replace([0,1,2,3,4],'weekday')
#test['day'] = test['day'].replace([5,6],'weekend')
#le.fit(['weekday','weekend'])
#train['day'] = le.transform(train['day'])
#test['day'] = le.transform(test['day'])
#
## =========================== Model Training Prepare ================================
## Divide the training data into test and train data to measure accuracy
#from sklearn.preprocessing import LabelEncoder
#labler = LabelEncoder()
#y = labler.fit_transform(train['country_destination'].values)
#X = train.drop('country_destination', axis=1)
#
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#
##find correlation
#corr = train.corr()
#fig = plt.figure()
#ax = fig.add_subplot(111)
#cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
#fig.colorbar(cax)
#ticks = np.arange(0,len(train.columns),1)
#ax.set_xticks(ticks)
#plt.xticks(rotation=90)
#ax.set_yticks(ticks)
#ax.set_xticklabels(train.columns)
#ax.set_yticklabels(train.columns)
#plt.show()
### +++++++++++++++++++++++ Encoder All Features +++++++++++++++++++++++++++++



# ～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～
## ～～～～～～～～～～～～～～～～～ Model ～～～～～～～～～～～～～～～～～～～～～～～～～～～

#+++++++++++++++++++++++++++++Logistic++++++++++++++++++++++++++++++++++
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(solver='lbfgs', multi_class='multinomial',random_state=1)
lr = lr.fit(X_train, y_train)
y_pred = lr.predict(X_train)
print('Training data accuracy', float(accuracy_score(y_pred, y_train))*100, '%') 
# 60.323609207618944 % train data
# 58.48320420664 % age gender countries features
y_pred = lr.predict(X_test)
print('Testing data accuracy', float(accuracy_score(y_pred, y_test))*100, '%') 
# 60.15985462598844 % train data
# 58.07152287795113 % age gender countries features
# ==================== submission ===============
lr.fit(X, y)
y_test = lr.predict_proba(test) 
#Taking the 5 classes with highest probabilities
import collections
ids = []  #list of ids
cts = []  #list of countries
for i in range(len(test_id)):
    idx = test_id[i]
    ids += [idx] * 5
    cts += labler.inverse_transform(np.argsort(y_test[i])[::-1])[:5].tolist()
# check unique number
len(collections.Counter(cts).keys())
#Generate submission
sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
sub.to_csv('submission.csv',index=False)
# 0.85604 train features
# 0.85673 age gender countries features


#+++++++++++++++++++++++++++++ Neural Network ++++++++++++++++++++++++++++++++++
#from sklearn.metrics import accuracy_score
#from sklearn.neural_network import MLPClassifier
#
#mlp = MLPClassifier(solver='lbfgs', activation='relu'\
#                    , hidden_layer_sizes=(5,3), learning_rate = 'adaptive', 
#                    random_state=10,early_stopping = True, alpha = 0.0001)
#mlp = mlp.fit(X_train, y_train)
#y_pred = mlp.predict(X_train)
#print('Training data accuracy', float(accuracy_score(y_pred, y_train))*100, '%') 
## 60.348082678376635 %
#y_pred = mlp.predict(X_test)
#print('Testing data accuracy', float(accuracy_score(y_pred, y_test))*100, '%') 
## 60.16127429407005 %

# 0.85917
# ++++++++++++++++++++++++++++++Decision Tree ++++++++++++++++++++++++++++++
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(max_depth=10)
dt = dt.fit(X_train, y_train)
y_pred = dt.predict(X_train)
print('Training data accuracy', float(accuracy_score(y_pred, y_train))*100, '%') 
# 64.1079070287808 % train 
# 64.06455402343859 % age gender country train
# 67.72298828070372 % age gender country train max depth 15
y_pred = dt.predict(X_test)
print('Testing data accuracy', float(accuracy_score(y_pred, y_test))*100, '%') 
# 62.69538181973054 % train
# 62.68686381124093 % age gender country train
# 61.43897556751231 % age gender country train max depth 15
# ==================== submission ===============
dt.fit(X, y)
y_ttest = dt.predict_proba(test) 
#Taking the 5 classes with highest probabilities
import collections
ids = []  #list of ids
cts = []  #list of countries
for i in range(len(test_id)):
    idx = test_id[i]
    ids += [idx] * 5
    cts += labler.inverse_transform(np.argsort(y_ttest[i])[::-1])[:5].tolist()
# check unique number
len(collections.Counter(cts).keys())
#Generate submission
sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
sub.to_csv('submission.csv',index=False)
#  0.86694 % train age gender countries

# ++++++++++++++++++++++++++++++ RF ++++++++++++++++++++++++++++++
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=10, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=500, n_jobs=None,
            oob_score=False, random_state=0, verbose=0, warm_start=False)
clf.fit(X_train,y_train)
clf.predict(X_test)
y_pred = clf.predict(X_train)
print('Training data accuracy', float(accuracy_score(y_pred, y_train))*100, '%') 
# 58.48320420664 %train 
# 60.77671803764719 % train age gender countries
y_pred = clf.predict(X_test)
print('Testing data accuracy', float(accuracy_score(y_pred, y_test))*100, '%') 
#58.07152287795113 %train
# 60.050540183705046 % train age gender countries
# ==================== submission ===============
clf.fit(X, y)
y_ttest = clf.predict_proba(test) 
#Taking the 5 classes with highest probabilities
import collections
ids = []  #list of ids
cts = []  #list of countries
for i in range(len(test_id)):
    idx = test_id[i]
    ids += [idx] * 5
    cts += labler.inverse_transform(np.argsort(y_ttest[i])[::-1])[:5].tolist()
# check unique number
len(collections.Counter(cts).keys())
#Generate submission
sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
sub.to_csv('submission.csv',index=False)
# 0.85281 % train
#  0.85674 train age gender countries

# +++++++++++++++++++++++++++ Xgboost +++++++++++++++++++++++++++++++++++
from xgboost import XGBClassifier

xgb = XGBClassifier(max_depth=10, learning_rate=0.3, n_estimators=22,
                    objective='multi:softprob', subsample=0.6, colsample_bytree=0.6, seed=0)
xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_train)
print('Training data accuracy', float(accuracy_score(y_pred, y_train))*100, '%') 
# 63.99462982127374 % train
# 64.08413280004474 %age gender contry train % max_depth=6
# 67.04682124576958 %age gender contry train % max_depth=10
y_pred = xgb.predict(X_test)
print('Testing data accuracy', float(accuracy_score(y_pred, y_test))*100, '%') 
# 63.288803077840406 % train
# 63.212141001433864 %age gender contry train % max_depth=6
# 62.895555019236504 %age gender contry train % max_depth=10
#  %age gender contry train % max_depth=10
# =================== submission ================
import xgboost
from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import LabelEncoder
# Implementation of the classifier (decision tree)
xgb = XGBClassifier(max_depth=10, learning_rate=0.3, n_estimators=22,
                    objective='multi:softprob', subsample=0.6, colsample_bytree=0.6, seed=0)               
xgb.fit(X, y)
y_ttest = xgb.predict_proba(test) 
#Taking the 5 classes with highest probabilities
import collections
ids = []  #list of ids
cts = []  #list of countries
for i in range(len(test_id)):
    idx = test_id[i]
    ids += [idx] * 5
    cts += labler.inverse_transform(np.argsort(y_ttest[i])[::-1])[:5].tolist()
# check unique number
len(collections.Counter(cts).keys())
#Generate submission
sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
sub.to_csv('submission.csv',index=False)
#  0.865 % train
#  0.86432 % train age gender countries 6
#  0.86288 % train age gender countries 10



# +++++++++++++++++++++++++++ Max_Voting +++++++++++++++++++++++++++ 
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

maxV = VotingClassifier(estimators=[('dt', dt),('rf',clf),('nt',mlp)], voting='hard')
maxV = maxV.fit(X_train,y_train)
y_pred = maxV.predict(X_train)
print('Training data accuracy', float(accuracy_score(y_pred, y_train))*100, '%') 
# 61.21374430117752 %
y_pred = maxV.predict(X_test)
print('Testing data accuracy', float(accuracy_score(y_pred, y_test))*100, '%') 
# 60.73907920328227 %


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
# 63.044359913853384 %
y_pred = sclf.predict(X_test)
print('Testing data accuracy', float(accuracy_score(y_pred, y_test))*100, '%') 
# 62.90691236388932 %




