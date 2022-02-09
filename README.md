# Bike_Casestudy_Sairam

**Reading and Understanding the Data**

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

bike = pd.read_csv("C://Users//Sairam//Downloads//Bike_Sharing//day.csv")
bike.head()
bike.info()
bike.describe()
bike.shape

**Check for NULL/MISSING values**

print(bike.isnull().sum())
print(bike.isnull().sum(axis = 1))

**Duplicate Check**
bike_dup = bike.copy()
bike_dup.drop_duplicates(subset = None, inplace = True)
bike_dup.shape

**Data Cleaning**
bike_dummy=bike.iloc[:,1:16]

for col in bike_dummy:
    print(bike_dummy[col].value_counts(ascending=False), '\n\n\n')
 
**Removing redundant & unwanted columns**

bike_new = bike[['season','yr','mnth','holiday','weekday','workingday','temp','atemp','hum','windspeed','cnt','weathersit']]
bike_new.info()
bike_new ['season'] = bike_new ['season'].astype('category')
bike_new ['mnth'] = bike_new ['mnth'].astype('category')
bike_new ['weathersit'] = bike_new ['weathersit'].astype('category')
bike_new ['weekday'] = bike_new ['weekday'].astype('category')
bike_new = pd.get_dummies(bike_new, drop_first = True)
bike_new.info()

**SPLITTING THE DATA**

from sklearn.model_selection import train_test_split
df_train, df_test  = train_test_split(bike_new, train_size = 0.7, test_size = 0.3, random_state = 333)
df_train.shape
df_test.shape

**EXPLORATORY DATA ANALYSIS**

bike_num = df_train[['temp','atemp','hum','windspeed','cnt']]

sns.pairplot(bike_num, diag_kind='kde')
plt.show()

**Visualising Catagorical Variables**

plt.figure(figsize = (20,12))
plt.subplot(2,3,1)
sns.boxplot(x = 'season', y = 'cnt', data = bike)
plt.subplot(2,3,2)
sns.boxplot(x = 'mnth', y = 'cnt', data = bike)
plt.subplot(2,3,3)
sns.boxplot(x = 'weekday', y = 'cnt', data = bike)
plt.subplot(2,3,4)
sns.boxplot(x = 'weathersit', y = 'cnt', data = bike)
plt.subplot(2,3,5)
sns.boxplot(x = 'holiday', y = 'cnt', data = bike)
plt.subplot(2,3,6)
sns.boxplot(x = 'workingday', y = 'cnt', data = bike)

**Correlation Matrix**
plt.figure(figsize = (25,20))
sns.heatmap(bike_new.corr(), annot = True, cmap="RdBu")
plt.show()

**RESCALING THE FEATURES**

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
nums_vars = ['temp','atemp','hum','windspeed','cnt']
df_train[nums_vars] = scaler.fit_transform(df_train[nums_vars])
df_train.head()
df_train.describe()

**BUILDING A LINEAR MODEL **

y_train = df_train.pop('cnt')
X_train = df_train
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
rfe = RFE (lm,15)
rfe = rfe.fit(X_train,y_train)
list(zip(X_train.columns,rfe.support_ ,rfe.ranking_))
col = X_train.columns[rfe.support_]
col
X_train.columns[~rfe.support_]
X_train_rfe = X_train[col]

**Building Linear Model using 'STATS MODEL'**

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
X = X_train_rfe.columns
vif['Features'] = X_train_rfe.columns
vif['VIF'] = [variance_inflation_factor(X_train_rfe.values, i) for i in range(X_train_rfe.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif

import statsmodels.api as sm  
X_train_lm1 = sm.add_constant(X_train_rfe)
lm1 = sm.OLS(y_train,X_train_lm1).fit() 

lm1.params

print(lm1.summary())

X_train_new = X_train_rfe.drop(['atemp'], axis=1)

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
X = X_train_new.columns
vif['Features'] = X_train_new.columns
vif['VIF'] = [variance_inflation_factor(X_train_new.values, i) for i in range(X_train_new.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif

X_train_lm2 = sm.add_constant(X_train_new)
lm2 = sm.OLS(y_train,X_train_lm2).fit() 

lm2.params

print(lm2.summary())

X_train_new = X_train_new.drop(['hum'], axis=1)

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
X = X_train_new.columns
vif['Features'] = X_train_new.columns
vif['VIF'] = [variance_inflation_factor(X_train_new.values, i) for i in range(X_train_new.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif

X_train_lm3 = sm.add_constant(X_train_new)
lm3 = sm.OLS(y_train,X_train_lm3).fit()

lm3.params

print(lm3.summary())

X_train_new = X_train_new.drop(['season_3'], axis=1)

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
X = X_train_new.columns
vif['Features'] = X_train_new.columns
vif['VIF'] = [variance_inflation_factor(X_train_new.values, i) for i in range(X_train_new.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif

X_train_lm4 = sm.add_constant(X_train_new)
lm4 = sm.OLS(y_train,X_train_lm4).fit()

lm4.params

print(lm4.summary())

X_train_new = X_train_new.drop(['mnth_10'], axis=1)

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
X = X_train_new.columns
vif['Features'] = X_train_new.columns
vif['VIF'] = [variance_inflation_factor(X_train_new.values, i) for i in range(X_train_new.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif

X_train_lm5 = sm.add_constant(X_train_new)
lm5 = sm.OLS(y_train,X_train_lm5).fit() 

lm5.params

print(lm5.summary())

X_train_lm6 = sm.add_constant(X_train_new)
lm6 = sm.OLS(y_train,X_train_lm6).fit() 

lm6.params

print(lm6.summary())

nums_vars = ['temp','atemp','hum','windspeed','cnt']
df_test[nums_vars] = scaler.transform(df_test[nums_vars])
df_test.head()
df_test.describe()

y_test = df_test.pop('cnt')
X_test = df_test

col1=X_train_new.columns
X_test=X_test[col1]
X_test_lm6 = sm.add_constant(X_test)
X_test_lm6.info()

y_pred = lm6.predict(X_test_lm6)

**MODEL EVALUATION**

figure = plt.figure()
plt.scatter(y_test,y_pred)
plt.suptitle('y_test vs y_pred', fontsize = 20)
plt.xlabel('y_test', fontsize = 16)
plt.ylabel('y_pred', fontsize = 16)
plt.show()

**R^2 Value for TEST**

from sklearn.metrics import r2_score

r2_score (y_test,y_pred)

r2 = 0.8203092200749708

n = X_test.shape[0]
p = X_test.shape[1]

adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
print(adjusted_r2)




