#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
import statsmodels.api as sm
import os
warnings.filterwarnings("ignore")

wd = "C:/Users/lenwy/scm"
pd.set_option("display.max_rows", 100, "display.max_columns", 100)


# In[ ]:


df = pd.read_excel("/content/drive/MyDrive/SCM/CNC-Milling Machine_Production data.xlsx")


# In[ ]:


df


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


df.nunique().sort_values(ascending=True)


# In[ ]:


df.isnull().sum()


# In[ ]:


plt.figure(figsize=(15,15))
sns.heatmap(data=df.corr(), annot=True) 
plt.show()


# Considering the above correlation graph with pearson method (default), we can see that there is low correlation with a lot of the weighted values.

# In[ ]:


df.corr().processing_time


# In[ ]:


df.corr().processing_time[df.corr().processing_time > 0.3]


# In[ ]:


col_processing_time = ['processing_time',
                       'number_of_missing_datapoints',
                      'raw_volume',
                      'number_of_lines_of_code',
                      'number_tool_changes',
                      'number_of_travels_to_machine_zero_point_in_rapid_traverse',
                      'number_axis_rotations',
                       'weighted_tool_diameter',
                      'weighted_cutting_length']


# Here we can see that there are high correlation values for predicting processing time such as 'number_of_missing_datapoints','number_of_travels_to_machine_zero_point_in_rapid_traverse','number_axis_rotations'.
# 
# The above features can be considered to predict processing time
# 
# 

# In[ ]:


df.corr().average_power_consumption


# on the other hand, average power consumption doesnt really have any strong correlation, positive or negative. 
# Lets see if Spearman method gives us any better results?

# In[ ]:


df.corr(method='spearman').average_power_consumption


# Once we use spearman, we can see moderate correalation from a lot of features now which can be valuable candidates for predicting the average power consumption

# In[ ]:


df.corr(method='spearman').average_power_consumption[df.corr(method='spearman').average_power_consumption < -0.25]


# In[ ]:


df.corr(method='spearman').average_power_consumption[df.corr(method='spearman').average_power_consumption > 0.25]


# In[ ]:


col_avg_pwr_c=['weighted_number_of_cutting_edges',
              'processing_time',
              'average_power_consumption',
              'raw_volume',
              'number_of_lines_of_code',
              'number_of_travels_to_machine_zero_point_in_rapid_traverse',
              'number_axis_rotations',
              'weighted_cutting_length']


# The above features can be considered to predict power consumption

# # Some EDA

# In[ ]:


plt.figure(figsize=(12,8))
sns.boxplot(data=df)


# In[ ]:


plt.figure(figsize=(12,8))
df[col_processing_time].plot.box()


# In[ ]:


plt.figure(figsize=(12,8))
df[col_avg_pwr_c].plot.box()


# In[ ]:


for i in col_processing_time:
    plt.figure()
    df[i].plot.box()


# In[ ]:


for i in col_avg_pwr_c:
    plt.figure()
    df[i].plot.box()


# The above box plot graph shows the Q1 to Q3 distribution as well as the outliers of the data. The green line the is Q2 or the median part of the data

# In[ ]:


df.hist(column=col_processing_time,figsize=(15,8))


# In[ ]:


df.hist(column=col_avg_pwr_c,figsize=(15,8))


# # Model Training

# In[ ]:


df_pwr = df[col_avg_pwr_c]
df_pt = df[col_processing_time]
y_pt = df_pt['processing_time']
y_pwr = df_pwr['average_power_consumption']
X_pt =df_pt.drop(['processing_time'],axis=1)
X_pwr =df_pwr.drop(['average_power_consumption'],axis=1)


# In[ ]:


from sklearn.preprocessing import StandardScaler



# In[ ]:


sc_x = StandardScaler()


# In[ ]:


Xpt = sc_x.fit_transform(X_pt)
Xpwr = sc_x.fit_transform(X_pwr)


# # SVM - Processing Time

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, SVR


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(Xpt,y_pt, test_size = 0.3, random_state = 65)


# In[ ]:


svc = SVC(degree = 2 )


# In[ ]:


svc.fit(X_train,y_train)


# In[ ]:


y_pred = svc.predict(X_test)


# In[ ]:


y_pred


# In[ ]:


from sklearn.metrics import mean_squared_error,r2_score


# In[ ]:


mean_squared_error(y_test, y_pred,squared=False) #RMSE 


# In[ ]:


mean_squared_error(y_test,y_pred,squared=True) #MSE


# In[ ]:


r2_score(y_test, y_pred)


# In[ ]:


df_plot = pd.DataFrame({'y_test': (y_test).to_list(), 'y_pred': y_pred}, index=y_test.index)
fig, axe = plt.subplots()
fig.suptitle("Scatter plot of actual and prediction.")
axe.scatter(x=df_plot.index.to_list(), y=df_plot['y_test'], label='Actual')
axe.scatter(x=df_plot.index.to_list(), y=df_plot['y_pred'], label='Predict')
axe.legend()
plt.show()


# # Random Forest - Processing Time

# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


rf = RandomForestRegressor(n_estimators = 100, max_features = 'sqrt', max_depth = 5, random_state = 18)


# In[ ]:


rf.fit(X_train,y_train)


# In[ ]:


y_pred = rf.predict(X_test)


# In[ ]:


mean_squared_error(y_test, y_pred,squared=False) #RMSE


# In[ ]:


mean_squared_error(y_test, y_pred,squared=True) #MSE


# In[ ]:


r2_score(y_test,y_pred)


# In[ ]:


df_plot = pd.DataFrame({'y_test': (y_test).to_list(), 'y_pred': y_pred}, index=y_test.index)
fig, axe = plt.subplots()
fig.suptitle("Scatter plot of actual and prediction.")
axe.scatter(x=df_plot.index.to_list(), y=df_plot['y_test'], label='Actual')
axe.scatter(x=df_plot.index.to_list(), y=df_plot['y_pred'], label='Predict')
axe.legend()
plt.show()


# In[ ]:


from sklearn.model_selection import GridSearchCV
grid = { 
    'n_estimators': [100,200,300,400,500],
    'max_features': ['sqrt','log2'],
    'max_depth' : [3,4,5,6,7],
    'random_state' : [18]
}

CV_rf = GridSearchCV(estimator=RandomForestRegressor(), param_grid=grid, cv= 5)
CV_rf.fit(X_train, y_train)



# In[ ]:


y_pred=CV_rf.predict(X_test)


# In[ ]:


mean_squared_error(y_test, y_pred,squared=False) #RMSE


# In[ ]:


mean_squared_error(y_test, y_pred,squared=True) #MSE


# In[ ]:


r2_score(y_test,y_pred)


# In[ ]:


df_plot = pd.DataFrame({'y_test': (y_test).to_list(), 'y_pred': y_pred}, index=y_test.index)
fig, axe = plt.subplots()
fig.suptitle("Scatter plot of actual and prediction.")
axe.scatter(x=df_plot.index.to_list(), y=df_plot['y_test'], label='Actual')
axe.scatter(x=df_plot.index.to_list(), y=df_plot['y_pred'], label='Predict')
axe.legend()
plt.show()


# # SVM - Average Power Consumption
# 
# 1.   List item
# 2.   List item
# 
# 

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(Xpwr,y_pwr, test_size = 0.3, random_state = 65)


# In[ ]:


svc = SVC(degree = 2 )


# In[ ]:


svc.fit(X_train,y_train)


# In[ ]:


y_pred = svc.predict(X_test)


# In[ ]:


mean_squared_error(y_test, y_pred,squared=False) #RMSE


# In[ ]:


mean_squared_error(y_test, y_pred,squared=True) #MSE


# In[ ]:


r2_score(y_test,y_pred)


# In[ ]:


df_plot = pd.DataFrame({'y_test': (y_test).to_list(), 'y_pred': y_pred}, index=y_test.index)
fig, axe = plt.subplots()
fig.suptitle("Scatter plot of actual and prediction.")
axe.scatter(x=df_plot.index.to_list(), y=df_plot['y_test'], label='Actual')
axe.scatter(x=df_plot.index.to_list(), y=df_plot['y_pred'], label='Predict')
axe.legend()
plt.show()


# # Random Forest - Average Power Consumption

# In[ ]:


rf = RandomForestRegressor(n_estimators = 100, max_features = 'sqrt', max_depth = 5, random_state = 18)
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)


# In[ ]:


y_pred = rf.predict(X_test)


# In[ ]:


mean_squared_error(y_test, y_pred,squared=False) #RMSE


# In[ ]:


mean_squared_error(y_test, y_pred,squared=True) #MSE


# In[ ]:


r2_score(y_test,y_pred)


# In[ ]:


df_plot = pd.DataFrame({'y_test': (y_test).to_list(), 'y_pred': y_pred}, index=y_test.index)
fig, axe = plt.subplots()
fig.suptitle("Scatter plot of actual and prediction.")
axe.scatter(x=df_plot.index.to_list(), y=df_plot['y_test'], label='Actual')
axe.scatter(x=df_plot.index.to_list(), y=df_plot['y_pred'], label='Predict')
axe.legend()
plt.show()


# In[ ]:


from sklearn.model_selection import GridSearchCV
grid = { 
    'n_estimators': [100,200,300,400,500],
    'max_features': ['sqrt','log2'],
    'max_depth' : [3,4,5,6,7],
    'random_state' : [18]
}

CV_rf = GridSearchCV(estimator=RandomForestRegressor(), param_grid=grid, cv= 5)
CV_rf.fit(X_train, y_train)



# In[ ]:


y_pred=CV_rf.predict(X_test)


# In[ ]:


mean_squared_error(y_test, y_pred,squared=False) #RMSE


# In[ ]:


mean_squared_error(y_test, y_pred,squared=True) #MSE


# In[ ]:


r2_score(y_test,y_pred)


# In[ ]:


df_plot = pd.DataFrame({'y_test': (y_test).to_list(), 'y_pred': y_pred}, index=y_test.index)
fig, axe = plt.subplots()
fig.suptitle("Scatter plot of actual and prediction.")
axe.scatter(x=df_plot.index.to_list(), y=df_plot['y_test'], label='Actual')
axe.scatter(x=df_plot.index.to_list(), y=df_plot['y_pred'], label='Predict')
axe.legend()
plt.show()


# # ------------------------------------------

# In[ ]:




