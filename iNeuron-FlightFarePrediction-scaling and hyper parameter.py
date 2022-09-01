import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

#Read Dataset
flight_train=pd.read_excel("D:\Flight-Fare\Data_Train.xlsx")
#Check column names and delete irrelevant columns
print(flight_train.columns)
flight_train.drop(['Route','Additional_Info'],axis=1,inplace=True)
#Check for NAN,if less then delete NAN
print(flight_train.isna().sum())
flight_train.dropna(inplace=True)
print(flight_train.isna().sum())
#Convert Dep_Time columns into hrs and mins
flight_train["Dep_Time_mins"]=pd.to_datetime(flight_train['Dep_Time']).dt.minute
flight_train["Dep_Time_hrs"]=pd.to_datetime(flight_train['Dep_Time']).dt.hour
flight_train.drop('Dep_Time',axis=1,inplace=True)
print(flight_train.columns)
#Convert Arrival_Time columns into hrs and mins
flight_train["Arrival_Time_mins"]=pd.to_datetime(flight_train['Arrival_Time']).dt.minute
flight_train["Arrival_Time_hrs"]=pd.to_datetime(flight_train['Arrival_Time']).dt.hour
flight_train.drop('Arrival_Time',axis=1,inplace=True)
print(flight_train.columns)
#Convert Date_of_Journey columns into date and month
flight_train['Journey_month']=pd.to_datetime(flight_train['Date_of_Journey'],format="%d/%m/%Y").dt.month
flight_train['Journey_day']=pd.to_datetime(flight_train['Date_of_Journey'],format="%d/%m/%Y").dt.day
flight_train.drop('Date_of_Journey',axis=1,inplace=True)
print(flight_train.columns)
#Replace Textual information from Total Stops
print(flight_train['Total_Stops'].value_counts())
replace_str={'non-stop' :0,
             '1 stop' :1,
             '2 stops':2,
             '3 stops':3,
             '4 stops':4
    }
flight_train['Total_Stops'].replace(replace_str,inplace=True)
print(flight_train['Total_Stops'].head(3))

#Separate duration 
duration=list(flight_train['Duration'])
for i in range(len(duration)):
    if (duration[i].split())!=2:
        if 'h' in duration[i]:
            duration[i]=duration[i]+' 0m'
        else:
            duration[i]="0h "+duration[i]

Duration_Hrs=[]
Duration_Min=[]
for i in range(len(duration)):
    Duration_Hrs.append(duration[i].split(sep="h")[0])
    Duration_Min.append(duration[i].split(sep='m')[0].split()[-1])
    
flight_train["Duration_Hrs"]=Duration_Hrs
flight_train['Duration_Min']=Duration_Min
flight_train.drop('Duration',axis=1,inplace=True)
        
    
#Handling Categorical Data
# print(flight_train['Airline'].value_counts(),flight_train['Source'].value_counts(),flight_train['Destination'].value_counts())
# #plot Airline vs Price
# sns.catplot(x='Airline',y='Price',data=flight_train.sort_values('Price',ascending=False),kind='boxen',height=6,aspect=5)
# #plot Source vs Price
# sns.catplot(x='Source',y='Price',data=flight_train.sort_values('Price',ascending=False),kind='boxen',height=6,aspect=6)
# #plot Destination vs Price
# sns.catplot(x='Destination',y='Price',data=flight_train.sort_values('Price',ascending=False),kind='boxen',height=6,aspect=6)
#Apply Encoding to Categorical data
print(flight_train.dtypes)
#convert Duration_Hrs and Duration_Min to int from object
flight_train['Duration_Hrs']= flight_train['Duration_Hrs'].astype(int)
flight_train['Duration_Min']= flight_train['Duration_Min'].astype(int)
print(flight_train.dtypes)

print(flight_train['Source'].unique())
print(flight_train['Destination'].unique())
print(flight_train['Airline'].unique())
flight_train['Destination'].replace({"New Delhi":"Delhi"},inplace=True)
print(flight_train['Destination'].unique())
print(flight_train.columns)
dfairline=pd.get_dummies(flight_train['Airline'],drop_first=True)
dfsource=pd.get_dummies(flight_train['Source'],drop_first=True)
dfdest=pd.get_dummies(flight_train['Destination'],drop_first=True)
flight_train.drop(['Airline','Source','Destination'],axis=1,inplace=True)
flight_train=pd.concat([dfairline,dfsource,dfdest,flight_train],axis=1)
flight_train.drop("Trujet",axis=1,inplace=True)


#Test Data
flight_test=pd.read_excel("D:\Flight-Fare\Test_set.xlsx")
#Check column names and delete irrelevant columns
print(flight_test.columns)
flight_test.drop(['Route','Additional_Info'],axis=1,inplace=True)
#Check for NAN,if less then delete NAN
print(flight_test.isna().sum())
flight_test.dropna(inplace=True)
print(flight_test.isna().sum())
#Convert Dep_Time columns into hrs and mins
flight_test["Dep_Time_mins"]=pd.to_datetime(flight_test['Dep_Time']).dt.minute
flight_test["Dep_Time_hrs"]=pd.to_datetime(flight_test['Dep_Time']).dt.hour
flight_test.drop('Dep_Time',axis=1,inplace=True)
print(flight_test.columns)
#Convert Arrival_Time columns into hrs and mins
flight_test["Arrival_Time_mins"]=pd.to_datetime(flight_test['Arrival_Time']).dt.minute
flight_test["Arrival_Time_hrs"]=pd.to_datetime(flight_test['Arrival_Time']).dt.hour
flight_test.drop('Arrival_Time',axis=1,inplace=True)
print(flight_test.columns)
#Convert Date_of_Journey columns into date and month
flight_test['Journey_month']=pd.to_datetime(flight_test['Date_of_Journey'],format="%d/%m/%Y").dt.month
flight_test['Journey_day']=pd.to_datetime(flight_test['Date_of_Journey'],format="%d/%m/%Y").dt.day
flight_test.drop('Date_of_Journey',axis=1,inplace=True)
print(flight_test.columns)
#Replace Textual information from Total Stops
flight_test['Total_Stops'].replace(replace_str,inplace=True)
print(flight_test['Total_Stops'].head(3))
#Separate duration 
duration=list(flight_test['Duration'])
for i in range(len(duration)):
    if (duration[i].split())!=2:
        if 'h' in duration[i]:
            duration[i]=duration[i]+' 0m'
        else:
            duration[i]="0h "+duration[i]

Duration_Hrs=[]
Duration_Min=[]
for i in range(len(duration)):
    Duration_Hrs.append(duration[i].split(sep="h")[0])
    Duration_Min.append(duration[i].split(sep='m')[0].split()[-1])
    
flight_test["Duration_Hrs"]=Duration_Hrs
flight_test['Duration_Min']=Duration_Min
flight_test.drop('Duration',axis=1,inplace=True)
#Handling Categorical Data
#Apply Encoding to Categorical data
print(flight_test.dtypes)
#convert Duration_Hrs and Duration_Min to int from object
flight_test['Duration_Hrs']= flight_test['Duration_Hrs'].astype(int)
flight_test['Duration_Min']= flight_test['Duration_Min'].astype(int)
print(flight_test.dtypes)
print(flight_test['Source'].unique())
print(flight_test['Destination'].unique())
print(flight_test['Airline'].unique())
flight_test['Destination'].replace({"New Delhi":"Delhi"},inplace=True)
print(flight_test['Destination'].unique())
dfairline=pd.get_dummies(flight_test['Airline'],drop_first=True)
dfsource=pd.get_dummies(flight_test['Source'],drop_first=True)
dfdest=pd.get_dummies(flight_test['Destination'],drop_first=True)
flight_test.drop(['Airline','Source','Destination'],axis=1,inplace=True)
flight_test=pd.concat([dfairline,dfsource,dfdest,flight_test],axis=1)


# Separate data into independent and dependent variable
X=flight_train.drop('Price',axis=1)
y=flight_train['Price']

# #Scaling of Data
# from sklearn.preprocessing import StandardScaler
# sc=StandardScaler()
# sc.fit_transform(X)


from sklearn.model_selection import train_test_split
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2)



from sklearn.ensemble import ExtraTreesRegressor
etr=ExtraTreesRegressor()
etr.fit(Xtrain,ytrain)
print(etr.feature_importances_)

from sklearn.ensemble import RandomForestRegressor
rfr=RandomForestRegressor()
rfr.fit(Xtrain,ytrain)
y_pred=rfr.predict(Xtest)
print(rfr.score(Xtrain,ytrain))
print(rfr.score(Xtest,ytest))

#METRICS
from sklearn import metrics
print('MAE:',metrics.mean_absolute_error(ytest, y_pred))
print('MSE:',metrics.mean_squared_error(ytest, y_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(ytest, y_pred)))
print(metrics.r2_score(ytest, y_pred))

#HYPER PARAMETER TUNING
from sklearn.model_selection import RandomizedSearchCV
#number of trees in Random Forest
n_estimators=[int(x) for x in np.linspace(start=100, stop=1200,num=12)]
#Number of features to consider at evey spli
max_fetaures=['auto','sqrt']
#Maximum number of levels in tree
max_depth=[int(x) for x in np.linspace(start=5, stop=30,num=6)]
#minimum number of samples required to split a node
min_samples_split=[2,5,10,15,100]
min_samples_leaf=[1,2,5,10]

#create a random grid

random_grid={
    "n_estimators":n_estimators,
    "max_features":max_fetaures,
    "max_depth":max_depth,
    'min_samples_split':min_samples_split,
    'min_samples_leaf':min_samples_leaf}

rf_random=RandomizedSearchCV(estimator=rfr, param_distributions=random_grid,scoring='neg_mean_squared_error',n_iter=10,cv=5,verbose=2,random_state=42,n_jobs=1)
rf_random.fit(Xtrain,ytrain)

print(rf_random.best_estimator_)     
prediction=rf_random.predict(Xtest)                        
# =============================================================================
# data=flight_test[0:1]
# print(data)
# print(rfr.predict(data))
# =============================================================================

# print(flight_train.columns)
# print(flight_test.columns)
#print(flight_train.shape,flight_test.shape)
pd.set_option('display.max_columns', None)
#Prediction
data=flight_test.loc[2600:2600]
print(data)
print(rfr.predict(data))
print(rf_random.predict(data))