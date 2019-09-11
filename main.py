import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
import matplotlib.pyplot       

#importing the dataset
data = pd.read_csv('matches.csv')
data1 = data

#checking if there are any missing data
miss_bool = data['winner'].isnull() 
print(sum(miss_bool))

#dropping those rows with NULL values
data = data.dropna(how = 'all')


#dropping thoses columns which arent making sense
data = data.drop(['dl_applied','umpire1','id','umpire2','umpire3','date','win_by_runs',
                  'win_by_wickets','player_of_match','result','city','season','venue'], axis=1)

#replace

data = data.replace('Rising Pune Supergiants', 'Rising Pune Supergiant')

#encoding the values using label encoder
le = LabelEncoder()

cols = ['team1','team2','winner','toss_winner','toss_decision']

for i in cols:
    data[i] = le.fit_transform(data[i])

#spilttinig into X and y
x = data.drop(['winner'],axis=1)
y = data['winner']

teams = ['Royal Challengers Bangalore',
'Sunrisers Hyderabad',
'Rajasthan Royals',
'Mumbai Indians',
'Kolkata Knight Riders',
'Kings XI Punjab',
'Delhi Capitals',
'Chennai Super Kings',
'Gujarat Lions',
'Rising Pune Supergiant',
'Pune Warriors India',
'Kochi Tuskers Kerala',
'Deccan Chargers']

teams.sort()
teams_le = le.fit_transform(teams)

#Test Teain Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.30, random_state=1008)

#--------------------------------------SVM
'''
svm = svm.SVC(gamma = 'scale')
svm.fit(x_train, y_train)
y_pred = svm.predict(x_test)

print(f1_score(y_test,y_pred, average='micro'))

#--------------------------------------Random Forest
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)
y_pred1 = classifier.predict(x_test)

print(f1_score(y_test,y_pred1,average='micro'))
'''
#--------------------------------------XGBoosr
xg = XGBClassifier()
xg.fit(x_train, y_train)
y_pred2 = xg.predict(x_test)

print("The F1 Score of the XGBoost is: ",f1_score(y_test,y_pred2,average='micro'))

import pickle
with open('model.pickle','wb') as k:
    pickle.dump(xg,k)

