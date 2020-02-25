import re
import time
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import mean_absolute_error,mean_squared_error

# Loading the data
df = pd.read_csv('E:/College/Analytics/Python/googleplaystore.csv')
df.info()
df.isnull().sum()
df.head()
# The best way to fill missing values might be using the median instead of mean.
df['Rating'] = df['Rating'].fillna(df['Rating'].median())

df['Current Ver'].unique().tolist()
replaces = [u'\u00AE', u'\u2013', u'\u00C3', u'\u00E3', u'\u00B3', '[', ']', "'"]
for i in replaces:
	df['Current Ver'] = df['Current Ver'].astype(str).apply(lambda x : x.replace(i, ''))

regex = [r'[-+|/:/;(_)@]', r'\s+', r'[A-Za-z]+']
for j in regex:
	df['Current Ver'] = df['Current Ver'].astype(str).apply(lambda x : re.sub(j, '0', x))

df['Current Ver'] = df['Current Ver'].astype(str).apply(lambda x : x.replace('.', ',',1).replace('.', '').replace(',', '.',1)).astype(float)

df['Current Ver'] = df['Current Ver'].fillna(df['Current Ver'].median())

i = df[df['Category'] == '1.9'].index
df.loc[i]
df = df.drop(i)
df = df[pd.notnull(df['Last Updated'])]
df = df[pd.notnull(df['Content Rating'])]

df.groupby('Category',as_index=False)['Rating'].mean()
df.groupby('Reviews',as_index=False)['Rating'].mean()
df.groupby('Genres',as_index=False)['Rating'].mean()
df.groupby('Type',as_index=False)['Rating'].mean()

sns.countplot(x='Type',data=df)
sns.barplot(x='Type', y='Rating', data=df)
plt.figure(figsize=(16,8))
sns.countplot(y='Category',data=df)
plt.show()
plt.figure(figsize=(16,8))
sns.barplot(y='Category', x='Rating', data=df)
sns.barplot(y='Content Rating', x='Content Rating', data=df)

# Encoding
le = preprocessing.LabelEncoder()
df['App'] = le.fit_transform(df['App'])
df['Genres'] = le.fit_transform(df['Genres'])
df['Content Rating'] = le.fit_transform(df['Content Rating'])

# Category features encoding
category_list = df['Category'].unique().tolist() 
category_list = ['cat_' + word for word in category_list]
df = pd.concat([df, pd.get_dummies(df['Category'], prefix='cat')], axis=1)

# Price cealning
df['Price'] = df['Price'].apply(lambda x : x.strip('$'))
# Installs cealning
df['Installs'] = df['Installs'].apply(lambda x : x.strip('+').replace(',', ''))
# Type encoding
df['Type'] = pd.get_dummies(df['Type'])

# Last Updated encoding
df['Last Updated'] = df['Last Updated'].apply(lambda x : time.mktime(datetime.datetime.strptime(x, '%B %d, %Y').timetuple()))

# Convert kbytes to Mbytes 
df['Size'].unique().tolist()

k_indices = df['Size'].loc[df['Size'].str.contains('k')].index.tolist()
converter = pd.DataFrame(df.loc[k_indices, 'Size'].apply(lambda x: x.strip('k')).astype(float).apply(lambda x: x / 1024).apply(lambda x: round(x, 3)).astype(str))
df.loc[k_indices,'Size'] = converter
# Size cleaning
df['Size'] = df['Size'].apply(lambda x: x.strip('M'))
df[df['Size'] == 'Varies with device'] = 0
df['Size'] = df['Size'].astype(float)

df.corr()['Rating'].sort_values()

features = ['App','Reviews', 'Size', 'Installs', 'Type', 'Price', 'Content Rating', 'Genres', 'Last Updated', 'Current Ver']
features.extend(category_list)
X = df[features]
y = df['Rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 10)

lr= LinearRegression()
lr.fit(X_train, y_train)
print(lr.intercept_)
lr_pred = lr.predict(X_test)
accuracy = lr.score(X_test,y_test)
'Accuracy: ' + str(np.round(accuracy*100, 2)) + '%'
mean_absolute_error(y_test, lr_pred)
mean_squared_error(y_test, lr_pred)
np.sqrt(mean_squared_error(y_test, lr_pred))

cross_val_lr = cross_val_score(estimator = LinearRegression(), X = X_train, y = y_train, cv = 10, n_jobs = -1)
print("Cross Validation Accuracy : ",round(cross_val_lr.mean() * 100 , 2),"%")

k_range= range(3,31)
accuracy_list = []

for k in k_range:
    knn= KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)
    accuracy_list.append(knn.score(X_test,y_test))

print(accuracy_list)

model = KNeighborsRegressor(n_neighbors=25)
model.fit(X_train, y_train)

kn_pred = model.predict(X_test)

accuracy = model.score(X_test,y_test)
'Accuracy: ' + str(np.round(accuracy*100, 2)) + '%'
mean_absolute_error(y_test, kn_pred)
mean_squared_error(y_test, kn_pred)

cross_val_knn = cross_val_score(estimator = KNeighborsRegressor(n_neighbors=15), X = X_train, y = y_train, cv = 10, n_jobs = -1)
print("Cross Validation Accuracy : ",round(cross_val_knn.mean() * 100 , 2),"%")


rf = RandomForestRegressor(n_jobs=-1)

estimators = np.arange(10, 200, 10)
scores = []
for n in estimators:
    rf.set_params(n_estimators=n)
    rf.fit(X_train, y_train)
    scores.append(rf.score(X_test, y_test))

plt.figure(figsize=(7, 5))
plt.title("Effect of Estimators")
plt.xlabel("no. estimator")
plt.ylabel("score")
plt.plot(estimators, scores)
results = list(zip(estimators,scores))
results

predictions = rf.predict(X_test)
'Mean Absolute Error:', metrics.mean_absolute_error(y_test, predictions)
'Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions))

accuracy = rf.score(X_test,y_test)
'Accuracy: ' + str(np.round(accuracy*100, 2)) + '%'

