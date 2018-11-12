import pandas as pd
import numpy as np
from preprocess_function import preprocess_data
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.linear_model import LogisticRegression
path = 'train.csv'

data_raw = pd.read_csv(path, sep=',')
data_raw = data_raw.set_index('PassengerId')

data_train, data_test_n = train_test_split(data_raw, test_size=0.2, random_state=40, shuffle=True)
# Indices of dataset are now the Ids of the passenger
list_col_names = list(data_train.columns)
for col in list_col_names:
    print(col + str(data_train[col].isnull().sum()))
# We remove column Cabin because there is a lot of missing data
col_to_delete = ['Name', 'Ticket', 'Cabin']
data_train = data_train.drop(col_to_delete, axis=1)
list_col_names = list(data_train.columns)
# Dropped columns that seem to be useless prior
data_train = data_train.dropna()
# Traitement des données manquantes
le = preprocessing.LabelEncoder()
list_col_names.pop(0)
for col in list_col_names:
    if col in ['Sex', 'Cabin', 'Embarked', 'Survived', 'Pclass', 'Sex', 'SibSp', 'Parch']:
        data_train[col] = le.fit_transform(data_train[col])

data_train['Age'] = (data_train['Age']-np.mean(data_train['Age']))/(np.std(data_train['Age']))
data_train['Fare'] = (data_train['Fare']-np.mean(data_train['Fare']))/(np.std(data_train['Fare']))

X = data_train.iloc[:, 1:data_train.shape[1]]
y = data_train.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40, shuffle=True)

mySVC = svm.SVC(kernel='linear')

mySVC.fit(X_train,y_train)

print('Avec SVM Linear le score est :' + str(mySVC.score(X_test, y_test)))


# logistic_regressor = LogisticRegression(penalty='l1', C=1.0, solver='liblinear')
logistic_regressor = GaussianProcessClassifier()
logistic_regressor.fit(X_train, y_train)
print('Avec la régression logistique le score est :' + str(logistic_regressor.score(X_test, y_test)))




# With test data
data_test = preprocess_data('test.csv')
list_col_names = list(data_test.columns)
for col in list_col_names:
    print(col + str(data_test[col].isnull().sum()))
data_test['Age'] = data_test['Age'].fillna(np.mean(data_test['Age']))
data_test['Fare'] = data_test['Fare'].fillna(np.mean(data_test['Fare']))
Xpred = data_test.iloc[:, 0:data_test.shape[1]]

ypred = logistic_regressor.predict(Xpred)
my_submission = pd.DataFrame(data={'PassengerId':data_test.index, 'Survived':ypred})

print(my_submission['Survived'].value_counts())

#Export Results to CSV
my_submission.to_csv('submission.csv', index = False)