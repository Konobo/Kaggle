import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
import math


class PreprocessData:

    def preprocess_data(self, path):
        data_raw = pd.read_csv(path, sep=',')
        data_raw = data_raw.set_index('PassengerId')
        # We scale Age and Fare
        data_temp = data_raw[['Age', 'Fare']].copy()
        data_temp = preprocessing.scale(data_temp)
        data_raw[['Age', 'Fare']] = data_temp.copy()
        # We work a bit on the data, because we think that Name, Ticket are not playing a role
        # data_raw = data_raw.drop(['Name', 'Ticket'], axis='columns')
        # data_raw = data_raw.drop(['Cabin'], axis='columns')
        # For Embarked there is a class with much more candidates, then the NaN belongs probably in this class
        number_pers_embarked = data_raw.groupby(['Embarked']).count()

        data_raw.loc[:, 'Embarked'].fillna(number_pers_embarked[number_pers_embarked['Pclass'] ==
                                                                np.max(number_pers_embarked)[0]].index[0], inplace=True)
        # For Sex we change it in classes
        le = preprocessing.LabelEncoder()
        data_raw['Sex'] = le.fit_transform(data_raw['Sex'])
        data_raw['Embarked'] = le.fit_transform(data_raw['Embarked'])
        # data_process_4 = data_raw.copy()
        # mean_acc_pclass = data_raw.groupby(['Pclass'])['Age'].mean()
        # for ind in data_process_4.index:
        #     if math.isnan(data_process_4['Age'][ind]):
        #         data_process_4.loc[ind, 'Age'] = mean_acc_pclass[data_process_4['Pclass'][ind]]
        return data_raw

    def fill_age(self, dataset):
        #On remplit d'abord l'âge des employés
        employee = dataset.Fare == 0
        dataset.loc[employee, 'Age'] = dataset.loc[employee, 'Age'].fillna(np.mean(dataset.loc[employee, 'Age']))
        #On remplit ensuite selon le titre si c'est justifié
        after_coma = dataset.loc[:,'Name'].str.split(', ', expand=True)[1]
        title = after_coma.str.split(' ', expand=True)[0]
        list_title = title.unique()
        newdataset = dataset.copy()
        newdataset['Title'] = title
        print(newdataset.groupby(['Title']).Name.count())
        thresh_freq = 5
        for cur_title in list_title:
            cur_select = newdataset.Title == cur_title
            if dataset.loc[cur_select, 'Age'].count() > thresh_freq:
                dataset.loc[cur_select, 'Age'] = dataset.loc[cur_select, 'Age'].fillna(np.mean(dataset.loc[cur_select,
                                                                                                       'Age']))
        #Si le titre n'est pas suffisant, on remplit selon la classe
        class1 = dataset.Pclass == 1
        class2 = dataset.Pclass == 2
        class3 = dataset.Pclass == 3
        sublist = [class1, class2, class3]
        for select in sublist:
            dataset.loc[select, 'Age'] = dataset.loc[select, 'Age'].fillna(np.mean(dataset.loc[select, 'Age']))
        return dataset

path = 'train.csv'

data_raw = pd.read_csv(path, sep=',')

data_raw = data_raw.set_index('PassengerId')
data_real_raw = data_raw.copy()
prep = PreprocessData()
filledAge = prep.fill_age(data_real_raw)
# We scaled Age and Fare
data_temp = data_raw[['Age', 'Fare']].copy()
data_temp = preprocessing.scale(data_temp)
data_raw[['Age', 'Fare']] = data_temp.copy()
# We work a bit on the data, because we think that Name, Ticket are not playing a role
# In fact they are "LINE" with FARE = 0 means that they are employee
data_raw = data_raw.drop(['Name', 'Ticket'], axis='columns')
data_raw = data_raw.drop(['Cabin'], axis='columns')
# For Embarked there is a class with much more candidates, then the NaN belongs probably in this class
number_pers_embarked = data_raw.groupby(['Embarked']).count()

data_raw.loc[:, 'Embarked'].fillna(number_pers_embarked[number_pers_embarked['Survived'] ==
                                                        np.max(number_pers_embarked)[0]].index[0], inplace=True)
# For Sex we change it in classes
le = preprocessing.LabelEncoder()
data_raw['Sex'] = le.fit_transform(data_raw['Sex'])
data_raw['Embarked'] = le.fit_transform(data_raw['Embarked'])
correlationMatrix = data_raw.corr()
# Cross validation on what to do with NaN
data_process_1 = data_raw.dropna(subset=['Age'],inplace=False)
data_process_2 = data_raw.copy()
data_process_2['Age'] = data_process_2['Age'].fillna(np.median(data_process_1['Age']))
data_process_3 = data_raw.copy()
data_process_3['Age'] = data_process_3['Age'].fillna(np.mean(data_process_1['Age']))
data_process_4 = data_raw.copy()
mean_acc_Pclass = data_raw.groupby(['Pclass'])['Age'].mean()
for ind in data_process_4.index:
    if math.isnan(data_process_4['Age'][ind]):
        data_process_4.loc[ind, 'Age'] = mean_acc_Pclass[data_process_4['Pclass'][ind]]
# list_data_train = [data_process_1, data_process_2, data_process_3, data_process_4]
# score=[]
# std = []
# kf = KFold(n_splits=10)
# for process in list_data_train:
#     XY = process.values
#     Y = XY[:, 0]
#     X = np.delete(XY, 0, 1)
#     score_temp = []
#     for train, test in kf.split(X):
#         Xtrain = X[train, :]
#         ytrain = Y[train]
#         Xtest = X[test, :]
#         ytest = Y[test]
#         gaussian_process.fit(Xtrain, ytrain)
#         score_temp += [gaussian_process.score(Xtest, ytest)]
#     score += [np.mean(score_temp)]
#     std += [np.std(score_temp)]
# print(score)
# print(std)

data_train, data_test_n = train_test_split(data_process_4, test_size=0.2, random_state=40, shuffle=True)

X_train = data_train.iloc[:, 1:data_train.shape[1]]
y_train = data_train.iloc[:, 0]
X_test = data_test_n.iloc[:, 1:data_test_n.shape[1]]
y_test = data_test_n.iloc[:, 0]

mySVC = svm.SVC(kernel='linear', C=1)
mySVC.fit(X_train, y_train)
print(mySVC.score(X_test, y_test))

path = 'test.csv'
data_test = prep.preprocess_data(path)
data_test = prep.fill_age(data_test)
mean_acc_Pclass_fare= data_test.groupby(['Pclass'])['Fare'].mean()
for ind in data_test.index:
    if math.isnan(data_test['Fare'][ind]):
        data_test.loc[ind, 'Fare'] = mean_acc_Pclass_fare[data_test['Pclass'][ind]]

ypred = mySVC.predict(data_test.iloc[:, 0:data_test.shape[1]])
my_submission = pd.DataFrame(data={'PassengerId':data_test.index, 'Survived':ypred})

print(my_submission['Survived'].value_counts())

#Export Results to CSV
my_submission.to_csv('submission.csv', index=False)








# tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
#                      'C': [1, 10, 100, 1000]},
#                     {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
# scores = ['precision', 'recall']
#
# for score in scores:
#     print("# Tuning hyper-parameters for %s" % score)
#     print()
#
#     clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=2,
#                        scoring='%s_macro' % score)
#     clf.fit(X_train, y_train)
#
#     print("Best parameters set found on development set:")
#     print()
#     print(clf.best_params_)
#     print()
#     print("Grid scores on development set:")
#     print()
#     means = clf.cv_results_['mean_test_score']
#     stds = clf.cv_results_['std_test_score']
#     for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#         print("%0.3f (+/-%0.03f) for %r"
#               % (mean, std * 2, params))
#     print()
#
#     print("Detailed classification report:")
#     print()
#     print("The model is trained on the full development set.")
#     print("The scores are computed on the full evaluation set.")
#     print()
#     y_true, y_pred = y_test, clf.predict(X_test)
#     print(classification_report(y_true, y_pred))
#     print()
#

#  We can just replace the na of Age by the mean
# We can adjust this mean according to the features

# X_train = data_train.iloc[:, 1:data_train.shape[1]]
# y_train = data_train.iloc[:, 0]
# X_test = data_test_n.iloc[:, 1:data_test_n.shape[1]]
# y_test = data_test_n.iloc[:, 0]

