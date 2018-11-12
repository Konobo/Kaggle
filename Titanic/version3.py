import pandas as pd
import numpy as np
from sklearn import preprocessing
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
import keras.losses as losses
import keras.optimizers as optimizers
import keras.metrics as metrics
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score
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
        self.fill_age(data_raw)
        self.create_family(data_raw)
        mean_acc_Pclass_fare = data_raw.groupby(['Pclass'])['Fare'].mean()
        for ind in data_raw.index:
            if math.isnan(data_raw['Fare'][ind]):
                data_raw.loc[ind, 'Fare'] = mean_acc_Pclass_fare[data_raw['Pclass'][ind]]
        number_pers_embarked = data_raw.groupby(['Embarked']).count()
        data_raw.loc[:, 'Embarked'].fillna(number_pers_embarked[number_pers_embarked['Name'] ==
                                                                np.max(number_pers_embarked)[0]].index[0], inplace=True)
        to_drop =['Cabin', 'Name', 'Ticket']
        data_raw.drop(to_drop, axis='columns', inplace=True)
        le = preprocessing.LabelEncoder()
        data_raw['Sex'] = le.fit_transform(data_raw['Sex'])
        one_hot = pd.get_dummies(data_raw['Embarked'])
        data_raw = data_raw.drop(['Embarked'], axis=1)
        data_raw = data_raw.join(one_hot)
        data_raw.drop(columns='Family')
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
    def create_family(self, dataset):
        dataset['Family'] = dataset.SibSp + dataset.Parch
        dataset.drop(['SibSp', 'Parch'], axis='columns', inplace=True)
        return dataset

# ############# ENTRAINEMENT #####################
#to_be_trained, data_test_n = train_test_split(data_train, test_size=0.2, random_state=40, shuffle=True)
# X_train = to_be_trained.iloc[:, 1:to_be_trained.shape[1]]
# y_train = to_be_trained.iloc[:, 0]
# X_test = data_test_n.iloc[:, 1:data_test_n.shape[1]]
# y_test = data_test_n.iloc[:, 0]
# tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-1, 1e-2],
#                      'C': [1, 10, 100]},
#                     {'kernel': ['linear'], 'C': [1, 10]}]
# scores = ['precision', 'recall']
#
# for score in scores:
#     print("# Tuning hyper-parameters for %s" % score)
#     print()
#     clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=2,
#                        scoring='%s_macro' % score)
#     clf.fit(X_train, y_train)
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
#     print("Detailed classification report:")
#     print()
#     print("The model is trained on the full development set.")
#     print("The scores are computed on the full evaluation set.")
#     print()
#     y_true, y_pred = y_test, clf.predict(X_test)
#     print(classification_report(y_true, y_pred))
#     print()


path_train = 'train.csv'
path_test = 'test.csv'
prep = PreprocessData()
data_train = prep.preprocess_data(path_train)
data_test = prep.preprocess_data(path_test)
to_be_trained, data_test_n = train_test_split(data_train, test_size=0.2, random_state=40, shuffle=True)
X_train = to_be_trained.iloc[:, 1:to_be_trained.shape[1]]
y_train = to_be_trained.iloc[:, 0]
X_test = data_test_n.iloc[:, 1:data_test_n.shape[1]]
y_test = data_test_n.iloc[:, 0]


################ XGBoost ##########################
#
# my_model = XGBClassifier(n_estimators=1000, learning_rate=0.05)
# my_model.fit(X_train, y_train, early_stopping_rounds=5,
#              eval_set=[(X_test, y_test)], verbose=True)
# predictions = my_model.predict(X_test)
# print("Mean Absolute Error : " + str(mean_absolute_error(predictions, y_test)))
# print("Accuracy score : " + str(accuracy_score(predictions, y_test)))
#
# ############### SVC ##########################
# #
# mySVC = svm.SVC(kernel='linear', C=10.0)
# mySVC.fit(X_train, y_train)
# predictions = mySVC.predict(X_test)
# print("Mean Absolute Error : " + str(mean_absolute_error(predictions, y_test)))
# print("Accuracy score : " + str(accuracy_score(predictions, y_test)))

# ############### Neural Networks ##########################
# myNN = MLPClassifier(hidden_layer_sizes=(10))
# myNN.fit(X_train, y_train)
# predictions = myNN.predict(data_test)
# print("Accuracy score : " + str(myNN.score(X_test,y_test)))
# ############### Deep Networks ##########################
# model_nn = Sequential()
# model_nn.add(Dense(units=5, input_dim=X_train.shape[1]))
# model_nn.add(Activation('relu'))
# model_nn.add(Dense(units=5))
# model_nn.add(Activation('relu'))
# model_nn.add(Dense(units=2))
# model_nn.add(Activation('softmax'))
# y_train_bin = to_categorical(y_train)
# y_test_bin = to_categorical(y_test)
# model_nn.compile(loss=losses.categorical_crossentropy,
#               optimizer=optimizers.SGD(lr=0.001, decay=0., nesterov=False),
#               metrics=[metrics.categorical_accuracy])
# history =model_nn.fit(X_train, y_train_bin, validation_data=(X_test,y_test_bin), epochs=300, batch_size=64)
# plt.figure(figsize=(20,10))
#
# for i, (name, values) in enumerate(sorted(history.history.items())):
#     plt.subplot(1, len(history.history.items()), i+1)
#     plt.plot(values)
#     plt.title(name)
#
# plt.show()
# ############### Best results ########################## 258-160
#  my_model = XGBClassifier(n_estimators=1000, learning_rate=0.05)
# my_model.fit(X_train, y_train, early_stopping_rounds=5,
#              eval_set=[(X_test, y_test)], verbose=True)
# predictions = my_model.predict(data_test)

# my_submission = pd.DataFrame(data={'PassengerId':data_test.index, 'Survived':predictions})
#
# print(my_submission['Survived'].value_counts())
#
# #Export Results to CSV
# my_submission.to_csv('submission.csv', index=False)