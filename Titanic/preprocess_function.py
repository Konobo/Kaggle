import pandas as pd
import numpy as np
from sklearn import preprocessing
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
        data_process_4 = data_raw.copy()
        mean_acc_pclass = data_raw.groupby(['Pclass'])['Age'].mean()
        for ind in data_process_4.index:
            if math.isnan(data_process_4['Age'][ind]):
                data_process_4.loc[ind, 'Age'] = mean_acc_pclass[data_process_4['Pclass'][ind]]
        return data_process_4

    def fill_age(self, dataset):
        employee = dataset[dataset.Fare == 0].copy()
        employee.loc[:, 'Age'].fillna(np.mean(employee.loc[:, 'Age']))
        return employee
