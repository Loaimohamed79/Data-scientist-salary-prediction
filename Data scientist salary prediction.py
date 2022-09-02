# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 12:33:27 2022

@author: Loai
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('L:\\Some Models\Data Science Job Salaries\\ds_salaries.csv')

HeadData = df.head()
TailData = df.tail()

df.nunique()
df.info()
df.duplicated().sum()
descrip = df.describe()

df['job_title'].value_counts()

df.columns

dataset=df.query('job_title =="Data Scientist" |job_title=="Machine Learning Engineer"|job_title =="Data Analyst"|job_title=="Data Engineer"')


plt.figure(figsize=(15, 7))
sns.barplot(data=dataset, x='experience_level', y='salary_in_usd', order=['EN', 'MI', 'SE', 'EX'], hue='job_title')
plt.title('Salary to Expirience Level for different roles')
plt.show()

sns.pointplot(y=df['salary_in_usd']/10, x=df['company_size'])
plt.ylabel("Salary / 10")

sns.kdeplot(df['salary_in_usd'])

sns.boxplot(df['company_size'], df['salary_in_usd'])

sns.histplot(df['salary_in_usd'])

OutLayerSalary = df[df.salary_in_usd > 500000]


CountEveryLevelInExperienceLevel = df['experience_level'].value_counts()

sns.countplot(df['experience_level'])

ExperienceLevel_CompanySize = df.groupby(
    'experience_level').company_size.value_counts()

df.groupby('experience_level').company_size.value_counts().plot(kind='bar')

sns.countplot(df['work_year'])

df['employee_residence'].value_counts()

df_GB = df[df['employee_residence'] == 'GB']
df_US = df[(df['employee_residence'] == 'US')]
df_GB_US =df.query('employee_residence == "US" | employee_residence == "GB"')

df_GB_US.shape
df_GB.shape
df_US.shape





df_data_scientist = df.query('job_title == "Data Scientist"')

X=df_data_scientist.iloc[:,1:-1]

X = X.drop(columns=['salary','salary_currency','salary_in_usd','job_title'])

Y=df_data_scientist['salary_in_usd']

X.columns
X=pd.get_dummies(X,columns=
                 ['experience_level','employment_type',
                  'employee_residence','company_location'],drop_first=True)

df['employment_type'].value_counts()

X.shape
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test =train_test_split(X,Y,test_size=0.2,random_state=42)


"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
X[ : ,-1] = labelEncoder_X.fit_transform(X[ : , -1])

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

X=X[:,1:]

"""



"""
from sklearn.neighbors import KNeighborsClassifier
Knn = KNeighborsClassifier(n_neighbors=5)
Knn.fit(X_train,Y_train)


y_pred = Knn.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

confknn=confusion_matrix(_Y_test, y_pred)
print("Accuracy: ", accuracy_score(y_pred, _Y_test)*100)
classifireport=classification_report(y_test, y_pred)
"""

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)


y_pred= regressor.predict(X_test)
np.set_printoptions(precision=2)


