# data analysis and wrangling
import numpy as np
import pandas as pd 
import random as rnd

# visualization
import matplotlib.pyplot as plt
import seaborn as sns

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

# aquire data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
combine_df = [train_df, test_df]
#print train_df.describe(include=['O'])
# print combine_df

# analyze by pivoting features
#print train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(
#	by='Survived', ascending=False)
#print train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(
#	by='Survived', ascending=False)
#print train_df[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(
#	by='Survived', ascending=False)
#print train_df[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(
#	by='Survived', ascending=False)
#print train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(
#	by='Survived', ascending=False)

# numerical correlation: age vs. survival
g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)

# numerical and ordinal correlation: age and Pclass vs. survival
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass')
grid.map(plt.hist, 'Age', alpha=1, bins=20)
grid.add_legend()

# categorical correlation: embarkment and sex vs. survival
grid2 = sns.FacetGrid(train_df, row='Embarked')
grid2.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid2.add_legend()

# categorical and numerical correlation: fare vs. survival  
grid3 = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid3.map(sns.barplot, 'Sex', 'Fare', alpha=0.7, ci=None)
grid3.add_legend()

# visualization
#plt.show()

# wrangling data: correcting by dropping features (ticket and cabin numbers)
# print("Before", train_df.shape, test_df.shape, combine_df[0].shape, combine_df[1].shape)
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine_df = [train_df, test_df]
# print ("After", train_df.shape, test_df.shape, combine_df[0].shape, combine_df[1].shape)

# creating new features extracting from existing: names and titles 
for dataset in combine_df:
	# looks for any word that ends with a period, aka title, 
	# and creates a new column called 'Title'
	dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

# print pd.crosstab(train_df['Title'], train_df['Sex'])

for dataset in combine_df:
	dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
 	dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
 	dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
 	dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

# print train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

title_mapping ={"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine_df:
	dataset['Title'] = dataset['Title'].map(title_mapping)
	dataset['Title'] = dataset['Title'].fillna(0)

# print train_df.head()

# drop Name and PassengerId feature
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine_df = [train_df, test_df]

# converting a categorical feature (sex to ordinal)
sex_mapping = {"female": 1, "male": 0}
for dataset in combine_df:
	dataset['Sex'] = dataset['Sex'].map(sex_mapping)

# completing a numerical continuous feature
grid4 = sns.FacetGrid(train_df, col='Sex', row='Pclass')
grid4.map(plt.hist, 'Age')
grid.add_legend
# plt.show()

guess_ages = np.zeros((2,3))

for dataset in combine_df:
	for i in range(0,2):
		for j in range(0,3):
			guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j+1)]['Age'].dropna()
			age_guess = guess_df.median()
			# convert the guessed age float to nearest 0.5 years
			guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
	# now fill the empty/NaN age columns with guessed ages (done within the main loop!)
	for i in range(0,2):
		for j in range(0,3):
			dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex ==i) & (dataset.Pclass == j+1),\
			'Age'] = guess_ages[i,j]

# now convert age (numerical feature) to age band (ordinal feature) for quicker analysis
train_df['AgeBand'] = pd.cut(train_df['Age'],5)
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand',
	ascending=True)

for dataset in combine_df:
	# dataset.loc[row properties, column properties] = desired ordinal value
	print type(dataset)
	dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
	dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
	dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
	dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
	dataset.loc[dataset['Age'] > 64, 'Age'] = 4

train_df = train_df.drop(['AgeBand'], axis=1)
combine_df = [train_df, test_df]
#print train_df.head()

# Create new features combining existing features: create "FamilySize", "IsAlone", and 'Age*Class'
for dataset in combine_df:
	dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

# print train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values( \
# 	by='Survived', ascending=False)

for dataset in combine_df:
	dataset['IsAlone'] = 0
	dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
# print train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()

train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine_df = [train_df, test_df]
# print train_df.head()

for dataset in combine_df:
	dataset['Age*Class'] = dataset['Age']*dataset['Pclass']
# print train_df.loc[:, ['Age*Class', 'Age','Pclass']].head()

# Completing categorical features: change S, Q, C values from 'Embarked'
# print np.where(pd.isnull(train_df['Embarked']))
freq_port = train_df.Embarked.dropna().mode()[0] # '[0]' is there to pick 'S' specifically (?)

for dataset in combine_df:
	dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

# print train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values( \
# 	'Survived', ascending=False)

# print train_df.loc[59:62,:]

# Converting categorical feature to numeric: changing 'Embarked' feature to numeric port feature
for dataset in combine_df:
	dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
# print train_df.head()

# Quick completing and converting 'Fare' numeric feature
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
# print train_df[['FareBand','Survived']].groupby(['FareBand'], as_index=False).mean().sort_values( \
# 	'Survived', ascending=False)

for dataset in combine_df:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]

print 'PREPARED FINAL DATASET'
print '\nTRAIN DATASET:\n'   
print train_df.head(10)
print '\nTEST DATASET:\n'
print test_df.head(10)

''' MODEL, PREDICT, and SOLVE
Application of the following methods:

-Logistic Regression
-KNN or k-Nearest Neighbors
-Support Vector Machines
-Naive Bayes Classifier
-Decision Tree
-Random Forrest
-Perception
-Artificial Neural Network
-Relevance Vector Machine

'''
# Preparation of data
X_train = train_df.drop('Survived', axis=1)
Y_train = train_df['Survived']
X_test = test_df.drop('PassengerId', axis=1).copy()
print X_train.shape, Y_train.shape, X_test.shape

# Logistics Regression
logreg = LogisticRegression()
logreg.fit(X_train,Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train,Y_train)*100, 2)
# print 'Logistic Regression Score: %s' %acc_log

coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df['Correlation'] = pd.Series(logreg.coef_[0])
# print coeff_df.sort_values(by='Correlation', ascending=False)

# Support Vector Machine
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train)*100, 2)
# print 'SVC Score: %s' %acc_svc

# K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train)*100, 2)
# print 'K-Nearest Neighbors Score: %s' %acc_knn

# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train)*100, 2)
# print 'Gaussian Naive Bayes Score: %s' %acc_gaussian

# Perceptron
perceptron = Perceptron()s
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train)*100, 2)
# print 'Perceptron Score: %s' %acc_perceptron

# Linear SVC
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc= round(linear_svc.score(X_train, Y_train)*100, 2)
# print 'Linear SVC Score: %s' %acc_linear_svc

# Stochastic Gradient Descent
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train)*100, 2)
# print 'Stochastic Gradient Descent Score: %s' %acc_sgd

# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train)*100, 2)
# print 'Decision Tree Score: %s' %acc_decision_tree

# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
acc_random_forest = round(random_forest.score(X_train, Y_train)*100, 2)
# print 'Random Forest Score: %s' %acc_random_forest

# Model Evaluation
models = pd.DataFrame({
	'Model': 	['Support Vector Machines', 'KNN', 'Logistic Regression',
				'Random Forest', 'Naive Bayes', 'Perceptron',
				'Stochastic Gradient Decent', 'Linear SVC', 'Decision Tree'],
	'Score':  	[acc_svc, acc_knn, acc_log, acc_random_forest, acc_gaussian,
				 acc_perceptron, acc_sgd, acc_linear_svc, acc_decision_tree]})
print 'MODEL RESULTS:'
print models.sort_values(by='Score', ascending=False)









