# -*- coding: utf-8 -*-
#Importing required packages.
import pandas as pd                                        # For dataframes
from pandas import DataFrame                               # For dataframes
import itertools
from numpy import ravel                                    # For matrices
import matplotlib.pyplot as plt                            # For plotting data
import seaborn as sns                                      # For plotting data
from sklearn.model_selection import train_test_split       # For train/test splits
from sklearn.feature_selection import VarianceThreshold    # Feature selector
from sklearn.pipeline import Pipeline, make_pipeline       # For setting up pipeline
from sklearn import decomposition, datasets                # For Decomposition
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction import DictVectorizer

# Classifier
from sklearn.neighbors import KNeighborsClassifier         # The k-nearest neighbor classifier
from sklearn.tree import DecisionTreeClassifier            # The decision tree classifier
from sklearn.svm import SVC                                # The SVC Classifier
from sklearn.naive_bayes import MultinomialNB              # The Naive Bayes Multinomial classifier
from sklearn import tree                                   # The decision tree classifier

# Various pre-processing steps
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler, PowerTransformer, MaxAbsScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV           # For optimization

#Loading dataset
file = pd.read_csv('C:\\Users\\vstei\\Desktop\\TCC\\Profile\\profile_csv_file_7users_150posts.csv',sep=',')

# Exploring dataset - Replace username for an ID
count_name = 0
replacements = dict(zip(file['Username'].unique(),itertools.count(1)))
file["ID"] = file["Username"].replace(replacements)
  
# Exploring dataset - Remove username
file = file.iloc[: , 1:]              # to remove the 1st column
file = file.drop('Username', axis=1)  # Remove username column

# Exploring dataset - Feature types
lexical_features = file[['Average word per sentence', 'Average sentence lenght', 'Average word lenght', 'Average word per post', 'Average post lenght', 'Average longwords in post', 'Average shortwords in post']] 
strutural_features = file[['Average URL per post', 'Average lowercase starting sentences', 'Averave uppercase starting sentences', 'Average sentence per post']]
syntactic_features = file[['Average ponctuation mark per post', 'adjectives', 'nouns', 'verbs', 'adverb', 'conjunction', 'article', 'pronoun']]

# Data preprocessing - Feature selection
X = file.drop('ID', axis=1)
y = file['ID']

# Data preprocessing - Split the data into test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Classifier - k-Nearest Neighbor
knn = KNeighborsClassifier(n_neighbors=9).fit(X_train, y_train)

# Parameter selection 
knn_parameters = {'scaler': [StandardScaler(), MinMaxScaler(),
    Normalizer(), MaxAbsScaler()],
    'selector__threshold': [0, 0.001, 0.01],
#	'classifier__n_neighbors': [1, 3, 5, 7, 10],
    'classifier__n_neighbors': [1,2,3,4,5,6,7,8,9,11,12,13,10],
    'classifier__p': [1, 2],
    'classifier__leaf_size': [1, 5, 10, 15]
}

knn_pipe = Pipeline([
('scaler', StandardScaler()),
('selector', VarianceThreshold()),
('classifier', KNeighborsClassifier())
])
 
knn_pipe.fit(X_train, y_train) 
grid = GridSearchCV(knn_pipe, knn_parameters, cv=5).fit(X_train, y_train)

# k-Nearest Neighbor results using GridSearch
print('k-Nearest Neighbor Classifier \n')
print('Training set score: ' + str(grid.score(X_train, y_train)))
print('Test set score: ' + str(grid.score(X_test, y_test)))
print('\n')

# Classifier - Decision Tree Classifier
# Using StandardScaler and PCA
std_slc = StandardScaler()
pca = decomposition.PCA()
dec_tree = tree.DecisionTreeClassifier().fit(X_train, y_train)

# Using Pipeline for GridSearchCV
dt_pipe = Pipeline(steps=[
    ('std_slc', std_slc),
    ('pca', pca),
    ('dec_tree', dec_tree)])

n_components = list(range(1,X.shape[1]+1,1))

criterion = ['gini', 'entropy']
max_depth = [2,4,6,8,10,12]

dt_parameters = dict(
    pca__n_components=n_components,
    dec_tree__criterion=criterion,
    dec_tree__max_depth=max_depth)

clf_GS = GridSearchCV(dt_pipe, dt_parameters)
clf_GS.fit(X_train, y_train)

# Decision Tree Classifier results using GridSearch
print('Decision Tree Classifier \n')
print('Training set score: ' + str(clf_GS.score(X_train,y_train)))
print('Test set score: ' + str(clf_GS.score(X_test,y_test)))
print('\n')


# SVC Classifier
svc_parameters = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'kernel': ['rbf']}

svc_grid = GridSearchCV(SVC(), svc_parameters, refit = True, verbose = 3)
svc_grid.fit(X_train, y_train)
svc_grid_predictions = svc_grid.predict(X_test)

# SVC Classifier results using GridSearch
print('SVC Classifier \n')
print('Training set score: ' + str(svc_grid.score(X_train,y_train)))
print('Test set score: ' + str(svc_grid.score(X_test,y_test)))
print('SVC Predictions: ' + str(classification_report(y_test, svc_grid_predictions)))
print('\n')

# Multinomial Naive Bayes classifier
mnb = MultinomialNB()
mnb.fit(X_train, y_train)

#dv = DictVectorizer(sparse=False)
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
nb_predict = mnb.predict(X_test)

# Multinomial Naive Bayes classifier result
print('Multinomial Naive Bayes Classifier \n')
print('Training set score: ' + str(mnb.score(X_train,y_train)))
print('Test set score: ' + str(mnb.score(X_test,y_test)))
print('NB Predictions: ' + str(classification_report(y_test, nb_predict)))
