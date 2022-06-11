# -*- coding: utf-8 -*-
#Importing required packages.
import pandas as pd
import itertools
import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt

# Classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB 

# Features
from sklearn.pipeline import make_pipeline
#from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

#Loading dataset
file = pd.read_csv('C:\\Users\\vstei\\Desktop\\TCC\\Profile\\6users_150posts.csv',sep=';')

count_name = 0

# Replace Username by an ID
replacements = dict(zip(file['Username'].unique(),itertools.count(1)))
file["ID"] = file["Username"].replace(replacements)
file = file.drop('Username', axis=1)  # Remove username column

lexical_features = file[['word per sentence', 'sentence lenght', 'word lenght', 'word per post', 'post lenght', 'longwords in post', 'shortwords in post']] 
strutural_features = file[['lowercase starting sentences', 'uppercase starting sentences', 'sentence per post']]
syntactic_features = file[['ponctuation mark per post','exclamacao','virgula','aspas_simples','aspas_duplas','ponto_virgula','ponto','traco','interrogacap','misspeled words','adjectives','nouns','verbs','adverb','conjunction','article','pronoun','preposicao','interjeicao']]

lexical_strutural = file[['word per sentence', 'sentence lenght', 'word lenght', 'word per post', 'post lenght', 'longwords in post', 'shortwords in post','lowercase starting sentences', 'uppercase starting sentences', 'sentence per post']]
lexical_syntactic = file[['word per sentence', 'sentence lenght', 'word lenght', 'word per post', 'post lenght', 'longwords in post', 'shortwords in post','ponctuation mark per post','exclamacao','virgula','aspas_simples','aspas_duplas','ponto_virgula','ponto','traco','interrogacap','misspeled words','adjectives','nouns','verbs','adverb','conjunction','article','pronoun','preposicao','interjeicao']]
Syntactic_strutural = file[['lowercase starting sentences', 'uppercase starting sentences', 'sentence per post','ponctuation mark per post','exclamacao','virgula','aspas_simples','aspas_duplas','ponto_virgula','ponto','traco','interrogacap','misspeled words','adjectives','nouns','verbs','adverb','conjunction','article','pronoun','preposicao','interjeicao']]

# Data preprocessing
#X = syntactic_features
#X = lexical_features
#X = strutural_features

X = lexical_strutural
#X = lexical_syntactic
#X = Syntactic_strutural

#X = file.drop('ID', axis=1)
y = file['ID']

# Train and Test splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state = 12)

# Applying Standard scaling to get optimized results
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Running the classifier - SVM Classifier 01
clf = svm.SVC()
clf.fit(X_train, y_train)
clf_y_pred = clf.predict(X_test)

# Evaluate the model
print('Running the classifier - SVM Classifier 01')
print(classification_report(y_test, clf_y_pred))
#print(confusion_matrix(y_test, clf_y_pred))
#print('Score: ', str(clf.score(X_test, y_test)))
#print('Score 2: ', str(f1_score(y_test, clf_y_pred, average='weighted', labels=np.unique(clf_y_pred))))
#print()

# Running the classifier - Decision tree
dt = make_pipeline(StandardScaler(), DecisionTreeClassifier())
dt.fit(X_train, y_train)
dt_y_pred = dt.predict(X_test)

# Evaluate the model
print('Running the classifier - Decision tree')
print(classification_report(y_test, dt_y_pred))
#print(confusion_matrix(y_test, dt_y_pred))
#print('Score: ', str(dt.score(X_test, y_test)))
#print('Score 2: ', str(f1_score(y_test, dt_y_pred, average='weighted', labels=np.unique(dt_y_pred))))
#print()

# Running the classifier - k-Nearest Neighbor
knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train, y_train)
knn_y_pred = knn.predict(X_test)

# Evaluate the model
print('Running the classifier - k-Nearest Neighbor')
print(classification_report(y_test, knn_y_pred))
#print(confusion_matrix(y_test, knn_y_pred))
#print('Score: ', str(knn.score(X_test, y_test)))
#print('Score 2: ', str(f1_score(y_test, knn_y_pred, average='weighted', labels=np.unique(knn_y_pred))))
#print()
