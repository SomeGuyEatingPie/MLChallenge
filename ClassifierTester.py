# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 14:56:06 2023

@author: Temp
"""

from mlxtend.classifier import StackingClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score
from sklearn.ensemble import BaggingClassifier

from load_data import load_data_from_csv

training_feature_names, training_instances, training_labels = load_data_from_csv("datasets/telco_customer_churn.test.csv")
test_feature_names, test_instances, test_labels = load_data_from_csv("datasets/telco_customer_churn.training.csv")

totalF = 0
lowF= 1
lowI = 0
highF = 0 
highI = 0
count = 0

for i in range(5,105,5):
    
    classifier =  BaggingClassifier(max_features=5, random_state=i)
    classifier.fit(training_instances, training_labels)
    predicted_labels = classifier.predict(test_instances)
    
    fScore = accuracy_score(test_labels, predicted_labels)

    totalF += fScore
    if (fScore < lowF):
        lowF = fScore
        lowI = i
        
    if (fScore > highF):
        highF = fScore
        highI = i
        
    print(i, fScore)
    count +=1
   
print("Avg: ", totalF/count)
print("Lowest: ", lowI, lowF)
print("Highest: ", highI, highF)