# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 18:10:14 2016

@author: chirag
"""


import pandas as pd
import os
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn import tree
import io
import matplotlib.pyplot as pl
from sklearn.metrics import precision_recall_curve, auc



os.chdir('C:\\Users\\chirag\\Desktop\\Data Science\\hw\\hw4\\homework4')

data_file=pd.read_csv("smote_creditData.csv")

cl=data_file.loc[:]['class']
gd=0
bd=0

for i in range(len(cl)):
    if cl[i] =='good':
        gd =gd+1
    else:
        bd =bd+1
    
print("good_count : ",gd,"bad_count : ",bd)


columns=data_file.columns
data2=data_file[columns[:-1]]
predictor=pd.get_dummies(data2)

predictor_scaled= preprocessing.scale(predictor)

tar_class=data_file[columns[-1]]

target=tar_class.map({'good':1,'bad':0})


predictor_scaled_df=pd.DataFrame(predictor_scaled)
target_df=pd.DataFrame(target)


result = pd.concat([predictor_scaled_df, target_df], axis=1)

from sklearn import decomposition
pca = decomposition.PCA(n_components=30)
pca.fit(predictor_scaled)
X = pca.transform(predictor_scaled)


from sklearn.cross_validation import train_test_split


train_X, test_X, train_Y, test_Y =  train_test_split(X, target, test_size = 0.4, random_state = 99)

classifier = LogisticRegression()
classifier.fit(train_X, train_Y)
predicted = classifier.predict(test_X)

import sklearn.metrics as metrics

print("Logistic Regression Result")
print("Accuracy: " + str(metrics.accuracy_score(test_Y, predicted)))
print("\nThe confusion matrix is: ")
print(metrics.confusion_matrix(test_Y, predicted))
print(metrics.classification_report(test_Y, predicted,  target_names = ["good", "bad"]))

    
    
probabilities = classifier.predict_proba(test_X)

precision, recall, prob_threholds = precision_recall_curve(test_Y, probabilities[:, 1])

area = auc(recall, precision)
print("\nArea Under Curve: %0.2f" % area)
#print(prob_thresholds)
pl.clf()
pl.plot(recall, precision, label='Precision-Recall curve')
pl.xlabel('Recall')
pl.ylabel('Precision')
pl.ylim([0.0, 1.05])
pl.xlim([0.0, 1.0])
pl.title('Precision-Recall example: AUC=%0.2f' % area)
pl.legend(loc="lower left")
pl.show()




# Naive baise


from sklearn.cross_validation import cross_val_score, KFold
from sklearn.naive_bayes import GaussianNB


train_X,test_X,train_Y,test_Y=train_test_split(predictor, target, test_size = 0.4, random_state = 99)

classifier_nb = GaussianNB()

classifier_nb.fit(train_X, train_Y)
predicted = classifier_nb.predict(test_X)



print("Naive Baise Result")

print("Accuracy: " + str(metrics.accuracy_score(test_Y, predicted)))
print("\nThe confusion matrix is: ")
print(metrics.confusion_matrix(test_Y, predicted))
print(metrics.classification_report(test_Y, predicted,  target_names = ["good", "bad"]))
probabilities = classifier_nb.predict_proba(test_X)
counter = 0


probabilities = classifier_nb.predict_proba(test_X)


precision, recall, prob_threholds = precision_recall_curve(test_Y, probabilities[:, 1])

area = auc(recall, precision)
print("\nArea Under Curve: %0.2f" % area)
#print(prob_thresholds)
pl.clf()
pl.plot(recall, precision, label='Precision-Recall curve')
pl.xlabel('Recall')
pl.ylabel('Precision')
pl.ylim([0.0, 1.05])
pl.xlim([0.0, 1.0])
pl.title('Precision-Recall example: AUC=%0.2f' % area)
pl.legend(loc="lower left")
pl.show()


#Random Forest

from sklearn.ensemble import RandomForestClassifier


rf_classifier = RandomForestClassifier(n_estimators = 100)

rf_classifier.fit(train_X, train_Y)

predicted = rf_classifier.predict(test_X)

print("Random Forest Result")
print("Accuracy: " + str(metrics.accuracy_score(test_Y, predicted)))
print("\nThe confusion matrix is: ")
print(metrics.confusion_matrix(test_Y, predicted))
print(metrics.classification_report(test_Y, predicted,  target_names = ["good", "bad"]))



probabilities = rf_classifier.predict_proba(test_X)

precision, recall, prob_threholds = precision_recall_curve(test_Y, probabilities[:, 1])

area = auc(recall, precision)
print("\nArea Under Curve: %0.2f" % area)
#print(prob_thresholds)
pl.clf()
pl.plot(recall, precision, label='Precision-Recall curve')
pl.xlabel('Recall')
pl.ylabel('Precision')
pl.ylim([0.0, 1.05])
pl.xlim([0.0, 1.0])
pl.title('Precision-Recall example: AUC=%0.2f' % area)
pl.legend(loc="lower left")
pl.show()


important_features = rf_classifier.feature_importances_


feature_dict = dict(zip(predictor.columns, important_features))
from collections import Counter
c = Counter(feature_dict)

print('The ten most important features are: \n', c.most_common(10))




#Decision Tree

classifier=tree.DecisionTreeClassifier(criterion='entropy') #default criterion is gini impurity
classifier=classifier.fit(train_X, train_Y)

predicted=classifier.predict(test_X)


print("Decision Tree")
print("Accuracy: " + str(metrics.accuracy_score(test_Y, predicted)))
print("\nThe confusion matrix is: ")
print(metrics.confusion_matrix(test_Y, predicted))
print(metrics.classification_report(test_Y, predicted,  target_names = ["good", "bad"]))


probabilities = classifier.predict_proba(test_X)

precision, recall, prob_threholds = precision_recall_curve(test_Y, probabilities[:, 1])

area = auc(recall, precision)
print("\nArea Under Curve: %0.2f" % area)
#print(prob_thresholds)
pl.clf()
pl.plot(recall, precision, label='Precision-Recall curve')
pl.xlabel('Recall')
pl.ylabel('Precision')
pl.ylim([0.0, 1.05])
pl.xlim([0.0, 1.0])
pl.title('Precision-Recall example: AUC=%0.2f' % area)
pl.legend(loc="lower left")
pl.show()


out = io.StringIO()
tree.export_graphviz(classifier, out_file=out)
import pydotplus
graph=pydotplus.graph_from_dot_data(out.getvalue())
graph.write_png("Decision_tree.png") #save tree in a png file




important_features = classifier.feature_importances_


feature_dict = dict(zip(predictor.columns, important_features))
from collections import Counter
c = Counter(feature_dict)

print('The ten most important features are: \n', c.most_common(10))










