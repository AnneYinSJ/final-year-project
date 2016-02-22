"""
A Scalable Classifier for processing Big Data Streams
Authors: Kiran Sudhir, Mayanka Pachaiyappa and Varun Bezzam
Sri Sivasubramaniya Nadar College of Engineering
Kalavakkam, Chennai, Tamil Nadu
"""
#Standard imports for machine learning
import pandas as pd
import numpy as np

#Imports for the particular algorithm being implemented
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import tree
from sklearn import cross_validation
from sklearn.linear_model import LinearRegression

#Imports for time
from datetime import datetime

#Import for math
from math import sqrt

#Load the training dataset
def load_data():
    data = pd.read_csv("C:/Users/HP-PC/Desktop/final-year-project/data/sea.data")
    data.columns = ["Attr1","Attr2","Attr3","Class"]
    X = data.loc[:,["Attr1","Attr2","Attr3"]]
    Y = data.loc[:,["Class"]]

    return [X, Y]

#Classification using Random Forests for the concept drift dataset
def classify(X,Y):
    print("Building the model for random forests...")
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.1)
    clf = ExtraTreesClassifier(n_estimators=10)
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)
    clf = clf.fit(X_train,y_train)
    print("Classification Score using Random Forests:" + str(clf.score(X_test,y_test)))

#Classification using decision trees for the concept drift dataset
def classify_dtree(X,Y):
    print("Building the model for decision trees...")
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.1)
    start_time = datetime.now()
    #print(start_time)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train,y_train)
    end_time = datetime.now()
    #print(end_time)
    print("Classification Score using Decision Tree:" + str(clf.score(X_test,y_test)))

#The Early Drift Detection Method is implemented here...
def eddm(X,Y):
    INIT_TRAIN_SIZE = 1000
    no_of_errors = 0
    pmax = 0.5
    smax = 0.5
    pi_bar = 0
    ALPHA = 0.95
    BETA = 0.90
    store = False
    data_frame_list = []
    data_label_list = []
    NO_OF_DRIFTS = 0
    init_data = X.loc[0:INIT_TRAIN_SIZE]
    init_labels = Y.loc[0:INIT_TRAIN_SIZE]
    #X = X.loc[INIT_TRAIN_SIZE:len(X)]
    #Y = Y.loc[INIT_TRAIN_SIZE:len(Y)]
    new_data = pd.DataFrame()
    new_labels = pd.DataFrame()

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(init_data,init_labels)

    assert (len(X)==len(Y)), "Dimension mismatch between X and Y"

    for i in X.index:
        if clf.predict(X.loc[i])[0] != Y.loc[i][0]:
            no_of_errors = no_of_errors + 1
            if no_of_errors <= 30:
                old_index = i
            elif no_of_errors>30:
                pi = (i - old_index)
                #print(pi)
                old_index = i
                pi_bar = ((no_of_errors-1)*pi_bar + pi)/(no_of_errors)
                assert (pi>0), "Average distance between errors is less than zero!"
                si_bar = sqrt(pi_bar*(1-pi_bar)/no_of_errors)
                if (pi_bar + 2*si_bar) > (pmax + 2*smax):
                    pmax = pi_bar
                    smax = si_bar
                    #print("Updated pmax and smax at" + str(i))

                print(str((pi_bar + 2*si_bar)/(pmax + 2*smax)) + "   " + str(i))

                if (pi_bar + 2*si_bar)/(pmax + 2*smax) < ALPHA:
                    store = True
                    print("Possibility of drift at :" + str(i))

                if (pi_bar + 2*si_bar)/(pmax + 2*smax) < BETA:
                    print("Drift confirmed at :" + str(i))
                    #break
                    new_data = new_data.append(X.loc[i])
                    new_labels = new_labels.append(Y.loc[i])
                    clf = clf.fit(new_data,new_labels)
                    data_frame_list.append(new_data)
                    data_label_list.append(new_labels)
                    NO_OF_DRIFTS = NO_OF_DRIFTS + 1
                    new_data = pd.DataFrame()
                    new_labels = pd.DataFrame()
                    no_of_errors = 0
                    pi_bar = 0
                    pmax = 0.5
                    smax = 0.5
                    store = False

                if (pi_bar + 2*si_bar)/(pmax + 2*smax) > ALPHA:
                    new_data = pd.DataFrame()
                    new_labels = pd.DataFrame()
                    store = False

        if store == True:
            new_data = new_data.append(X.loc[i])
            new_labels = new_labels.append(Y.loc[i])

    return [data_frame_list,data_label_list]

#The main method is defined here...
def main():
    [X, Y] = load_data()
    #classify(X,Y)
    #classify_dtree(X,Y)
    [nd,nl] = eddm(X,Y)

#This is how the main method is called...
if __name__ == '__main__':
    main()
