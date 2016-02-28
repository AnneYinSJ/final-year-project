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
    pmax = 0.0000001
    smax = 0.0000001
    pi_bar = 0
    ALPHA = 0.70
    BETA = 0.68
    store = False
    data_frame_list = []
    data_label_list = []
    NO_OF_DRIFTS = 0
    init_data = X.loc[0:INIT_TRAIN_SIZE]
    init_labels = Y.loc[0:INIT_TRAIN_SIZE]
    new_data = pd.DataFrame()
    new_labels = pd.DataFrame()

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(init_data,init_labels)

    assert (len(X)==len(Y)), "Dimension mismatch between X and Y"

    for i in X.index:
        if clf.predict(X.loc[i])[0] != Y.loc[i][0]:
            no_of_errors = no_of_errors + 1
            if no_of_errors == 1:
                old_index = i
            else:
                pi = (i - old_index)
                old_index = i
                pi_bar = ((no_of_errors-2)*pi_bar + pi)/(no_of_errors-1)
                #print("Pi : " + str(pi))
                #print("Pi' : " + str(pi_bar))
                #print("No of errors : " + str(no_of_errors))
                assert (pi>0), "Average distance between errors is less than zero!"
                si_bar = sqrt(pi_bar*(1-pi_bar)/(no_of_errors-1))

                if (pi_bar + 2*si_bar) > (pmax + 2*smax):
                    pmax = pi_bar
                    smax = si_bar
                    #print("Updated pmax and smax at" + str(i))

                #print(str((pi_bar + 2*si_bar)/(pmax + 2*smax)) + "   " + str(i))

                if (pi_bar + 2*si_bar)/(pmax + 2*smax) < ALPHA:
                    store = True
                    print("Possibility of drift at :" + str(i))

                if (pi_bar + 2*si_bar)/(pmax + 2*smax) < BETA:
                    print("Drift confirmed at :" + str(i))
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


'''
Our own drift detection method is being implemented here. Both performance and accuracy are significantly better than EDDM.
Preliminary results look very promising.
'''
# New model trained with data from 15475 to 16375 for the first drift detected.
# New model trained with data from 30390 to 31100 for the second drift detected.
# New model trained with data from 44999 to 45673 for the third drift detected.
def myeddm(X,Y):
    no_errors = 0
    NO_OF_ERRORS_TO_BE_SEEN = 225
    ALLOWED_SPACE = 18
    INIT_TRAIN_SIZE = 1000
    store = True
    init_data = X.loc[0:INIT_TRAIN_SIZE]
    init_labels = Y.loc[0:INIT_TRAIN_SIZE]
    old_index = -1
    start_index = 0
    new_data = pd.DataFrame()
    new_labels = pd.DataFrame()
    dlist = []
    llist = []
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(init_data,init_labels)
    for i in X.index:
        if i%1000 == 0 and i!=0:
            print("Index is :" + str(i))
            clf = clf.fit(X.loc[start_index:i],Y.loc[start_index:i])
        if clf.predict(X.loc[i])[0] != Y.loc[i][0]:
            #print(no_errors)
            no_errors = no_errors + 1
            if old_index == -1:
                old_index = i-1
            error_space = i - old_index
            old_index = i
            if error_space <= ALLOWED_SPACE:
                store = True
                if no_errors >= NO_OF_ERRORS_TO_BE_SEEN:
                    print("Drift confirmed at :" + str(i))
                    print("New model trained with data from " + str(new_data.index[0]) + " to " + str(i) + " indices")
                    clf = clf.fit(new_data,new_labels)
                    start_index = new_data.index[0]
                    dlist.append(new_data)
                    llist.append(new_labels)
                    new_data = pd.DataFrame()
                    new_labels = pd.DataFrame()
                    no_errors = 0
                    old_index = -1

            else:
                new_data = pd.DataFrame()
                new_labels = pd.DataFrame()
                no_errors = 0
                store = False
        if store == True:
            new_data = new_data.append(X.loc[i])
            new_labels = new_labels.append(Y.loc[i])

    return [dlist,llist]

#The main method is defined here...
def main():
    [X, Y] = load_data()
    #classify(X,Y)
    #classify_dtree(X,Y)
    [nd,nl] = myeddm(X,Y)

#This is how the main method is called...
if __name__ == '__main__':
    main()
