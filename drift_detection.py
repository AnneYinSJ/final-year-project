"""
A Scalable Classifier for processing Big Data Streams
Authors: Kiran Sudhir, Mayanka Pachaiyappa and Varun Bezzam
Sri Sivasubramaniya Nadar College of Engineering
Kalavakkam, Chennai, Tamil Nadu
"""
#!Tabulate for different values of error_space and no_of_errors to be seen.
#Extension to real world ELEC dataset.
#!Display the decision tree. Output into pdf using pydot.
#!Tabulate accuracy for EDDM.
#!Increase in the no. of instances.

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
from statistics import mode
import statistics

#Import for VFDT
from subprocess import call
#Global Variables
res = []

#Load the special training dataset with 120000 instances
#Classification accuracy is again 81%
def load_big_data():
    data = pd.read_csv("C:/Users/HP-PC/Desktop/final-year-project/data/sea120000.csv")
    data.columns = ["Attr1","Attr2","Attr3","Class"]
    X = data.loc[:,["Attr1","Attr2","Attr3"]]
    Y = data.loc[:,["Class"]]

    return [X, Y]

#Load the training dataset
def load_data():
    data = pd.read_csv("C:/Users/HP-PC/Desktop/final-year-project/data/sea.data")
    data.columns = ["Attr1","Attr2","Attr3","Class"]
    X = data.loc[:,["Attr1","Attr2","Attr3"]]
    Y = data.loc[:,["Class"]]

    return [X, Y]

#The VFDT method is implemented here.
def vfdt():
    call("java -cp moa.jar -javaagent:sizeofag.jar moa.DoTask \"EvaluatePrequential -l trees.HoeffdingTree -s (ArffFileStream -f C:/Users/HP-PC/Desktop/final-year-project/data/sear.data) -o op.csv\"")
    df = pd.read_csv("C:/Users/HP-PC/Desktop/final-year-project/op.csv")
    count=0
    for index,row in df.iterrows():
        if row[0] == row[1]:
            count = count + 1
    print("Classification Accuracy using VFDT is :" + str(count/len(df)))

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

#Classification using decision trees for the concept drift dataset. Classification Accuracy is 87%
def many_classify_dtree(X,Y):
    print("Building the model for decision trees...")
    x = []
    x.append(X.loc[0:15000])
    x.append(X.loc[15000:30000])
    x.append(X.loc[30000:45000])
    x.append(X.loc[45000:59999])
    y = []
    y.append(Y.loc[0:15000])
    y.append(Y.loc[15000:30000])
    y.append(Y.loc[30000:45000])
    y.append(Y.loc[45000:60000])
    scores = []
    for i in range(0,4):
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(x[i], y[i], test_size=0.1)
        start_time = datetime.now()
        #print(start_time)
        clf = ExtraTreesClassifier(n_estimators=10)
        y_train = np.ravel(y_train)
        y_test = np.ravel(y_test)
        clf = clf.fit(X_train,y_train)
        end_time = datetime.now()
        #print(end_time)
        scores.append(clf.score(X_test,y_test))
    s = 0
    for i in range(0,4):
        s= s +scores[i]
        #print(scores[i])

    print("Classification Score using Decision Tree with Drift Detection:" + str(s/4))

#The Early Drift Detection Method is implemented here...
def eddm(X,Y):
    INIT_TRAIN_SIZE = 1000
    no_of_errors = 0
    pmax = 0.0000001
    smax = 0.0000001
    pi_bar = 0
    ALPHA = 0.95
    BETA = 0.90
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
            if no_of_errors <= 30:
                old_index = i
            else:
                pi = (i - old_index)
                old_index = i
                pi_bar = ((no_of_errors-2)*pi_bar + pi)/(no_of_errors-1)
                #print("Pi : " + str(pi))
                #print("Pi' : " + str(pi_bar))
                #print("No of errors : " + str(no_of_errors))
                assert (pi>0), "Average distance between errors is less than zero!"
                si_bar = sqrt(pi_bar*(pi_bar)/(no_of_errors-1))

                if (pi_bar + 2*si_bar) > (pmax + 2*smax):
                    pmax = pi_bar
                    smax = si_bar
                    #print("Updated pmax and smax at" + str(i))

                #print(str((pi_bar + 2*si_bar)/(pmax + 2*smax)) + "   " + str(i))

                if (pi_bar + 2*si_bar)/(pmax + 2*smax) < ALPHA:
                    store = True
                    #print("Possibility of drift at :" + str(i))

                if (pi_bar + 2*si_bar)/(pmax + 2*smax) < BETA:
                    new_data = new_data.append(X.loc[i])
                    new_labels = new_labels.append(Y.loc[i])
                    print("Drift confirmed at :" + str(i) +" using data from index: " + str(new_data.index[0]))
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
# Usual classification accuracy is 81%
def myeddm(X,Y):
    no_errors = 0
    overall_no_errors = 0
    #NO_OF_ERRORS_TO_BE_SEEN = 225
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
            overall_no_errors = overall_no_errors + 1
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

    print("Classification accuracy using our own drift detection method is: " + str((len(Y)-overall_no_errors)/len(Y)))
    return [dlist,llist]

'''
Method that returns the predicted result of the ensemble method
'''
def ensemble_predict(ensemble, data):
    result_set = []
    for clf in ensemble:
        result_set.append(clf.predict(data)[0])
    try:
        result = mode(result_set)
    except statistics.StatisticsError:
        result = result_set[-1]
    res.append(result)
    return result

'''
An ensembling based method to drift detection and prediction is being implemented here.
'''
def ensemble_eddm(X,Y):
    no_errors = 0
    NO_OF_ERRORS_TO_BE_SEEN = 225
    ALLOWED_SPACE = 17
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
    ensemble = []
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(init_data,init_labels)
    ensemble.append(clf)
    for i in X.index:
        if i%1000 == 0 and i!=0:
            print("Index is :" + str(i))
            ensemble[-1] = ensemble[-1].fit(X.loc[start_index:i],Y.loc[start_index:i])
        if ensemble_predict(ensemble,X.loc[i]) != Y.loc[i][0]:
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
                    ensemble.append(clf)
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

#This method computes the accuracy of the ensembling method. The classification accuracy comes out to be close to 80.5%
def find_accuracy(res,Y):
    no_errors = 0
    for i in range(0,len(Y)):
        if Y.loc[i][0] != res[i]:
            no_errors = no_errors + 1
    print("Classification accuracy using ensemble of models is :" + str((len(Y) - no_errors)/len(Y)))

#The main method is defined here...
def main():
    #[X, Y] = load_data()
    [X, Y] = load_big_data()
    #classify(X,Y)
    #classify_dtree(X,Y)
    #[nd,nl] = eddm(X,Y)
    #[nd,nl] = ensemble_eddm(X,Y)
    #find_accuracy(res,Y)
    [nd,nl] = myeddm(X,Y)
    #vfdt()
    #many_classify_dtree(X,Y)


#This is how the main method is called...
if __name__ == '__main__':
    main()
