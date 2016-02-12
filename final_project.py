"""
A Scalable Classifier for processing Big Data Streams
Authors: Kiran Sudhir, Mayanka Pachaiyappa and Varun Bezzam
Sri Sivasubramaniya Nadar College of Engineering
Kalavakkam, Chennai, Tamil Nadu
"""

#Standard imports for machine learning
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import datetime

#Imports for the particular algorithm being implemented
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn import tree
from sklearn.cross_validation import StratifiedShuffleSplit

#Test data and labels are loaded
def load_test_data():
	#DataTable creation using pandas
	table = pd.read_csv("C:/Users/HP-PC/Desktop/final-year-project/data/poker-hand-training-true.data")
	table.columns = ["Suit1","Card1","Suit2","Card2","Suit3","Card3","Suit4","Card4","Suit5","Card5","Poker Hand"]
	X = table.loc[:,["Suit1","Card1","Suit2","Card2","Suit3","Card3","Suit4","Card4","Suit5","Card5"]]
	Y = table.loc[:,["Poker Hand"]]
	print("Test data loaded...")
	return [X,Y]

#The training dataset is created
def load_training_data():
	training_data = pd.read_csv("C:/Users/HP-PC/Desktop/final-year-project/data/poker-hand-testing.data")
	training_data.columns = ["Suit1","Card1","Suit2","Card2","Suit3","Card3","Suit4","Card4","Suit5","Card5","Poker Hand"]
	training_labels = training_data.loc[:,["Poker Hand"]]
	training_data = training_data.loc[:,["Suit1","Card1","Suit2","Card2","Suit3","Card3","Suit4","Card4","Suit5","Card5"]]
	print("Training data loaded...")
	return [training_data,training_labels]

#RandomForests using scikit-learn
def classify(X,Y,test_data,test_labels):
	print("Building the model for random forests...")
	Y = np.ravel(Y)
	test_labels = np.ravel(test_labels)
	clf = ExtraTreesClassifier(n_estimators=10)
	clf = clf.fit(X,Y)
	print("Classification Score using Random Forests:" + str(clf.score(test_data,test_labels)))
	output = clf.predict(test_data)
	return output

#Decision Tree using scikit-learn
#Usual score is 66% for 999999 records and time to execute is 10.96 seconds
def classify_dtree(X,Y,test_data,test_labels):
	print("Building the model for decision trees...")
	start_time = datetime.now()
	print(start_time)
	clf = tree.DecisionTreeClassifier()
	clf = clf.fit(X,Y)
	end_time = datetime.now()
	print(end_time)
	print("Classification Score using Decision Tree:" + str(clf.score(test_data,test_labels)))

#Reservoir Sampling method
#Score using reservoir sampled data of 100000 records is 56% and time to execute is 0.6 seconds
#Score using reservoir sampled data of 300000 records is 59% and time to execute is 2.95 seconds
def reservoir_sampler(data,labels):
	SAMPLE_COUNT = 300000
	# Force the value of the seed so the results are repeatable
	random.seed(12345)
	sampled_data = pd.DataFrame()
	sampled_labels = pd.DataFrame()
	#Generate the reservoir
	sampled_data = sampled_data.append(data.loc[0:SAMPLE_COUNT-1])
	sampled_labels = sampled_labels.append(labels.loc[0:SAMPLE_COUNT-1])

	for i in range(SAMPLE_COUNT,999998):
			# Randomly replace elements in the reservoir
            # with a decreasing probability.
            # Choose an integer between 0 and index (inclusive)
			print(i)
			r = random.randint(0,i)
			if r < SAMPLE_COUNT:
				sampled_data.loc[r] = data.loc[i]
				sampled_labels.loc[r] = labels.loc[i]
	sampled_data.to_csv("reservoir_sampled_data.csv")
	sampled_labels.to_csv("reservoir_sampled_labels.csv")
	return [sampled_data,sampled_labels]

#Random Sampling Method
#Score using random sampled data of 100000 records is 38% and time to execute is 0.72 seconds
#Score using random sampled data of 300000 records is 60% and time to execute is 2.24 seconds
def random_sampler(data,labels):
	NO_OF_SAMPLES = 99999
	sampled_data = pd.DataFrame()
	sampled_labels = pd.DataFrame()
	random.seed(12345)
	for i in range(0,NO_OF_SAMPLES):
		print(i)
		r = random.randint(0,999998)
		sampled_data = sampled_data.append(data.loc[r])
		sampled_labels = sampled_labels.append(labels.loc[r])
	sampled_data.to_csv("random_sampled_data.csv")
	sampled_labels.to_csv("random_sampled_labels.csv")
	return [sampled_data,sampled_labels]

#Stratified Sampling Method
#Score using stratified sampled data of 100000 records is 56% and time to execute is 0.74 seconds
#Score using stratified sampled data of 300000 records is 57% and time to execute is 2.80 seconds
def stratified_sampler():
	table = pd.read_csv("C:/Users/HP-PC/Desktop/final-year-project/data/poker-hand-testing.data")
	table.columns = ["Suit1","Card1","Suit2","Card2","Suit3","Card3","Suit4","Card4","Suit5","Card5","Poker Hand"]
	target = table["Poker Hand"]
	table = table.drop("Poker Hand", axis=1)
	sss = StratifiedShuffleSplit(target,test_size=0.3)
	for train_index, test_index in sss:
		xtrain,xtest = table.loc[train_index],table.loc[test_index]
		ytrain,ytest = target[train_index],target[test_index]
	return [xtrain,xtest,ytrain,ytest]
	# Check target series for distribution of classes
	#ytrain.value_counts()
	#ytest.value_counts()

#Main method
def main():
	[X,Y] = load_training_data()
	[td,tl] = load_test_data()
	op = classify(X,Y,td,tl)
	classify_dtree(X,Y,td,tl)

