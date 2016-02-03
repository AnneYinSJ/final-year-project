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

#Imports for the particular algorithm being implemented
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn import tree

#Test data and labels are loaded
def load_test_data():
	#DataTable creation using pandas
	table = pd.read_csv("C:/Users/HP-PC/Desktop/Padippu/Project/FinalYearProject/poker-hand-training-true.data")
	table.columns = ["Suit1","Card1","Suit2","Card2","Suit3","Card3","Suit4","Card4","Suit5","Card5","Poker Hand"]
	X = table.loc[:,["Suit1","Card1","Suit2","Card2","Suit3","Card3","Suit4","Card4","Suit5","Card5"]]
	Y = table.loc[:,["Poker Hand"]]
	print("Test data loaded...")
	return [X,Y]

#The training dataset is created
def load_training_data():
	training_data = pd.read_csv("C:/Users/HP-PC/Desktop/Padippu/Project/FinalYearProject/poker-hand-testing.data")
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
#Usual score is 66%
#Score using random sampled data of 100000 records is 38%
#Score using random sampled data of 300000 records is 60%
def classify_dtree(X,Y,test_data,test_labels):  
	print("Building the model for decision trees")
	clf = tree.DecisionTreeClassifier()
	clf = clf.fit(X,Y)
	print("Classification Score using Decision Tree:" + str(clf.score(test_data,test_labels)))

#Reservoir Sampling method
def reservoir_sampler(data,labels):
	SAMPLE_COUNT = 100000
	# Force the value of the seed so the results are repeatable
	random.seed(12345)
	sampled_data = pd.DataFrame()
	sampled_labels = pd.DataFrame()
	for i in range(0,data.size-1):
		#Generate the reservoir
		if i < SAMPLE_COUNT:
			sampled_data.append(data.loc[i])
			sampled_labels.append(labels.loc[i])
		else:
			# Randomly replace elements in the reservoir
            # with a decreasing probability.
            # Choose an integer between 0 and index (inclusive)
			r = random.randint(0,i)
			if r < SAMPLE_COUNT:
				sampled_data.loc[r] = data.loc[i]
				sampled_labels.loc[r] = labels.loc[i]
	print(sampled_data) 
	print(sampled_labels)
	return [sampled_data,sampled_labels]

#Random Sampling Method
def random_sampler(data,labels):
	sampled_data = pd.DataFrame()
	sampled_labels = pd.DataFrame()
	random.seed(12345)
	for i in range(0,299999):
		#print(i)
		r = random.randint(0,999998)
		sampled_data = sampled_data.append(data.loc[r])
		sampled_labels = sampled_labels.append(labels.loc[r])
	sampled_data.to_csv("random_sampled_data.csv")
	sampled_labels.to_csv("random_sampled_labels.csv")
	return [sampled_data,sampled_labels]

#Main method 
def main():
	[X,Y] = load_training_data()
	[td,tl] = load_test_data()
	op = classify(X,Y,td,tl)
	classify_dtree(X,Y,td,tl)
	