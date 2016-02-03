"""
A Scalable Classifier for processing Big Data Streams
Authors: Kiran Sudhir, Mayanka Pachaiyappa and Varun Bezzam
Sri Sivasubramaniya Nadar College of Engineering
Kalavakkam, Chennai, Tamil Nadu
"""

import pandas as pd
import numpy as np
import random

#Manual Comparison of results
#Use this method to properly understand results
def compare(output,test_labels):
	print("Comparing the results...")
	count = 0
	test_labels = np.ravel(test_labels)
	for i in range(0,output.size):
		if output[i] != test_labels[i]:
			count = count+1
	print("No.of misclassified instances:" + str(count))
	print("Total no. of instances:" + str(output.size))
	print("Percentage of misclassified instances:" + str(count*100/output.size) + "%")

#Some processing operations to replace numeric values with categorical labels
def transform(X,Y):
	suit_list = ["Suit1","Suit2","Suit3","Suit4","Suit5"]
	rank_list = ["Card1","Card2","Card3","Card4","Card5"]
	num_list = [1,2,3,4,5,6,7,8,9,10,11,12,13]
	card_list = ["Ace","Two","Three","Four","Five","Six","Seven","Eight","Nine","Ten","Jack","Queen","King"]
	for s in suit_list:
		X.loc[:,s] = X.loc[:,s].replace([1,2,3,4],["Hearts", "Spades", "Diamonds", "Clubs"])
	for r in rank_list:
		X.loc[:,r] = X.loc[:,r].replace(num_list,card_list)
	X.to_csv("C:/Users/HP-PC/Desktop/Padippu/Project/FinalYearProject/categorical_training_data.csv")
	Y.to_csv("C:/Users/HP-PC/Desktop/Padippu/Project/FinalYearProject/categorical_training_labels.csv")

#DataTable viewing using pandas
def view_table(table):	
	table.head()
	table.tail()
	table.shape
	table.values
	table.describe()