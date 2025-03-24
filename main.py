"""This is simple example logistic regression"""

#import the necessary libraries
import pandas as pd 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.linear_model import LogisticRegression

#Reading csv file
df = pd.read_csv("data/diabetes-dataset.csv")

#Defining x , y
x = df.drop(["Outcome"] , axis=1)
y = df["Outcome"]

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.2)

#Defining model to be used
model = LogisticRegression()
model.fit(x_train , y_train)

#checking the model
print(model.score(x_test ,y_test ))