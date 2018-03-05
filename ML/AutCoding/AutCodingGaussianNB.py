#Import Library of Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB
import numpy as np

#assigning predictor and target variables
x= np.array([[39179,8022],[39180,8023],[39179,8022],[39179,8022]])
Y = np.array(['01-110','011-210','011-210','011-210'])

#Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets 
model.fit(x, Y)

#Predict Output 
predicted= model.predict([[39179,8022]])
print (predicted)