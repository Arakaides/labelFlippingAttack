from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics 

# Load the training data from a file
#trainingData = np.loadtxt('trainingData.txt', delimiter=',')

#Random Data
np.random.seed(0)
x = np.sort(5*np.random.rand(100,1), axis=0)
y = np.where(x > 2.5, 1, -1)

# Split the data into training and testing sets
#xTrain, xTest, yTrain, yTest = train_test_split(trainingData[:, :-1], trainingData[:, -1], test_size=0.3, random_state=42)

# Train the SVM on the training set
svm = SVC(kernel='linear', C=1)
#svm.fit(xTrain, yTrain)
#Alternate fit method for random data
svm.fit(x,y)

# Load the testing data from a file
#testData = np.loadtxt('test_data.txt', delimiter=',')

#Plot the decision boundary
plt.scatter(x[:, 0], y, color='black')
plt.plot(x, svm.predict(x), color='red', linewidth=2)
plt.show()

#evaluate accuracy of SVM on test set
yPred = svm.predict(x)
accOrig = sklearn.metrics.balanced_accuracy_score(y, yPred)
print('Accuracy on test set: ', accOrig)

# Classify the testing data using the trained SVM
#predictions = svm.predict(testData)

#print(predictions)