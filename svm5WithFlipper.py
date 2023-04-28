import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.utils import shuffle
import random



def remove_correlated_features(X):
    corr_threshold = 0.9
    corr = X.corr()
    drop_columns = np.full(corr.shape[0], False, dtype=bool)
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[0]):
            if corr.iloc[i,j] >= corr_threshold:
                drop_columns[j] = True
    columns_dropped = X.columns[drop_columns]
    X.drop(columns_dropped, axis=1, inplace=True)
    return columns_dropped
def remove_less_significant_features(X, Y):
    sl = 0.05
    regression_ols = None
    columns_dropped = np.array([])
    for itr in range(0, len(X.columns)):
        regression_ols = sm.OLS(Y, X).fit()
        max_col = regression_ols.pvalues.idxmax()
        max_val = regression_ols.pvalues.max()
        if max_val > sl:
            X.drop(max_col, axis='columns', inplace=True)
            columns_dropped = np.append(columns_dropped, [max_col])
        else:
            break
    regression_ols.summary()
    return columns_dropped

def compute_cost(W, X, Y):
    N = X.shape[0]
    distances = 1 - Y * (np.dot(X, W))
    distances[distances < 0] = 0  # equivalent to max(0, distance)
    hinge_loss = reg_strength * (np.sum(distances) / N)
    cost = 1 / 2 * np.dot(W,W) + hinge_loss
    return cost

def calculate_cost_gradient(W, X_batch, Y_batch):
    if type(Y_batch) == np.float64:
        Y_batch = np.array([Y_batch])
        X_batch = np.array([X_batch])
    distance = 1 - (Y_batch * np.dot(X_batch, W))
    dw = np.zeros(len(W))

    for ind, d in enumerate(distance):
        if max(0, d) == 0:
            di = W
        else:
            di = W - (reg_strength * Y_batch[ind] * X_batch[ind])
        dw += di
    dw = dw/len(Y_batch)
    return dw
def sgd(features, outputs):
    max_epochs = 5000
    weights = np.zeros(features.shape[1])
    nth = 0
    prev_cost = float("inf")
    cost_threshold = 0.01
    for epoch in range(1, max_epochs):
        X, Y = shuffle(features, outputs)
        for ind, x in enumerate(X):
            ascent = calculate_cost_gradient(weights, x, Y[ind])
            weights = weights - (learning_rate * ascent)
        if epoch == 2 ** nth or epoch == max_epochs - 1:
            cost = compute_cost(weights, features, outputs)
            #print("Epoch is: {} and Cost is: {}".format(epoch, cost))
            if abs(prev_cost - cost) < cost_threshold * prev_cost:
                return weights
            prev_cost = cost
            nth += 1
    return weights

avgMetrics = {}

def init2(files, num):
    print("reading dataset...")


    runs = num
    filesNames = files
    resultsDF = pd.DataFrame(columns=["file name", "accuracy", "precision", "recall"])
    check = 0
    rowI = 0
    for fileName in filesNames:
        fn = fileName
        accuracySum = 0
        precisionSum = 0
        recallSum = 0

        for i in range(runs):
            data2 = pd.read_csv(fileName)

            

            #print("i: ", i)

            data2.drop(data2.columns[[-1, 0]], axis=1, inplace=True)
            #print("applying feature engineering...")
            diagnosis_map = {'M':1.0, 'B':-1.0}
            data2['diagnosis'] = data2['diagnosis'].map(diagnosis_map)
            

            Y2 = data2.loc[:, 'diagnosis']  # all rows of 'diagnosis' 
            X2 = data2.iloc[:, 1:]  # all rows of column 1 and ahead (features)
            
            remove_correlated_features(X2)
            remove_less_significant_features(X2,Y2)

            X2_normalized = MinMaxScaler().fit_transform(X2.values)
            X2 = pd.DataFrame(X2_normalized)

            X2.insert(loc=len(X2.columns), column='intercept', value=1)
            #print("splitting dataset into train and test sets...")
            X2_train, X2_test, y2_train, y2_test = tts(X2, Y2, test_size=0.2, random_state=42)
            

            #print("training started...")
            W2 = sgd(X2_train.to_numpy(), y2_train.to_numpy())
            #print("training finished.")
            #print("weights are: {}".format(W))

            #print("testing the model...")
            y2_test_predicted = np.array([])
            for i in range(X2_test.shape[0]):
                yp2 = np.sign(np.dot(W2, X2_test.to_numpy()[i]))
                y2_test_predicted = np.append(y2_test_predicted, yp2)

            y2_test_predicted = np.array([])
            for i in range(X2_test.shape[0]):
                yp2 = np.sign(np.dot(X2_test.to_numpy()[i], W2))
                y2_test_predicted = np.append(y2_test_predicted, yp2)
            print("accuracy on test dataset : {}".format(accuracy_score(y2_test.to_numpy(), y2_test_predicted)))
            print("recall on test dataset: {}".format(recall_score(y2_test.to_numpy(), y2_test_predicted)))
            print("precision on test dataset: {}".format(precision_score(y2_test.to_numpy(), y2_test_predicted)))
            accuracyDF = accuracy_score(y2_test.to_numpy(), y2_test_predicted)
            precisionDF = precision_score(y2_test.to_numpy(), y2_test_predicted)
            recallDF = recall_score(y2_test.to_numpy(), y2_test_predicted)
            

            accuracySum += accuracyDF
            precisionSum += precisionDF
            recallSum += recallDF

            resultsDF.loc[rowI] = [fileName, accuracyDF, precisionDF, recallDF]
            
            rowI += 1
        avgAccuracy = accuracySum / runs
        avgPrecision = precisionSum / runs
        avgRecall = recallSum / runs

        print("average accuracy for this file: ", avgAccuracy)
        print("average precision for this file: ", avgPrecision)
        print("average recall for this file: ", avgRecall)

        avgMetrics[fn] = {"accuracy": avgAccuracy, "precision": avgPrecision, "recall": avgRecall}
    check += 1
    avgMetricsDF = pd.DataFrame.from_dict(avgMetrics, orient="index")
    avgMetricsDF.to_excel("svmAVGMetrics.xlsx")
    resultsDF.to_excel("svmResults.xlsx", index=False)

    print("Thank you for using this program! Your results have been saved to files svmAVGMetrics.xlsx and svmResults.xlsx")

    

def init1(file):
    print("reading dataset...")
    data = pd.read_csv(file)

    data.drop(data.columns[[-1, 0]], axis=1, inplace=True)
    print("applying feature engineering...")
    diagnosis_map = {'M':1.0, 'B':-1.0}
    data['diagnosis'] = data['diagnosis'].map(diagnosis_map)
    

    Y = data.loc[:, 'diagnosis']  # all rows of 'diagnosis' 
    X = data.iloc[:, 1:]  # all rows of column 1 and ahead (features)
    
    remove_correlated_features(X)
    remove_less_significant_features(X,Y)

    X_normalized = MinMaxScaler().fit_transform(X.values)
    X = pd.DataFrame(X_normalized)

    X.insert(loc=len(X.columns), column='intercept', value=1)
    print("splitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = tts(X, Y, test_size=0.2, random_state=42)
    

    print("training started...")
    W = sgd(X_train.to_numpy(), y_train.to_numpy())
    print("training finished.")
    print("weights are: {}".format(W))

    print("testing the model...")
    y_test_predicted = np.array([])
    for i in range(X_test.shape[0]):
        yp = np.sign(np.dot(W, X_test.to_numpy()[i]))
        y_test_predicted = np.append(y_test_predicted, yp)

    y_test_predicted = np.array([])
    for i in range(X_test.shape[0]):
        yp = np.sign(np.dot(X_test.to_numpy()[i], W))
        y_test_predicted = np.append(y_test_predicted, yp)
    print("accuracy on test dataset: {}".format(accuracy_score(y_test.to_numpy(), y_test_predicted)))
    print("recall on test dataset: {}".format(recall_score(y_test.to_numpy(), y_test_predicted)))
    print("precision on test dataset: {}".format(precision_score(y_test.to_numpy(), y_test_predicted)))

def flipper(fPath, numF, cName):
    oPath = fPath
    numOfFiles = numF
    columnName = cName
    dfO = pd.read_csv(oPath)
    for i in range(numOfFiles):
        dfCopy = dfO.copy()
        labelChanges = int(input("Enter the number of labels to be flipped for copy #{}".format(i+1)))
        numRows = dfCopy.shape[0]
        rowsChanged = []
        while labelChanges > 0:
            rowIdx = random.randint(0, numRows - 1)
            currentValue = dfCopy.loc[rowIdx, columnName]

            if currentValue == "":
                newValue = ""
            elif currentValue == "":
                newValue = ""
            else:
                continue
            
            dfCopy.at[rowIdx, columnName] = newValue
            rowsChanged.append((rowIdx, currentValue))
            labelChanges -= 1
        print("For copy #{}, the following labels have been changed:".format(i+1))
        for label in labelChanges:
            print("Label {}, value: {}".format(label[0], label[1]))

        modifiedPath = "dataFlipped_{}.csv".format(i+1)
        dfCopy.to_csv(modifiedPath, index=False)


reg_strength = 10000
learning_rate = 0.000001


print("Welcome User, to Label Flipping Attack's Effects on Machine Learning Models")
print("Label flipping attacks attempt to manipulate the training data of the SVM ")
print("in an attempt to control the prediction behavior of the model.")
print("The goal of this program is to SIMULATE a label flipping attack")
print("Would you like to test the Support Vector Machine (SVM) or test run a Label Flipping Attack?")
print("NOTE: Please have an csv file ready to use for the SVM!!")
svmOrFlip = input("For testing the SVM, please input SVM. For performing an attack, please input ATTACK: ")

if svmOrFlip == "SVM":
    print("Provide the csv file that you wish to use.")
    print("Please make sure the file you wish to use is in the same directory as this program")
    print("Be sure to include .csv in your filename")
    fileDesired = input("Enter the filename now: ")
    init1(fileDesired)
elif svmOrFlip == "ATTACK":
    print("If you have a csv file with labels flipped already, enter n")
    print("otherwise, enter y to begin random label flipping")
    flipYN = input("(y,n): ")
    if flipYN == "n":
        numFiles = int(input("Please input the number of files you would like to use for this simulation: "))
        namesOfFiles = []
        for i in range(numFiles):
            fName = input("Please enter filename {} of {}: ".format(i+1, numFiles))
            namesOfFiles.append(fName)
        print("This mode is designed to evaluate the average accuracy, precision, and recall of the SVM")
        print("therefore the average shall be determined by how many times the SVM runs.")
        print("If you would like the most accurate average, input 100. If you would like to run less, input a number between 1-100")
        print("NOTE: more runs means more time. Keep in mind the computational speed of your machine and your available time!")
        numRuns = int(input("Please input the amount of times you would like the SVM to run each file: "))
        init2(namesOfFiles, numRuns)
    elif flipYN == "y":
        fPath = input("Please enter the name of the csv file, include the .csv extension: ")
        print("Now determine how many files you would like to create")
        print("The files created will be copies of your original csv file, however they will have some labels flipped")
        numF = int(input("How many files would you like to create?: "))  
        cLum = input("Please input the name of the column that has your labels")      
        flipper(fPath, numF, cLum)
    else:
        print("Invalid choice, please try again.")
else:
    print("Invalid choice. Please try again.")