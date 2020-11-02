# Hkxx26 Machine Learning Coursework

"""

How to Use:

1) We require the following libraries dependencies to be installed to run this file classifer.py
    - python3 : At least version 3.5 or newer.
    - pandas : pip install pandas
    - seaborn : pip install seaborn
    - numpy : pip install numpy
    - sklearn : pip install scikit-learn
    - seaborn : pip install seaborn
    - matplotlib : pip install matplotlib

2) Download the dataset.
    - Download the OULAD dataset from: 'https://analyse.kmi.open.ac.uk/open_dataset'
    - Make sure the directory 'anonymisedData' from the dataset is in the SAME directory as this file.
        - We will be using the file './anonymisedData/studentInfo.csv'

3) classifier.py
    - Now you're ready to run the code.
    - In your chosen terminal 'cd' to the directory containing this file, 'classifer.py'.
    - Type the command 'python classifier.py' to run the program.
    - All feature graphs produced will be saved in the 'graphs' directory. (Will be created if not found)
        - The graphs in this directory will contain all the feature graphs.
    - All other graphs from the Machine Learning model and Heatmap is saved in the 'models' directory.
        - Graphs saved in the 'graphs' folder are:
            - HeatMap.png (Heatmap of the features)
            - Logistic_Regression_NCM.png (Normalised Confusion Matrix)
            - Logistic_Regression_ROC.png (ROC Curve)
            - Random_Forest_NCM.png (Normalised Confusion Matrix)
            - Random_Forest_ROC.png (ROC Curve)
            - Random_Forest_Tree.png (A Random Forest Tree)

4) TROUBLESHOOTING
    - The program has been tested on macOS, Linux, Windows Machines and most importantly Mira.
    - But if errors occur it most likely is due to insufficient permissions in creating the folders to store the graphs.
    - If this is the case then in the same directory as 'classifer.py' create the directories:
        - graphs
        - models

"""

""" Imports """

# Imports pandas for data preparation.
import pandas as pd
import numpy as np
import seaborn as sb
import os

""" If running over SSH (I.E. Mira) need to disable showing plots (TKinter Error) uncomment the lines below"""
# import matplotlib as mpl
#
# if os.environ.get('DISPLAY', '') == '':
#     print('no display found. Using non-interactive Agg backend')
#     mpl.use('Agg')

import matplotlib.pyplot as plt


# Importing scikit-learn to split the data into training and testing sets
from sklearn.model_selection import train_test_split

# Imports Logistic Regression from scikit-learn.
from sklearn.linear_model import LogisticRegression

# Imports Random Forest from scikit-learn.
from sklearn.ensemble import RandomForestClassifier

# Saving a random forest tree to rfsampletree.png
from sklearn.tree import export_graphviz

# Imports ROC curve
from sklearn.metrics import roc_curve, auc

# Imports classification report to display recall and precision data
from sklearn.metrics import classification_report

""" Data Preparation """


# Reads in data from file.
def readData(file):
    return pd.read_csv(file)


# Creates a graph of the data features.
def graph(dataset, feature1, feature2):
    table = pd.crosstab(dataset[feature1], dataset[feature2])
    table.div(table.sum(1).astype(float), axis=0).plot(kind='bar')
    plt.title('{0} vs {1}'.format(feature1.replace("_", " "), feature2.replace("_", " ")))
    plt.ylabel(feature2)
    plt.xlabel(feature1)

    plt.savefig('graphs/{0}_graph.png'.format(feature1), dpi=100)
    plt.show()


# Removes feature from dataset.
def removeFeature(dataset, feature):
    return dataset.drop(columns=[feature], axis=1, inplace=True)


# Removes feature from dataset.
def convertFeature(dataset, feature, entryFrom, entryToo):
    for index in range(len(dataset[feature])):
        if dataset.iloc[index][9] == entryFrom:
            dataset.iloc[index, dataset.columns.get_loc(feature)] = entryToo
    return dataset


# Removes entry from dataset.
def removeEntries(dataset, feature, entry):
    return dataset[dataset[feature] != entry]


# Displays dataset.
def displayData(dataset, feature):
    print(dataset.groupby(feature).count())


# Produces a heat-map of the data to show missing data.
def heatMap(dataset):
    plt.tight_layout()
    sb.heatmap(dataset.isnull(), yticklabels=False, cbar=False, cmap='viridis')

    roc = plt.gcf()
    plt.show()
    roc.savefig('models/HeatMap.png', dpi=100)


# Drops missing data from dataset.
def dropMissingData(dataset):
    return dataset.dropna()


# Balances Data Pass and Withdraw Data to the same amount.
def balanceData(dataset):
    min_count = dataset.groupby("final_result").count().min()[1]
    max_count = dataset.groupby("final_result").count().max()[1]
    count = 0
    i = 0
    while count < (max_count - min_count):
        if dataset.iloc[i][9] == "Pass":
            dataset.drop(index=dataset.index[i], inplace=True)
            count += 1
        i += 1

    return dataset


# One Hot encodes the data.
def oneHotEncoding(dataset):
    # Replace student results 'Pass' and 'Fail' in final_result with 1 and 0.
    dataset["final_result"] = dataset["final_result"].replace("Pass", 1)
    dataset["final_result"] = dataset["final_result"].replace("Withdrawn", 0)

    # One hot encodes the data.
    one_hot_encoded = pd.get_dummies(dataset)

    return one_hot_encoded


# Returns the labels, feature names and feature data.
def saveData(dataset):
    # Stores Labels
    label = np.array(dataset['final_result'])

    # Removes the label column.
    feature = dataset.drop('final_result', axis=1)

    # Stores Feature Names.
    feature_name = list(feature.columns)

    # Stores Feature Data.
    features_data = np.array(feature)

    return label, feature_name, features_data


""" Splitting up the data """


# Separating the data into two partitions, Training (75%) and Test (25%) data.
def splitData(feature, label):
    return train_test_split(feature, label, test_size=0.25)


""" Machine Learning Methods """


# Returns a logistic regression model trained on training features: 'data' and labels: 'labels'.
def logisticRegression(feature, label, iteration):
    return LogisticRegression(max_iter=iteration).fit(feature, label)


# Returns a Random Forest model trained on training features: 'data' and labels: 'labels'. Uses Bootstrap parameter method.
def randomForest(feature, label, tree, max_depth):
    return RandomForestClassifier(n_estimators=tree, max_depth=max_depth, bootstrap=True).fit(feature, label)


""" Save Random Forest Tree """


# Saves a Random Forest Tree.
def saveRandomForestTree(model, model_name, feature_name):
    export_graphviz(model.estimators_[5], out_file='tree.dot', feature_names=feature_name, rounded=True, precision=1)
    os.system('dot -Tpng tree.dot -o models/{0}_Tree.png'.format(model_name.replace(" ", "_")))
    os.remove("tree.dot")


""" Display Statistics """


# Calculates the accuracy of the logistic regression model.
def displayAccuracy(model, model_name, feature, label):
    logistic_score = model.score(feature, label)
    print("{0} Accuracy: {1}%\n".format(model_name, (logistic_score * 100).__round__(2)))


def plot_confusion_matrix(df_confusion, model_name, cmap=plt.cm.gray_r):
    plt.matshow(df_confusion, cmap=cmap)
    plt.title('{0}: Normalised Confusion Matrix'.format(model_name))
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)

    roc = plt.gcf()

    plt.show()
    roc.savefig('models/{0}_NCM.png'.format(model_name.replace(" ", "_")), dpi=100)


# Displays The ROC curve and saves it in png format.
def displayROC(model, model_name, feature, label):
    model_score = model.predict_proba(feature)[:, 1]
    # False Positive Rate, True Positive Rate.
    fpr, tpr, threshold = roc_curve(label, model_score)
    roc_auc = auc(fpr, tpr)

    # Create the plot graph
    plt.figure(figsize=(10, 10))
    plt.plot([0, 1], [0, 1], ls='--')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.title('{0} ROC Curve'.format(model_name))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.grid(True)
    plt.plot(fpr, tpr, label='Area Under Curve = {0}'.format(roc_auc))
    plt.legend(loc="lower right", shadow=True, fancybox=True)

    roc = plt.gcf()

    plt.show()
    roc.savefig('models/{0}_ROC.png'.format(model_name.replace(" ", "_")), dpi=100)


# Displays the statistics of the Machine Learning Module to the console.
def displayStatistics(model, model_name, feature, label):
    print('\n---- Displaying Statistics for: {0} ----\n'.format(model_name))

    # Getting the test prediction labels from the test data.
    model_prediction = model.predict(feature)

    # Displays the model accuracy.
    displayAccuracy(model, model_name, feature, label)

    # Displays classification data - including precision and recall
    print('{0} Test report: \n'.format(model_name))
    print(classification_report(label, model_prediction))

    # Displays confusion matrix
    print('{0} Confusion Matrix: \n'.format(model_name))
    confusionMatrix = pd.crosstab(label, model_prediction, rownames=['Actual'], colnames=['Predicted'], margins=True)
    print(confusionMatrix)

    # Normalises the Confusion Matrix
    confusionMatrixNorm = pd.crosstab(label, model_prediction, rownames=['Actual'], colnames=['Predicted'])
    confusionMatrixNorm = confusionMatrixNorm / confusionMatrixNorm.sum(axis=1)
    plot_confusion_matrix(confusionMatrixNorm, model_name)

    # Displays ROC curve.
    displayROC(model, model_name, feature, label)


""" Importing and Preparing the data """

# Reads in the student data from studentInfo.csv
data = readData('./anonymisedData/studentInfo.csv')

# Displays the raw data in grouped by final_result.
print("\nRaw Data: ")
displayData(data, "final_result")

# Replaces entries where the results are "Distinction" or "Fail" to 'Pass' or 'Withdraw' (Respectively)
data = convertFeature(data, "final_result", "Distinction", "Pass")
data = convertFeature(data, "final_result", "Fail", "Withdrawn")

# Removes entries where the results are "Distinction" or "Fail" (Any Residual)
data = removeEntries(data, "final_result", "Distinction")
data = removeEntries(data, "final_result", "Fail")

# Creates a graphs directory if not already created to save the feature graphs.
if not os.path.isdir('graphs'):
    os.makedirs('graphs')

# Creates a models directory if not found for saving the model graphs.
if not os.path.isdir('models'):
    os.makedirs('models')

# Creates all the graphs of the features vs Final_result.
for feature in list(data.columns):
    if feature not in ['final_result', 'id_student']:
        graph(data, feature, 'final_result')

# Produces a heat-map of the data.
heatMap(data)

# Remove the features id_student and gender
removeFeature(data, "id_student")
removeFeature(data, "gender")

# Dropping rows with missing data.
data = dropMissingData(data)

# Balances the data by making equal Pass and Fail Entrees.
data = balanceData(data)

# Displays the processed data in grouped by final_result.
print("\nProcessed Data: ")
displayData(data, "final_result")

# One Hot Encodes the Data.
data_one_hot_encoded = oneHotEncoding(data)

# Saves the labels, feature names & feature data.
labels, feature_names, features = saveData(data_one_hot_encoded)

# Separating the data into two partitions, Training (75%) and Test (25%) data.
x_train, x_test, y_train, y_test = splitData(features, labels)

""" Creating & Displaying the Models """

# Getting the logistic regression model from the training features and label.
# Tweak the Logistic Regression Parameter iterations for tuning.
iterations = 1000
logistic_model = logisticRegression(x_train, y_train, iterations)

# Getting the Random Forest model from the training features and label.
# Tweak the Random Forest Parameters trees and depth for tuning.
trees = 500
depth = 7
randomForest_model = randomForest(x_train, y_train, trees, depth)

# Displays the statistics of the logistic model.
displayStatistics(logistic_model, 'Logistic Regression', x_test, y_test)

# # Displays the statistics of the Random Forest model.
displayStatistics(randomForest_model, 'Random Forest', x_test, y_test)

# Saves a Random Forest tree into RF_Tree.png
saveRandomForestTree(randomForest_model, 'Random Forest', feature_names)
