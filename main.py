from read import Read
import pandas as pd
import numpy as np
import perceptron
import enum
import matplotlib.pyplot as plt

# Lazy algorithms
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import Perceptron

def plot(array, predicts, mode, algorithm, num):
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1)
    #ax.set_title("n="+str(num)+"  PCA")
    ax.set_title(mode + " " + algorithm + " n="+str(num))
#    colors = itertools.cycle(['r', 'g', 'b'])
    scatter = ax.scatter(array[:, 0], array[:, 1], c=predicts, alpha = 0.5)
    legend = ax.legend(*scatter.legend_elements(), loc="upper right")
    ax.add_artist(legend)

    plt.savefig("titanic_" + mode + "_" + algorithm + "_"  + str(num) + ".png")
    plt.close(fig)

def dropData(trainData, testData):

    """ Data can be put back into the dataframe by commenting out the assignments
    """
    # this block drops ALL cabin data from the data frame
    cols = [c for c in trainData.columns if c[:5] != 'Cabin']
#    trainData = trainData[cols]
    cols = [c for c in testData.columns if c[:5] != 'Cabin']
#    testData = testData[cols]

    # drops ALL embarkment locations
    cols = [c for c in trainData.columns if c[:8] != 'Embarked']
    trainData = trainData[cols]
    cols = [c for c in testData.columns if c[:8] != 'Embarked']
    testData = testData[cols]

    # drops ALL tickets
    cols = [c for c in trainData.columns if c[:6] != 'Ticket']
    trainData = trainData[cols]
    cols = [c for c in testData.columns if c[:6] != 'Ticket']
    testData = testData[cols]

    # drops ALL sex
    cols = [c for c in trainData.columns if c[:3] != 'Sex']
#    trainData = trainData[cols]
    cols = [c for c in testData.columns if c[:3] != 'Sex']
#    testData = testData[cols]
    
    # drops ALL Fare
    cols = [c for c in trainData.columns if c[:4] != 'Fare']
    trainData = trainData[cols]
    cols = [c for c in testData.columns if c[:4] != 'Fare']
    testData = testData[cols]

    # drop sibling/spouse
#    cols = [c for c in trainData.columns if c[:5] != 'SibSp']
    trainData = trainData[cols]
#    cols = [c for c in testData.columns if c[:5] != 'SibSp']
    testData = testData[cols]

    #drop parch
    cols = [c for c in trainData.columns if c[:5] != 'Parch']
#    trainData = trainData[cols]
    cols = [c for c in testData.columns if c[:5] != 'Parch']
#    testData = testData[cols]

    #drop Age
    cols = [c for c in trainData.columns if c[:3] != 'Age']
    trainData = trainData[cols]
    cols = [c for c in testData.columns if c[:3] != 'Age']
    testData = testData[cols]

    #drop title
    cols = [c for c in trainData.columns if c[:5] != 'Title']
    trainData = trainData[cols]
    cols = [c for c in testData.columns if c[:5] != 'Title']
    testData = testData[cols]

def main():

    # Number of hiddens
    n=5

    # Instance of Read class to read in data
    trainData, testData, train, test = Read().read()

    # Ensure that both the training and testing dataframes have the same columns.
    # If we're adding any new columns through this, make all values in that column = 0.
    trainData, testData = trainData.align(testData, join='outer', fill_value = 0, axis = 1)

    # Run the training and testing data through an Sklearns model.
    model = SVC()
    model_GNB = GaussianNB()
    model_Bern = BernoulliNB()
    model_MNB = MultinomialNB()
    model_MLP = MLPClassifier(hidden_layer_sizes=(n,n,n), activation='logistic', max_iter=2000)
    # 3 hidden layers (n, n, n)
    testScore = 0
    targetLabels = train['Survived'].values
    testLabels = test['Survived'].values

    model.fit(trainData, train['Survived'])
    model_GNB.fit(trainData, train['Survived'])
    model_Bern.fit(trainData, train['Survived'])
    model_MNB.fit(trainData, train['Survived'])
    model_MLP.fit(trainData, train['Survived'])
    model_Perceptron.fit(trainData, train['Survived'])

    # Let's test a few different models ..
  
    # Now do support vector machine
    trainScore = model.score(trainData, train['Survived'])*100
    testScore = model.score(testData, test['Survived'])*100
    print("Support vector results:\n", "Train: ", trainScore, "%", "Test: ", testScore, "%")

    # Now do Gaussian Naive Bayes
    trainScore = model_GNB.score(trainData, train['Survived'])*100
    testScore = model_GNB.score(testData, test['Survived'])*100
    print("Gaussian NB results:\n", "Train: ", trainScore, "%", "Test: ", testScore, "%")

    # Now do Bernoulli Naive Bayes
    trainScore = model_Bern.score(trainData, train['Survived'])*100
    testScore = model_Bern.score(testData, test['Survived'])*100
    print("Bernoulli NB results:\n", "Train: ", trainScore, "%", "Test: ", testScore, "%")

    # Now do Multinomial Naive Bayes
    trainScore = model_MNB.score(trainData, train['Survived'])*100
    testScore = model_MNB.score(testData, test['Survived'])*100
    print("Multinomial NB results:\n", "Train: ", trainScore, "%", "Test: ", testScore, "%")

    # Now do MLP
    trainScore = model_MLP.score(trainData, train['Survived'])*100
    testScore = model_MLP.score(testData, test['Survived'])*100
    print("Multi-Layer Perceptron results:\n", "Train: ", trainScore, "%", "Test: ", testScore, "%")
    
    # Now do Perceptron
    trainScore = model_Perceptron.score(trainData, train['Survived'])*100
    testScore = model_Perceptron.score(testData, test['Survived'])*100
    print("Perceptron Learning Algorithm results:\n", "Train: ", trainScore, "%", "Test: ", testScore, "%\n\n\n")



    # Some PCA magic, reduce down to n parameters

    numComp = 2

    pca = PCA(n_components=numComp)
    pca.fit(trainData)
    trainData = pca.transform(trainData)
    testData = pca.transform(testData)

    model_MLP.fit(trainData, train['Survived'])
    trainScore = model_MLP.score(trainData, train['Survived'])*100
    testScore = model_MLP.score(testData, test['Survived'])*100
    print("(PCA) Multi-Layer Perceptron results:\n", "Train: ", trainScore, "%", "Test: ", testScore, "%")
    #predicts = pca.score_samples(testData)
    plot(testData, test['Survived'], "PCA", "MLP", numComp)
    
    model_Perceptron.fit(trainData, train['Survived'])
    trainScore = model_Perceptron.score(trainData, train['Survived'])*100
    testScore = model_Perceptron.score(testData, test['Survived'])*100
    print("Perceptron Learning Algorithm results:\n", "Train: ", trainScore, "%", "Test: ", testScore, "%\n\n\n")


if __name__ == "__main__":
    main()
