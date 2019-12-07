from read import Read
import pandas as pd
import numpy as np
import perceptron
import enum

# Lazy algorithms
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import Perceptron


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
    model_Perceptron = Perceptron(n_iter_no_change= 10)

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
    pca = PCA(n_components=4)
    pca.fit(trainData)
    trainData = pca.transform(trainData)
    testData = pca.transform(testData)

    model_MLP.fit(trainData, train['Survived'])
    trainScore = model_MLP.score(trainData, train['Survived'])*100
    testScore = model_MLP.score(testData, test['Survived'])*100
    print("(PCA) Multi-Layer Perceptron results:\n", "Train: ", trainScore, "%", "Test: ", testScore, "%")
    
    model_Perceptron.fit(trainData, train['Survived'])
    trainScore = model_Perceptron.score(trainData, train['Survived'])*100
    testScore = model_Perceptron.score(testData, test['Survived'])*100
    print("Perceptron Learning Algorithm results:\n", "Train: ", trainScore, "%", "Test: ", testScore, "%\n\n\n")


if __name__ == "__main__":
    main()