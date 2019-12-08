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
    testScore = 0
    targetLabels = train['Survived'].values
    testLabels = test['Survived'].values


    # just uncomment the data you want to use

    #trainFeatures = trainData
    #testFeatures = testData

    trainFeatures = train[["Pclass", "Age", "Sex", "Fare"]].values
    testFeatures = test[["Pclass", "Age", "Sex", "Fare"]].values

    # 3 hidden layers (n, n, n)
    model_Perceptron = Perceptron(n_iter_no_change= 10)

    model.fit(trainFeatures, targetLabels)
    model_GNB.fit(trainFeatures, targetLabels)
    model_Bern.fit(trainFeatures, targetLabels)
    model_MNB.fit(trainFeatures, targetLabels)
    model_MLP.fit(trainFeatures, targetLabels)
    model_Perceptron.fit(trainFeatures, targetLabels)

    # Let's test a few different models ..
  
    # Now do support vector machine
    trainScore = model.score(trainFeatures, targetLabels)*100
    testScore = model.score(testFeatures, testLabels)*100
    print("Support vector results:\n", "Train: ", trainScore, "%", "Test: ", testScore, "%")

    # Now do Gaussian Naive Bayes
    trainScore = model_GNB.score(trainFeatures, targetLabels)*100
    testScore = model_GNB.score(testFeatures, testLabels)*100
    print("Gaussian NB results:\n", "Train: ", trainScore, "%", "Test: ", testScore, "%")

    # Now do Bernoulli Naive Bayes
    trainScore = model_Bern.score(trainData, targetLabels)*100
    testScore = model_Bern.score(testData, testLabels)*100
    print("Bernoulli NB results:\n", "Train: ", trainScore, "%", "Test: ", testScore, "%")

    # Now do Multinomial Naive Bayes
    trainScore = model_MNB.score(trainData, targetLabels)*100
    testScore = model_MNB.score(testData, testLabels)*100
    print("Multinomial NB results:\n", "Train: ", trainScore, "%", "Test: ", testScore, "%")

    # Now do MLP
    trainScore = model_MLP.score(trainFeatures, targetLabels)*100
    testScore = model_MLP.score(testFeatures, testLabels)*100
    print("Multi-Layer Perceptron results:\n", "Train: ", trainScore, "%", "Test: ", testScore, "%")
    
    # Now do Perceptron
    trainScore = model_Perceptron.score(trainFeatures, targetLabels)*100
    testScore = model_Perceptron.score(testFeatures, testLabels)*100
    print("Perceptron Learning Algorithm results:\n", "Train: ", trainScore, "%", "Test: ", testScore, "%\n\n\n")

    # Some PCA magic, reduce down to n parameters
    pca = PCA(n_components=4)
    pca.fit(trainData)
    trainData = pca.transform(trainData)
    testData = pca.transform(testData)

    model_MLP.fit(trainData, train['Survived'])
    trainScore = model_MLP.score(trainData, targetLabels)*100
    testScore = model_MLP.score(testData, testLabels)*100
    print("(PCA) Multi-Layer Perceptron results:\n", "Train: ", trainScore, "%", "Test: ", testScore, "%")
    
    model_Perceptron.fit(trainData, targetLabels)
    trainScore = model_Perceptron.score(trainData, targetLabels)*100
    testScore = model_Perceptron.score(testData, testLabels)*100
    print("(PCA) Perceptron Learning Algorithm results:\n", "Train: ", trainScore, "%", "Test: ", testScore, "%\n\n\n")


if __name__ == "__main__":
    main()