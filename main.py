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

    plt.show()
    plt.savefig("titanic_" + mode + "_" + algorithm + "_"  + str(num) + ".png")
    plt.close(fig)

def dropData(trainData, testData):

    # this block drops ALL cabin data from the data frame
    cols = [c for c in trainData.columns if c[:5] != 'Cabin']
    trainData = trainData[cols]
    cols = [c for c in testData.columns if c[:5] != 'Cabin']
    testData = testData[cols]

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
#    trainData = trainData[cols]
    cols = [c for c in testData.columns if c[:4] != 'Fare']
#    testData = testData[cols]

    # drop sibling/spouse
    cols = [c for c in trainData.columns if c[:5] != 'SibSp']
    trainData = trainData[cols]
    cols = [c for c in testData.columns if c[:5] != 'SibSp']
    testData = testData[cols]

    #drop parch
    cols = [c for c in trainData.columns if c[:5] != 'Parch']
    trainData = trainData[cols]
    cols = [c for c in testData.columns if c[:5] != 'Parch']
    testData = testData[cols]

    #drop Age
    cols = [c for c in trainData.columns if c[:3] != 'Age']
    trainData = trainData[cols]
    cols = [c for c in testData.columns if c[:3] != 'Age']
    testData = testData[cols]

    #drop title
    cols = [c for c in trainData.columns if c[:5] != 'Title']
#    trainData = trainData[cols]
    cols = [c for c in testData.columns if c[:5] != 'Title']
#    testData = testData[cols]

    #drop Pclass
    cols = [c for c in trainData.columns if c[:8] != 'Pclass_2']
#    trainData = trainData[cols]
    cols = [c for c in testData.columns if c[:8] != 'Pclass_2']
#    testData = testData[cols]


    return trainData, testData
def main():

    # Number of hiddens
    n=10

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
    model_MLP = MLPClassifier(hidden_layer_sizes=(n,n), activation='logistic', max_iter=2000)
    testScore = 0
    targetLabels = train['Survived'].values
    testLabels = test['Survived'].values

    trainData, testData = dropData(trainData, testData)
    # just uncomment the data you want to use

    trainFeatures = trainData
    testFeatures = testData

#    trainFeatures = train[["Pclass", "Age", "Sex", "Fare"]].values
#    testFeatures = test[["Pclass", "Age", "Sex", "Fare"]].values
  
    print(trainFeatures.shape)
    print(trainFeatures)
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
    print("Support vector results:\n", "Train: ", trainScore, "%", "Test: ", testScore, "%\n\n")

    # Now do Gaussian Naive Bayes
    trainScore = model_GNB.score(trainFeatures, targetLabels)*100
    testScore = model_GNB.score(testFeatures, testLabels)*100
    print("Gaussian NB results:\n", "Train: ", trainScore, "%", "Test: ", testScore, "%\n\n")

    # Now do Bernoulli Naive Bayes
#    trainScore = model_Bern.score(trainFeatures, targetLabels)*100
#    testScore = model_Bern.score(testFeatures, testLabels)*100
#    print("Bernoulli NB results:\n", "Train: ", trainScore, "%", "Test: ", testScore, "%")

    # Now do Multinomial Naive Bayes
#    trainScore = model_MNB.score(trainFeatures, targetLabels)*100
#    testScore = model_MNB.score(testFeatures, testLabels)*100
#    print("Multinomial NB results:\n", "Train: ", trainScore, "%", "Test: ", testScore, "%")

    # Now do MLP
    trainScore = model_MLP.score(trainFeatures, targetLabels)*100
    testScore = model_MLP.score(testFeatures, testLabels)*100
    print("Multi-Layer Perceptron results:\n", "Train: ", trainScore, "%", "Test: ", testScore, "%\n\n")
    
    # Now do Perceptron
    trainScore = model_Perceptron.score(trainFeatures, targetLabels)*100
    testScore = model_Perceptron.score(testFeatures, testLabels)*100
    print("Perceptron Learning Algorithm results:\n", "Train: ", trainScore, "%", "Test: ", testScore, "%\n\n\n\n")

    # Some PCA magic, reduce down to n parameters
    """ Everything below this point is using PCA for dimensionality reduction in the data

    """
    numComp = 2
    pca = PCA(n_components=numComp)
    pca.fit(trainFeatures)
    trainFeatures = pca.transform(trainFeatures)
    testFeatures = pca.transform(testFeatures)
    print(trainFeatures.shape)

    model.fit(trainFeatures, targetLabels)
    model_GNB.fit(trainFeatures, targetLabels)
    model_Bern.fit(trainFeatures, targetLabels)

    # Now do support vector machine
    trainScore = model.score(trainFeatures, targetLabels)*100
    testScore = model.score(testFeatures, testLabels)*100
    print("(PCA) Support vector results:\n", "Train: ", trainScore, "%", "Test: ", testScore, "%\n\n")

    # Now do Gaussian Naive Bayes
    trainScore = model_GNB.score(trainFeatures, targetLabels)*100
    testScore = model_GNB.score(testFeatures, testLabels)*100
    print("(PCA) Gaussian NB results:\n", "Train: ", trainScore, "%", "Test: ", testScore, "%\n\n")

    # Now do Bernoulli Naive Bayes
#    trainScore = model_Bern.score(trainFeatures, targetLabels)*100
#    testScore = model_Bern.score(testFeatures, testLabels)*100
#    print("Bernoulli NB results:\n", "Train: ", trainScore, "%", "Test: ", testScore, "%")

    model_MLP.fit(trainFeatures, targetLabels)
    trainScore = model_MLP.score(trainFeatures, targetLabels)*100
    testScore = model_MLP.score(testFeatures, testLabels)*100
    print("(PCA) Multi-Layer Perceptron results:\n", "Train: ", trainScore, "%", "Test: ", testScore, "%\n\n")
    
    model_Perceptron.fit(trainFeatures, targetLabels)
    trainScore = model_Perceptron.score(trainFeatures, targetLabels)*100
    testScore = model_Perceptron.score(testFeatures, testLabels)*100
    print("(PCA) Perceptron Learning Algorithm results:\n", "Train: ", trainScore, "%", "Test: ", testScore, "%\n\n\n")



#    fig = plt.figure(figsize = (8,8))
#    ax = fig.add_subplot(1,1,1)
#    ax.set_title("n="+str(numComp)+"  PCA")

#    colors = itertools.cycle(['r', 'g', 'b'])
#    plt.scatter(trainFeatures[:, 0], trainFeatures[:, 1], alpha = 0.5)
#    print(trainFeatures)
#    plt.show()
    plot(trainFeatures, targetLabels, "PCA", "MLP", numComp)


if __name__ == "__main__":
    main()