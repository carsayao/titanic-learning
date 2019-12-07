# Absolutely necessary
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
# The only important aspect of each passenger's name is their title. This indicates social status, and might
# help in the prediction of survivability.
name_titles = {
    'Mr':           'Mr',
    'Miss':         'Ms',
    'Ms':           'Ms',
    'Mme':          'Ms',
    'Mrs':          'Mrs',
    'Master':       'Upper',
    'Rev':          'Upper',
    'Dr':           'Upper',
    'Capt':         'Upper',
    'Col':          'Upper',
    'Major':        'Upper',
    'Don':          'Royalty',
    'Jonkheer':     'Royalty',
    'Sir':          'Royalty',
    'the Countess': 'Royalty',
    'Dona':         'Royalty',
    'Lady':         'Royalty'
}

# Extract title from passenger names, and assign general social status title.
def extractTitle(df):
    df = df.map(lambda n: n.split(',')[1].split('.')[0].strip())
    df = df.map(name_titles)
    return df

# Extract important Cabin information from 'Cabin' column. Replace NaN values with '?'.
def extractCabin(df):
    df = df.fillna('?')
    df = df.map(lambda c: c[0])
    return df

# Extract important Ticket information from 'Ticket' column. Replace unimportant Ticket values with '?'.
def extractTicket(df):
    df = df.replace('.', '')
    df = df.replace('/', '')
    df = df.split()
    df = map(lambda t: t.strip(), df)
    df = list(filter(lambda t: not t.isdigit(), df))
    if len(df) > 0:
        return df[0]
    else:
        return '?'

# Runs through the extraction functions above in order to extract only the important information from certain data columns.
def fixData(df):
    df['Name'] = extractTitle(df['Name'])

    # Replace 'male' with 0 and 'female' with 1 in 'Sex' column.
    df['Sex'] = pd.Series(np.where(df['Sex'] == 'male', 0, 1), name = 'Sex')
    # Extract important cabin info.
#    df['Cabin'] = extractCabin(df['Cabin'])
    # Extract important ticket info.
#    df['Ticket'] = df['Ticket'].map(extractTicket)
    # Replace NaN with mean of entire data set
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df['Fare'] = df['Fare'].fillna(df['Fare'].mean())


    return df

def binaryRepresentation(df):
    # Additionally, replace age values with 1s or 0s. If they are older than 50, put a 0, 1 otherwise
    df['Age'] = pd.Series(np.where(df['Age'] > 30, 0, 0), name = 'Age')
    
    # Do the same for all other data that isn't 1 or 0
    return df


# Create a single pandas.DataFrame object which contains all the data we'll be using as input.
# Every column is a separate input node.
def concatData(df):
    # Separate columns for every possible attribute value.
    # Example column names: 'Title_Mr', 'Title_Ms', 'Title_Upper', etc...
    # Row values = 0 if false, 1 if true.
    pclass = pd.get_dummies(df['Pclass'], prefix = 'Pclass')
    title = pd.get_dummies(df['Name'], prefix = 'Title')
    cabin = pd.get_dummies(df['Cabin'], prefix = 'Cabin')
    ticket = pd.get_dummies(df['Ticket'], prefix = 'Ticket')
    embarked = pd.get_dummies(df['Embarked'], prefix = 'Embarked')

    # This determines what values we end up using as input nodes.
    # Mess with this if you want to see the difference in accuracy % based on what you use as input.
    return pd.concat([pclass, title, df['Sex'], df['Age'], df['SibSp'], df['Parch'], ticket, df['Fare'], cabin, embarked], axis = 1)

def main():
    # Import training and testing data into pandas.DataFrame objects.
    train = pd.read_csv('titanic/rawdata/train.csv')
    test = pd.read_csv('titanic/rawdata/test.csv')
    train = fixData(train)
#    train = binaryRepresentation(train)
    test = fixData(test)
    mlp = perceptron.multiLayer(1)

    trainData = concatData(train)
    testData = concatData(test)
    n = 5

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

#    print("\n\n\n", trainData)




if __name__ == "__main__":
    main()