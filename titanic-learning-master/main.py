# Absolutely necessary
import pandas as pd
import numpy as np
import perceptron
import enum

# Lazy algorithms
from sklearn.svm import SVC

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
    # Replace passenger names with titles.
    df['Name'] = extractTitle(df['Name'])

    # Replace 'male' with 0 and 'female' with 1 in 'Sex' column.
    df['Sex'] = pd.Series(np.where(df['Sex'] == 'male', 0, 1), name = 'Sex')

    # Extract important cabin info.
    df['Cabin'] = extractCabin(df['Cabin'])

    # Extract important ticket info.
    df['Ticket'] = df['Ticket'].map(extractTicket)

    # Replace NaN with mean of entire data set.
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df['Fare'] = df['Fare'].fillna(df['Fare'].mean())

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
    test = fixData(test)
    mlp = perceptron.multiLayer(1)

    trainData = concatData(train)
    testData = concatData(test)

    # Ensure that both the training and testing dataframes have the same columns.
    # If we're adding any new columns through this, make all values in that column = 0.
    trainData, testData = trainData.align(testData, join='outer', fill_value = 0, axis = 1)

    # Run the training and testing data through an Sklearns model.
    model = SVC()
    model.fit(trainData, train['Survived'])
    print(model.score(trainData, train['Survived']), model.score(testData, test['Survived']))

 #   print(train['Sex'], "\n\n", train['Name'], "\n\n", train['Cabin'], "\n\n", train['Ticket'])

#    print(train['Ticket'][0])  
#    print(train['PassengerId'][0])
#    print(name_titles['Don'])






if __name__ == "__main__":
    main()