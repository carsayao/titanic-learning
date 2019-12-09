# Absolutely necessary
import pandas as pd
import numpy as np
import perceptron
import enum
import matplotlib.pyplot as plt

# The only important aspect of each passenger's name is their title.
# This indicates social status, and might help in the prediction
# of survivability.
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

class Read:

    # Extract title from passenger names, and assign general social
    # status title.
    def extractTitle(self, df):
        df = df.map(lambda n: n.split(',')[1].split('.')[0].strip())
        df = df.map(name_titles)
        return df

    # Extract important Cabin information from 'Cabin' column.
    # Replace NaN values with '?'.
    def extractCabin(self, df):
        df = df.fillna('?')
        df = df.map(lambda c: c[0])
        return df

    # Extract important Ticket information from 'Ticket' column.
    # Replace unimportant Ticket values with '?'.
    def extractTicket(self, df):
        df = df.replace('.', '')
        df = df.replace('/', '')
        df = df.split()
        df = map(lambda t: t.strip(), df)
        df = list(filter(lambda t: not t.isdigit(), df))
        if len(df) > 0:
            return df[0]
        else:
            return '?'

    # Runs through the extraction functions above in order to extract only
    # the important information from certain data columns.
    def fixData(self, df):
        df['Name'] = self.extractTitle(df['Name'])

        # Replace 'male' with 0 and 'female' with 1 in 'Sex' column.
        df['Sex'] = pd.Series(np.where(df['Sex'] == 'male', 0, 1),
                              name = 'Sex')
        # Extract important cabin info.
        df['Cabin'] = self.extractCabin(df['Cabin'])
        # Extract important ticket info.
        df['Ticket'] = df['Ticket'].map(self.extractTicket)
        # Replace NaN with mean of entire data set
        df['Age'] = df['Age'].fillna(df['Age'].mean())
        df['Fare'] = df['Fare'].fillna(df['Fare'].mean())

        return df

    def binaryRepresentation(self, df):
        # Additionally, replace age values with 1s or 0s. If they are older
        # than 50, put a 0, 1 otherwise
        df['Age'] = pd.Series(np.where(df['Age'] > 30, 0, 0), name = 'Age')
        
        # Do the same for all other data that isn't 1 or 0
        return df

    # Create a single pandas.DataFrame object which contains all the data
    #  we'll be using as input. Every column is a separate input node.
    def concatData(self, df):
        # Separate columns for every possible attribute value.
        # Example column names: 'Title_Mr', 'Title_Ms', 'Title_Upper', etc...
        # Row values = 0 if false, 1 if true.
        pclass = pd.get_dummies(df['Pclass'], prefix = 'Pclass')
        title = pd.get_dummies(df['Name'], prefix = 'Title')
        cabin = pd.get_dummies(df['Cabin'], prefix = 'Cabin')
        ticket = pd.get_dummies(df['Ticket'], prefix = 'Ticket')
        embarked = pd.get_dummies(df['Embarked'], prefix = 'Embarked')

        # This determines what values we end up using as input nodes.
        # Mess with this if you want to see the difference in accuracy %
        # based on what you use as input.
        return pd.concat([df['Pclass'], title, df['Sex'], df['Age'], df['SibSp'],
                          df['Parch'], ticket, df['Fare'], cabin, embarked],
                          axis = 1)
    
    def read(self):
        # Import training and testing data into pandas.DataFrame objects.
        train = pd.read_csv('titanic/rawdata/train.csv')
        test = pd.read_csv('titanic/rawdata/test.csv')
        train = self.fixData(train)
    #    train = binaryRepresentation(train)
        test = self.fixData(test)
        trainData = self.concatData(train)
        testData = self.concatData(test)
        # Ensure that both the training and testing dataframes have the same
        # columns. If we're adding any new columns through this, make all
        # values in that column = 0.
        trainData, testData = trainData.align(testData, join='outer',
                                              fill_value = 0, axis = 1)
        return trainData, testData, train, test


def main():
    # Instance
    trainData, testData, train, test = Read().read()
    print(train.shape)
    print(test.shape)
    ## If you run this main() by itself, it'll give you some pretty graphs
    df = pd.read_csv("titanic/rawdata/train.csv")
    fig = plt.figure(figsize = (20,10))
    
    plt.subplot2grid((2,3), (0,0))
    df.Survived.value_counts(normalize=True).plot(kind="bar", alpha=0.5)
    plt.title("Survived")
#    plt.show()

    plt.subplot2grid((2,3), (0,1))
    plt.scatter(df.Survived, df.Age, alpha=0.1)
    plt.title("Age with respect to survivability")

    plt.subplot2grid((2,3), (0,2))
    df.Pclass.value_counts(normalize=True).plot(kind="bar", alpha = 0.5)
    plt.title("Class")

#    plt.show()

    plt.subplot2grid((2,3), (1,0), colspan=2)
    for x in [1,2,3]:
        df.Age[df.Pclass == x].plot(kind="kde")
    plt.title("Class with respect to age")
    plt.legend(("1st", "2nd", "3rd"))

    plt.subplot2grid((2,3), (1,2))
    df.Embarked.value_counts(normalize=True).plot(kind="bar", alpha = 0.5)
    plt.title("Embarked")

    plt.subplot2grid((3,4), (0,0))
    df.Survived.value_counts(normalize=True).plot(kind="bar", alpha=0.5)
    plt.title("Survived")

   # plt.show()

    plt.subplot2grid((3,4), (0,1))
    df.Survived[df.Sex == "male"].value_counts(normalize=True).plot(kind="bar", alpha=0.5)
    plt.title("Men that survived")

    plt.subplot2grid((3,4), (0,2))
    df.Survived[df.Sex == "female"].value_counts(normalize=True).plot(kind="bar", alpha=0.5, color = "m")
    plt.title("Women that survived")

    plt.subplot2grid((3,4), (0,3))
    df.Sex[df.Survived == 1].value_counts(normalize=True).plot(kind="bar", alpha=0.5, color = ['m', 'b'])
    plt.title("Sex of survivors")

    plt.clf() # clear the plot
    plt.subplot2grid((3,4), (1,0), colspan=4)
    for x in [1,2,3]:
        df.Survived[df.Pclass == x].plot(kind="kde")
    plt.title("Class with respect to Survival")
    plt.legend(("1st", "2nd", "3rd"))
    plt.show() # show the plot

if __name__ == "__main__":
    main()