import pandas as pd
import numpy as np
import csv

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
def extractImportant(df):
    df['Name'] = extractTitle(df['Name'])
    df['Cabin'] = extractCabin(df['Cabin'])
    df['Ticket'] = df['Ticket'].map(extractTicket)
    return df

# Import training and testing data into pandas.DataFrame objects.
train = pd.read_csv('titanic/rawdata/train.csv')
test = pd.read_csv('titanic/rawdata/test.csv')

train = extractImportant(train)
print(train)

# 2 columns. Each possible 'Sex' value. 1 if true, 0 if false.
sex = pd.get_dummies(train['Sex'], prefix = 'Sex')

# 5 columns. Each possible 'Title' value. 1 if true, 0 if false.
title = pd.get_dummies(train['Name'], prefix = 'Title')

# 9 columns. Each possible 'Cabin' value. 1 if true, 0 if false.
cabin = pd.get_dummies(train['Cabin'], prefix = 'Cabin')

# 31 columns. Each possible 'Ticket' value. 1 if true, 0 if false.
ticket = pd.get_dummies(train['Ticket'], prefix = 'Ticket')