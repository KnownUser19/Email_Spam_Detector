from os import sep
from typing import Counter
import pandas as pd
import numpy as np
from sklearn.metrics  import accuracy_score
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

df = pd.read_csv("spam.csv", encoding="latin-1")

# first 5 rows
print(df.head())

# Last 5 rows
print(df.tail())

# Random 5 rows
print(df.sample(5))

# show columns
print(df.columns)

# drop specific columns
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],inplace=True)

#print the first five rows
print(df.head())

df.rename(columns={'class':'result', 'message':'input'}, inplace = True)

print(df.head())

# unique values and their count 
print(df['result'].value_counts())

print(df.isnull().sum())

print(df.duplicated().sum())

df = df.drop_duplicates()
#  final check that duplicated rows are deleted or not
print(df.duplicated().sum())

# this many sms are important for classification
print(df.shape)

import matplotlib.pyplot as plt

plt.pie(df['result'].value_counts(), labels = ['not spam', 'spam'],autopct = '%0.2f')
plt.show()



import nltk

#nltk.download('all')

# represent into numbers
df = df.replace({"ham": 0,"spam": 1})

print(df.head())

import string
import warnings
warnings.filterwarnings('ignore')

# Importing necessary libraries
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Download punkt tokenizer
nltk.download('punkt')

# Assuming df is your DataFrame
# Creating a new column with the count of characters
df['countCharacters'] = df['input'].apply(len)

# Creating a new column with the count of words
df['countWords'] = df['input'].apply(lambda i: len(nltk.word_tokenize(i)))

# Creating a new column with the count of sentences
df['countSentences'] = df['input'].apply(lambda i: len(nltk.sent_tokenize(i)))

# Printing the first few rows of the DataFrame
print(df.head())

# Descriptive statistics for non-SPAM messages (result == 0)
print(df[df['result'] == 0][['countCharacters', 'countWords', 'countSentences']].describe())

# Descriptive statistics for SPAM messages (result == 1)
print(df[df['result'] == 1][['countCharacters', 'countWords', 'countSentences']].describe())

# Visualization using seaborn
plt.figure(figsize=(15, 15))

# Plotting histograms for non-SPAM and SPAM messages
sns.histplot(df[df['result'] == 0]['countCharacters'], color='yellow', kde=True, label='Non-SPAM')
sns.histplot(df[df['result'] == 1]['countCharacters'], color='black', kde=True, label='SPAM')

# Adding legend and title
plt.legend()
plt.title('Distribution of Character Counts in SPAM and Non-SPAM Messages')
plt.xlabel('Character Count')
plt.ylabel('Frequency')

# Pairplot for better visualization
sns.pairplot(df, hue='result')

# Show the plots
plt.show()



#nltk.download('stopwords')


import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Create stopwords list
stopwords_list = stopwords.words("english")

# Define the transform_text function
def transform_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Tokenization
    words = nltk.word_tokenize(text)
    
    # Removing special characters (keep only alphanumeric tokens)
    words = [word for word in words if word.isalnum()]
    
    # Removing stop words and punctuation
    words = [word for word in words if word not in stopwords_list and word not in string.punctuation]
    
    # Stemming the data using PorterStemmer
    ps = PorterStemmer()
    words = [ps.stem(word) for word in words]
    
    # Join processed words back into a string
    return " ".join(words)



# Apply the transformation to a DataFrame column
df['processed'] = df['input'].apply(transform_text)

# Print the first few rows of the DataFrame
print(df.head())


# extracting the most common words used in both SPAM and not SPAM 

spamWords = list()

for msg in df[df['result'] == 1]['processed'].tolist():
    for word in msg.split():
        spamWords.append(word)
        
print(spamWords)

# to count the frequency of the words
spamWordsDictionary = Counter(spamWords)
spamWordsDictionary.most_common(40)

mostCommonSPAM = pd.DataFrame(spamWordsDictionary.most_common(40))

plt.figure(figsize=(12,6))
sns.barplot(data = mostCommonSPAM, x=0, y=1)
plt.xticks(rotation = 'vertical')
plt.show()



cv = CountVectorizer()
tf = TfidfVectorizer()

# Fit and transform the data
X = cv.fit_transform(df['processed']).toarray()
print(X.shape)

# Target variable
y = df['result'].values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=49)

# Instantiate Naive Bayes models
gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()

# Gaussian Naive Bayes
gnb.fit(X_train, y_train)
y_pred1 = gnb.predict(X_test)
print(accuracy_score(y_test, y_pred1))
print(confusion_matrix(y_test, y_pred1))
print(precision_score(y_test, y_pred1))

# Multinomial Naive Bayes
mnb.fit(X_train, y_train)
y_pred2 = mnb.predict(X_test)
print(accuracy_score(y_test, y_pred2))
print(confusion_matrix(y_test, y_pred2))
print(precision_score(y_test, y_pred2))

# Bernoulli Naive Bayes
bnb.fit(X_train, y_train)
y_pred3 = bnb.predict(X_test)
print(accuracy_score(y_test, y_pred3))
print(confusion_matrix(y_test, y_pred3))
print(precision_score(y_test, y_pred3))

import pickle

# Save the vectorizer and model using pickle

pickle.dump(tf, open('vectorizer.pkl', 'wb')) 
pickle.dump(mnb, open('model.pkl', 'wb')) 