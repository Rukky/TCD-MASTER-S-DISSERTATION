from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
from textblob import TextBlob
from nltk import tokenize

df = pd.read_csv('DOGE/dogecoin.csv')

df['Title'] = df['Title'].astype('str')

def get_polarity(text):
    return TextBlob(text).sentiment.polarity

df['Polarity'] = df['Title'].apply(get_polarity)

df.head()