import pandas as pd
import re
import datetime
import numpy as np
import nltk

nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

crypto_name = "ADA"


def clean_tweet(tweet):
    '''
    Utility function to clean tweet text by removing links, special characters
    using simple regex statements.
    '''
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())


def get_tweet_sentiment(tweet):
    '''
    Utility function to classify sentiment of passed tweet
    using textblob's sentiment method
    '''
    # create TextBlob object of passed tweet text
    analysis = TextBlob(tweet)

    return analysis.sentiment.polarity


def truncate_time(time):
    return time.replace(hour=0, minute=0, second=0, microsecond=0)


sid = SentimentIntensityAnalyzer()


def vader(tweet):
    ss = sid.polarity_scores(tweet)
    return (ss['compound'])


def pos_or_not(tweet_sentiment):
    if (tweet_sentiment > 0):
        return 1
    else:
        return 0


tweets = pd.read_csv('C:/Users/Musta/PycharmProjects/Bachelorarbeit/data/twitter_data/twitter_{0}_06_16.csv'.format(crypto_name))
print(len(tweets))

tweets['hashtags'] = tweets['hashtags'].str.lower()
tweets['tweet'] = tweets['tweet'].str.lower()

tweets = tweets.drop_duplicates(subset='tweet')
print(len(tweets))

# Clean the text
tweets['text'] = tweets['tweet'].map(clean_tweet)

# Polarity by vader
tweets['polarity_vader'] = tweets['text'].map(vader)

# Polarity by textblob
tweets['polarity_textblob'] = tweets['text'].map(get_tweet_sentiment)

# Create Timestamp
tweets['timestamp'] = pd.to_datetime(tweets['date'] + ' ' + tweets['time'])
# tweets['timestamp'] = tweets['timestamp'].map(truncate_time)
tweets['timestamp'] = pd.Series(tweets['timestamp']).dt.round('H')

print(tweets['timestamp'])
tweets = tweets[['text', 'timestamp', 'polarity_textblob', 'polarity_vader', 'hashtags']]
tweets = tweets.rename(columns={'text': 'Text', 'polarity_textblob': 'Polarity_Textblob', 'timestamp': 'Timestamp',
                                'polarity_vader': 'Polarity_Vader', 'hashtags': 'Hashtags'})

to_drop = ['#lottery', '#makemoney', '#free', '#bet', '#freeethereum', '#webbot', '#freeminingsoftware',
           '#yabtcl', '#ethereumbet', '#tradingtool', '#trading', '#residualethereum', '#faucet', '#casino', 'tradeethereum',
           '#sportsbook', '#game', '#simplefx', '#nitrogensportsbook', '#makemoney', '#makeyourownlane',
           '#ethereumprice', '#tradingeth', '#mpgvip', '#footballcoin', '#earnethereum', '#winethereum']

for i in to_drop:
    tweets = tweets[tweets.Hashtags.str.contains(i) == False]
print(len(tweets))

to_drop = ['free', 'trading', 'trade', 'win', 'gamble', 'performing currency', 'altcoin', 'fintech', 'betting']

for i in to_drop:
    tweets = tweets[tweets.Text.str.contains(i) == False]
print(len(tweets))

tweets_by_timestamp = tweets.groupby(['Timestamp'], as_index=False).mean()

count = pd.DataFrame(data=tweets.Timestamp.value_counts())
count.reset_index(inplace=True)
count.columns = ['Timestamp', 'Count_of_Tweets']
tweets_by_timestamp = pd.merge(tweets_by_timestamp, count)

tweets['Positive_Or_Not'] = tweets['Polarity_Vader'].map(pos_or_not)
tweets_count = tweets.groupby(['Timestamp'], as_index=False).sum()
tweets_by_timestamp['Count_Of_Positive_Tweets'] = tweets_count['Positive_Or_Not']
tweets_by_timestamp['Count_Of_Negative_Tweets'] = tweets_by_timestamp['Count_of_Tweets'] - tweets_count[
    'Positive_Or_Not']

print(tweets_by_timestamp)

df = pd.read_csv('C:/Users/Musta/PycharmProjects/Bachelorarbeit/data/market_data/{0}-USD_data.csv'.format(crypto_name))
df.drop(['conversionSymbol'], axis=1, inplace=True)
df = df.rename(columns={'time': 'Timestamp'})
print(df)
df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
print(df)
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# google_trends = pd.read_csv('Google_Trends_Data.csv')
# google_trends = google_trends.rename(columns={'date': 'Timestamp'})
# print(google_trends)
print(len(tweets))
print(len(df))
# print(len(google_trends))

data1 = pd.merge(tweets_by_timestamp, df, on='Timestamp', how='left')
# google_trends['Timestamp'] = pd.to_datetime(google_trends['Timestamp'])
# data = pd.merge(data1, google_trends, on='Timestamp', how='inner')
# data1.key.astype('datetime64[ns]')
# google_trends.key.astype('datetime64[ns]')
# data = data1.merge(google_trends, on='Timestamp', how='left')
# Last modifications
# data.drop('Unnamed: 0', axis=1,inplace = True)
print(data1.to_string(header=True))
data1.to_csv('{0}_full_data.csv'.format(crypto_name))
print(len(data1))
