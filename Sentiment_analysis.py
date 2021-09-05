#Getting set up
import pandas as pd
import matplotlib.pyplot as plt
import nltk
nltk.download('vader_lexicon')

df = pd.read_csv('pfizer_feb20_aug21.csv')

#Applying NLTK's VADER model to code each tweet
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

#Output is a dictionary that we are appending as 'scores'
df['scores'] = df['tweet'].apply(lambda tweet: sia.polarity_scores(tweet))

#Compound score is cool because it integrates both the valence and intensity of each sentiment in a given tweet
#Positive values = postive sentiments, negative = negative - size of value indicates intensity of sentiment

#Pulling scores into their own columns
df['compound']  = df['scores'].apply(lambda score_dict: score_dict['compound'])
df['pos_score']  = df['scores'].apply(lambda score_dict: score_dict['pos'])
df['neg_score']  = df['scores'].apply(lambda score_dict: score_dict['neg'])

#Descriptives for score columns
for column in df[['compound', 'pos_score', 'neg_score']]:
    mean_column = df[column].mean()
    std_column = df[column].std()
    max_column = max(df[column])
    min_column = min(df[column])
    print(column, 'MEAN = ', mean_column, '; ', 'STANDARD DEV = ', std_column, '; ',
        'MAX = ', max_column, '; ', 'MIN = ', min_column)


df.to_csv('pfizer_coded.csv')

#Transforming data into dates w/ means
date_df = pd.DataFrame(df.groupby(['date'])['compound', 'pos_score', 'neg_score'].mean())

print(date_df)

#Now need to pull in Pfizer stock data, so saving progress as csv and opening new script
date_df.to_csv('pfizer_dated_sentiments_prices.csv')



