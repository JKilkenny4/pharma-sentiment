# pip install twint
# thanks to this article https://betterprogramming.pub/how-to-scrape-tweets-with-snscrape-90124ed006af


import numpy as np
import pandas as pd
import twint


#Using twint to pull a list of tweets
'''
c = twint.Config()
c.Store_csv = True
c.Search = 'pfizer'
c.Since = "2020-02-01 00:00:00"
#c.Until = "2021-07-31 00:00:00"
c.Lang = 'en'
c.Links = 'Include'
c.Stats = True
c.Output = '/Users/jackkilkenny/Documents/Projects/pharma-sentiment/trial_run_pfizer2.csv'

twint.run.Search(c)'''

df = pd.read_csv('pfizer_feb20_aug21.csv')


print(len(df))
print(df.columns)
print(min(df['date']))
print(max(df['date']))

