import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

#Bringing in sentiment data
df = pd.read_csv('pfizer_dated_sentiments_prices.csv')
print(df.head())


#Bringing in stock data
historical_pfe = yf.download('PFE')
print(historical_pfe.head())

#Aligning indices before merging
historical_pfe.reset_index(inplace=True)
historical_pfe = historical_pfe.rename(columns = {'Date':'date'})
historical_pfe['date']=historical_pfe['date'].astype(str)

df['date']=df['date'].astype(str)
df.set_index('date')

#Merging
merged = pd.merge(df, historical_pfe)
print(merged)

#Examining correlations
print(merged.corr(method='pearson'))

#Rescaling prior to plotting
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

merged_ints = merged.iloc[:, 1:]
ints_scaled = pd.DataFrame(sc.fit_transform(merged_ints))
print(ints_scaled.corr(method='pearson'))


#Plotting data
date = merged['date']
closing_price = ints_scaled[6]
pos_score = ints_scaled[1]
neg_score = ints_scaled[2]
compound = ints_scaled[0]

plt.plot(date, closing_price, label="closing price", linestyle="-")
plt.plot(date, neg_score, label="negative sentiments", color='red', linestyle='--')
plt.plot(date, pos_score, label="positive sentiments", color='cyan', linestyle='--')
plt.plot(date, compound, label="compound score", color='green', linestyle="--")
plt.legend()
plt.show()
