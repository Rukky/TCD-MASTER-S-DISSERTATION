import pandas as pd  # To read data
import seaborn as sns
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import glob, os    

#df_bnb = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/BNB/BNB.csv', index_col='Date')
#df_btc = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/BTC/BTC.csv', index_col='Date')
#df_doge = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/DOGE/DOGE.csv', index_col='Date')
df_ada = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/ADA/ADA.csv')
#df_eos = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/EOS/EOS.csv', index_col='Date')
#df_eth = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/ETH/ETH.csv', index_col='Date')
#df_ltc = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/LTC/LTC.csv', index_col='Date')
df_miota = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/MIOTA/MIOTA.csv', index_col='Date')
#df_neo = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/NEO/NEO.csv', index_col='Date')
#df_vet = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/VET/VET.csv', index_col='Date')
#df_xlm= pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/XLM/XLM.csv', index_col='Date')
#df_xmr = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/XMR/XMR.csv', index_col='Date')
#df_xrp = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/XRP/XRP.csv', index_col='Date')


analyzer = SentimentIntensityAnalyzer()

df_ada.drop(columns=[ 'Total No. of Comments','Permalink', 'Post ID', 'Score'], inplace=True)
df_ada['Publish Date'] = pd.to_datetime(df_ada['Publish Date'])
ADAs = df_ada['Title'].apply(lambda x: analyzer.polarity_scores(x))
df_ada= pd.concat([df_ada,ADAs.apply(pd.Series)],1)
df_ada[["neg", "neu", "pos", "compound"]].describe()
df_ada = df_ada.set_index('Publish Date') 
ADA = df_ada.resample('D').mean().fillna(0)
ADA.to_csv('ADAVADER.csv')

