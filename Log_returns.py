import pandas as pd 
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns



df_bnb = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/BNB/BNB-USD.csv', index_col='Date')
df_btc = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/BTC/BTC-USD.csv', index_col='Date')
df_doge = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/DOGE/DOGE-USD.csv', index_col='Date')
df_ada = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/ADA/ADA-USD.csv', index_col='Date')
df_eos = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/EOS/EOS-USD.csv', index_col='Date')
df_eth = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/ETH/ETH-USD.csv', index_col='Date')
df_ltc = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/LTC/LTC-USD.csv', index_col='Date')
df_miota = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/IOTA/IOTA-USD.csv', index_col='Date')
df_neo = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/NEO/NEO-USD.csv', index_col='Date')
df_vet = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/VET/VET-USD.csv', index_col='Date')
df_xlm= pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/XLM/XLM-USD.csv', index_col='Date')
df_xmr = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/XMR/XMR-USD.csv', index_col='Date')
df_xrp = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/XRP/XRP-USD.csv', index_col='Date')

bnb = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/BNB/BNBVol.csv' , index_col= 'Publish Date')
btc = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/BTC/BTCVol.csv' , index_col= 'Publish Date')
doge = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/DOGE/DOGEVol.csv' , index_col= 'Publish Date')
ada = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/ADA/ADAVol.csv' , index_col= 'Publish Date')
eos = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/EOS/EOSVol.csv', index_col= 'Publish Date')
eth = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/ETH/ETHVol.csv', index_col= 'Publish Date')
ltc = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/LTC/LTCVol.csv', index_col= 'Publish Date')
miota = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/IOTA/IOTAVol.csv', index_col= 'Publish Date')
neo = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/NEO/NEOVol.csv', index_col= 'Publish Date')
vet = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/VET/VETVol.csv', index_col= 'Publish Date')
xlm= pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/XLM/XLMVol.csv', index_col= 'Publish Date')
xmr = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/XMR/XMRVol.csv', index_col= 'Publish Date')
xrp = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/XRP/XRPVol.csv', index_col= 'Publish Date')



df = pd.DataFrame({

                   
                   'LTC':df_ltc['Close'],
                   'IOTA':df_miota['Close'],
                   'NEO':df_neo['Close'],
                   'VET':df_vet['Close'],
                   'XLM':df_xlm['Close'],
                   'XMR':df_xmr['Close'],
                   'XRP':df_xrp['Close'],
                   
                   })

df_count = pd.DataFrame({

                 'ada' : ada['Count'],
                 'bnb' : bnb['Count'],
                 'btc' : btc['Count'],
                 'doge' :doge['Count'],
                 'eos' : eos['Count'],
                 'eth' : eth['Count'],
                 'ltc' : ltc['Count'],
                 'miota' :miota['Count'],
                 'neo' : neo['Count'],
                 'vet' : vet['Count'],
                 'xlm' : xlm['Count'],
                 'xmr' : xmr['Count'],
                 'xrp' : xrp['Count']

                   
                   })


print (df_count.describe())

df.index = df.index.map(pd.to_datetime)
df = df.sort_index()
print(np.isnan(df))
df_ret = df.pct_change()
df_lag= df_count.shift(1)
df_change = df.apply(lambda x: np.log(x) - np.log(x.shift(1)))
df_change.plot(figsize=(15, 10)).axhline(color='black', linewidth=2)




