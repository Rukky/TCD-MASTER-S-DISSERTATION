import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tools import add_constant
from arch.unitroot import ADF
from arch.unitroot import ZivotAndrews
from arch.unitroot import PhillipsPerron
from arch.unitroot import KPSS
import statsmodels.api as sm
import warnings
from scipy.stats import pearsonr
warnings.filterwarnings('ignore')
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_error
from stargazer.stargazer import Stargazer
from IPython.core.display import HTML


# evaluate an ridge regression model on the dataset
from numpy import mean
from numpy import std
from numpy import absolute
from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Ridge
%matplotlib inline

df_bnb = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/BNB/log.csv')
df_btc = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/BTC/log.csv')
df_doge = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/DOGE/log.csv')
df_ada = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/ADA/log.csv')
df_eos = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/EOS/log.csv')
df_eth = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/ETH/log.csv')
df_ltc = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/LTC/log.csv')
df_miota = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/IOTA/log.csv')
df_neo = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/NEO/log.csv')
df_vet = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/VET/log.csv')
df_xlm= pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/XLM/log.csv')
df_xmr = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/XMR/log.csv')
df_xrp = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/XRP/log.csv')

bnbs = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/BNB/BNBVADER.csv')
btcs = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/BTC/BTCVADER.csv')
doges = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/DOGE/DOGEVADER.csv')
adas = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/ADA/ADAVADER.csv')
eoss = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/EOS/EOSVADER.csv')
eths = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/ETH/ETHVADER.csv')
ltcs = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/LTC/LTCVADER.csv')
miotas = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/IOTA/IOTAVADER.csv')
neos = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/NEO/NEOVADER.csv')
vets = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/VET/VETVADER.csv')
xlms= pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/XLM/XLMVADER.csv')
xmrs = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/XMR/XMRVADER.csv')
xrps = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/XRP/XRPVADER.csv')

bnb = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/BNB/BNBVol.csv')
btc = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/BTC/BTCVol.csv')
doge = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/DOGE/DOGEVol.csv')
ada = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/ADA/ADAVol.csv')
eos = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/EOS/EOSVol.csv')
eth = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/ETH/ETHVol.csv')
ltc = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/LTC/LTCVol.csv')
miota = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/IOTA/IOTAVol.csv')
neo = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/NEO/NEOVol.csv')
vet = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/VET/VETVol.csv')
xlm= pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/XLM/XLMVol.csv')
xmr = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/XMR/XMRVol.csv')
xrp = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/XRP/XRPVol.csv')



ada = ada['Count']
bnb = bnb['Count']
btc = btc['Count']
doge = doge['Count']
eos = eos['Count']
eth = eth['Count']
ltc = ltc['Count']
miota = miota['Count']
neo = neo['Count']
vet = vet['Count']
xlm = xlm['Count']
xmr = xmr['Count']
xrp = xrp['Count']


def fillna (data) :
     data.interpolate(method= 'linear', limit_direction = 'forward')
     data= data

df_ada = df_ada['ADA'].interpolate(method= 'linear', limit_direction = 'forward')
df_bnb = df_bnb['BNB'].interpolate(method= 'linear', limit_direction = 'forward')
df_btc = df_btc['BTC'].interpolate(method= 'linear', limit_direction = 'forward')
df_doge = df_doge['DOGE'].interpolate(method= 'linear', limit_direction = 'forward')
df_eos = df_eos['EOS'].interpolate(method= 'linear', limit_direction = 'forward')
df_eth = df_eth['ETH'].interpolate(method= 'linear', limit_direction = 'forward')
df_miota = df_miota['IOTA'].interpolate(method= 'linear', limit_direction = 'forward')
df_ltc = df_ltc['LTC'].interpolate(method= 'linear', limit_direction = 'forward')
df_neo = df_neo['NEO'].interpolate(method= 'linear', limit_direction = 'forward')
df_vet = df_vet['VET'].interpolate(method= 'linear', limit_direction = 'forward')
df_xlm = df_xlm['XLM'].interpolate(method= 'linear', limit_direction = 'forward')
df_xmr = df_xmr['XMR'].interpolate(method= 'linear', limit_direction = 'forward')
df_xrp = df_xrp['XRP'].interpolate(method= 'linear', limit_direction = 'forward')




fillna(df_ada)
fillna(df_bnb)
fillna(df_btc)
fillna(df_doge)
fillna(df_eos)
fillna(df_eth)
fillna(df_miota)
fillna(df_ltc)
fillna(df_neo)
fillna(df_vet)
fillna(df_xlm)
fillna(df_xmr)
fillna(df_xrp)


adas = adas['compound']
bnbs = bnbs['compound']
btcs = btcs['compound']
doges = doges['compound']
eoss = eoss['compound']
eths = eths['compound']
miotas = miotas['compound']
ltcs = ltcs ['compound']
neos = neos ['compound']
vets = vets['compound']
xlms = xlms['compound']
xmrs = xmrs['compound']
xrps = xrps['compound']

df_vader = pd.DataFrame({
          'ADA':adas,
          'BNB': bnbs,
          'BTC': btcs,
          'DOGE': doges,
          'EOS':eoss,
          'ETH':eths,
          'IOTA':miotas,
          'LTC':ltcs,
          'NEO':neos,
          'VET':vets,
          'XLM':xlms,
          'XMR':xmrs,
          'XRP':xrps,
          


})




#acf = pd.DataFrame(sm.tsa.stattools.acf(df_doge), columns=["ACF"])
#fig = acf[1:].plot(kind="bar", title="Autocorrelations of ADA log price returns")


adf = ADF(xrp[1:608])
adf.trend="ctt"
adf.lags = 0
print(adf.summary().as_text())

pp = PhillipsPerron(xrp[1:608])
pp.trend = "ct"
pp.test_type = "rho"
print(pp.summary().as_text())


pp = PhillipsPerron(xrp[1:608])
pp.trend = "ct"
pp.test_type = "tau"
print(pp.summary().as_text())


kpss = KPSS(xrp[1:608])
print(kpss.summary().as_text())

x = np.array(xrp[1:608], dtype = np.float)
za = ZivotAndrews(x)
za.trend ="ct"
print(za.summary().as_text())

#Ljung box test
res = sm.tsa.ARMA(xrp[1:608], (1,1)).fit(disp=-1)

print(sm.stats.acorr_ljungbox(res.resid, lags=[1], return_df=True))

