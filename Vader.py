
import seaborn as sns
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import glob, os  
import ssl  


#https://github.com/ekapope/Combine-CSV-files-in-the-folder/blob/master/Combine_CSVs.py

df_bnb = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/BNB/BNB.csv')
df_btc = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/BTC/BTC.csv')
df_doge = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/DOGE/DOGE.csv')
df_ada = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/ADA/ADA.csv')
df_eos = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/EOS/EOS.csv')
df_ETH = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/ETH/ETH.csv')
df_ltc = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/LTC/LTC.csv')
df_miota = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/IOTA/IOTA.csv')
df_neo = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/NEO/NEO.csv')
df_vet = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/VET/VET.csv')
df_xlm= pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/XLM/XLM.csv')
df_xmr = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/XMR/XMR.csv')
df_xrp = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/XRP/XRP.csv')

analyzer = SentimentIntensityAnalyzer()


def cleaning (data): 
    data.drop(columns=['Permalink', 'Post ID', 'Url', 'Flair', 'Author', ], inplace=True)

    data.dropna(inplace=True)

    blanks = []  # start with an empty list

    for i in data:  
     if type(i)==str:            
        if i.isspace():        
            blanks.append(i)     

    data['Title'].drop(blanks, inplace=True)

cleaning(df_doge)
cleaning(df_ada)
cleaning(df_xmr)
cleaning(df_vet)
cleaning(df_miota)
cleaning(df_neo)
cleaning(df_eos)
cleaning(df_ETH)
cleaning(df_ltc)
cleaning(df_xrp)
cleaning(df_bnb)
cleaning(df_btc)

#ADA
df_ada['Publish Date'] = pd.to_datetime(df_ada['Publish Date'])
ADAs = df_ada['Title'].apply(lambda x: analyzer.polarity_scores(x))
df_ada= pd.concat([df_ada,ADAs.apply(pd.Series)],1)
df_ada[["neg", "neu", "pos", "compound"]].describe()
df_ada = df_ada.set_index('Publish Date') 
ADA = df_ada.resample('D').mean().fillna(0)


#BNB
df_bnb['Publish Date'] = pd.to_datetime(df_bnb['Publish Date'])
BNBs = df_bnb['Title'].apply(lambda x: analyzer.polarity_scores(x))
df_bnb= pd.concat([df_bnb,BNBs.apply(pd.Series)],1)
df_bnb[["neg", "neu", "pos", "compound"]].describe()
df_bnb = df_bnb.set_index('Publish Date') 
BNB = df_bnb.resample('D').mean().fillna(0)


#BTC
df_btc['Publish Date'] = pd.to_datetime(df_btc['Publish Date'])
BTCs = df_btc['Title'].apply(lambda x: analyzer.polarity_scores(x))
df_btc= pd.concat([df_btc,BTCs.apply(pd.Series)],1)
df_btc[["neg", "neu", "pos", "compound"]].describe()
df_btc = df_btc.set_index('Publish Date') 
BTC = df_btc.resample('D').mean().fillna(0)



#DOGE
df_doge['Publish Date'] = pd.to_datetime(df_doge['Publish Date'])
DOGEs = df_doge['Title'].apply(lambda x: analyzer.polarity_scores(x))
df_doge = pd.concat([df_doge,DOGEs.apply(pd.Series)],1)
df_doge[["neg", "neu", "pos", "compound"]].describe()
df_doge = df_doge.set_index('Publish Date') 
DOGE = df_doge.resample('D').mean().fillna(0)

#EOS
df_eos['Publish Date'] = pd.to_datetime(df_eos['Publish Date'])
EOSs = df_eos['Title'].apply(lambda x: analyzer.polarity_scores(x))
df_eos= pd.concat([df_eos,EOSs.apply(pd.Series)],1)
df_eos[["neg", "neu", "pos", "compound"]].describe()
df_eos = df_eos.set_index('Publish Date') 
EOS = df_eos.resample('D').mean().fillna(0)

#ETH
df_ETH['Publish Date'] = pd.to_datetime(df_ETH['Publish Date'])
ETHs = df_ETH['Title'].apply(lambda x: analyzer.polarity_scores(x))
df_ETH= pd.concat([df_ETH,ETHs.apply(pd.Series)],1)
df_ETH[["neg", "neu", "pos", "compound"]].describe()
df_ETH = df_ETH.set_index('Publish Date') 
ETH = df_ETH.resample('D').mean().fillna(0)

#LTC
df_ltc['Publish Date'] = pd.to_datetime(df_ltc['Publish Date'])
LTCs = df_ltc['Title'].apply(lambda x: analyzer.polarity_scores(x))
df_ltc= pd.concat([df_ltc,LTCs.apply(pd.Series)],1)
df_ltc[["neg", "neu", "pos", "compound"]].describe()
df_ltc = df_ltc.set_index('Publish Date') 
LTC = df_ltc.resample('D').mean().fillna(0)

#MIOTA
df_miota['Publish Date'] = pd.to_datetime(df_miota['Publish Date'])
MIOTAs = df_miota['Title'].apply(lambda x: analyzer.polarity_scores(x))
df_miota= pd.concat([df_miota,MIOTAs.apply(pd.Series)],1)
df_miota[["neg", "neu", "pos", "compound"]].describe()
df_miota = df_miota.set_index('Publish Date') 
MIOTA = df_miota.resample('D').mean().fillna(0)

#NEO
df_neo['Publish Date'] = pd.to_datetime(df_neo['Publish Date'])
NEOs = df_neo['Title'].apply(lambda x: analyzer.polarity_scores(x))
df_neo= pd.concat([df_neo,NEOs.apply(pd.Series)],1)
df_neo[["neg", "neu", "pos", "compound"]].describe()
df_neo = df_neo.set_index('Publish Date') 
NEO = df_neo.resample('D').mean().fillna(0)

#VET
df_vet['Publish Date'] = pd.to_datetime(df_vet['Publish Date'])
VETs = df_vet['Title'].apply(lambda x: analyzer.polarity_scores(x))
df_vet= pd.concat([df_vet,VETs.apply(pd.Series)],1)
df_vet[["neg", "neu", "pos", "compound"]].describe()
df_vet= df_vet.set_index('Publish Date') 
VET= df_vet.resample('D').mean().fillna(0)

#XLM
df_xlm['Publish Date'] = pd.to_datetime(df_xlm['Publish Date'])
XLMs = df_xlm['Title'].apply(lambda x: analyzer.polarity_scores(x))
df_xlm= pd.concat([df_xlm,XLMs.apply(pd.Series)],1)
df_xlm[["neg", "neu", "pos", "compound"]].describe()
df_xlm= df_xlm.set_index('Publish Date') 
XLM= df_xlm.resample('D').mean().fillna(0)

#XMR
df_xmr['Publish Date'] = pd.to_datetime(df_xmr['Publish Date'])
XMRs = df_xmr['Title'].apply(lambda x: analyzer.polarity_scores(x))
df_xmr= pd.concat([df_xmr,XMRs.apply(pd.Series)],1)
df_xmr[["neg", "neu", "pos", "compound"]].describe()
df_xmr= df_xmr.set_index('Publish Date') 
XMR= df_xmr.resample('D').mean().fillna(0)

#XRP
df_xrp['Publish Date'] = pd.to_datetime(df_xrp['Publish Date'])
XRPs = df_xrp['Title'].apply(lambda x: analyzer.polarity_scores(x))
df_xrp= pd.concat([df_xrp,XRPs.apply(pd.Series)],1)
df_xrp[["neg", "neu", "pos", "compound"]].describe()
df_xrp= df_xrp.set_index('Publish Date') 
XRP= df_xrp.resample('D').mean().fillna(0)


#find all csv files in the folder
#use glob pattern matching -> extension = 'csv'
#save result in list -> all_filenames
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
#print(all_filenames)

#combine all files in the list
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
#export to csv
combined_csv.to_csv( "combined_csv.csv", index=False, encoding='utf-8-sig')





#df['mean'] = df['compound'].expanding().mean()
#df[["neg", "neu", "pos", "compound", "mean"]].describe()
#df['Date']=pd.to_datetime(df.Date).dt.date