import pandas as pd
import matplotlib.pyplot as plt


bnbs = pd.read_csv('/Users/rukky/Documents/TCD Masters Dissertation/XRP/XRP.csv')

bnbs.drop(columns=['Permalink', 'Post ID', 'Url', 'Flair', 'Author', 'Title' ])
bnbs['Publish Date'] = pd.to_datetime(bnbs['Publish Date'])
cnt = bnbs.groupby('Publish Date').size().values
bnbs= bnbs.drop_duplicates(subset='Publish Date').assign(Count=cnt)
bnbs = bnbs.set_index('Publish Date')
bnbs= bnbs['Count'].resample('D').sum()
bnbs.to_csv('XRP/XRPVol.csv')


