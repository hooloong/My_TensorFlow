import tushare as ts
import pandas as pd
import random
from matplotlib import pyplot

sdhj=pd.read_csv('SDHJ_stock_data.csv')
print(sdhj)
sdhj_df = pd.DataFrame(sdhj,columns=["open","close","price_change"])
print(sdhj_df)
pyplot.plot(sdhj_df)
pyplot.show()
