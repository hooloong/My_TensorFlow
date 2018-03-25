import tushare as ts
import pandas as pd

# d = ts.get_tick_data('601318', date='2017-03-23')
# print(d)
# 山东黄金
e = ts.get_hist_data('600547', start='2012-01-1', end='2018-03-23')
print(e)
sdhj_df = pd.DataFrame(e)
# print(sdhj_df)
# sdhj_df.to_csv("SDHJ_stock_data1.csv",index=False)
sdhj_df = sdhj_df.iloc[::-1]
print(sdhj_df)
sdhj_df.to_csv("SDHJ_stock_data.csv")

