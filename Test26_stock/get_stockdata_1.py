from yahoo_finance import Share
import pandas as pd

share = Share("IBM")
print(share)
stock_history = share.get_historical("2000-01-01", "2018-01-01")
ibm_df = pd.DataFrame(stock_history)
ibm_df = ibm_df.iloc[::-1]

ibm_df.to_csv("SIGM_stock_data.csv", index=False)