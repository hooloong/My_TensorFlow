
import matplotlib.pyplot as plt
import pandas as pd
import requests
import io
import numpy as np

# url = 'http://blog.topspeedsnail.com/wp-content/uploads/2016/12/铁路客运量.csv'
# ass_data = requests.get(url).content
#
# df = pd.read_csv(io.StringIO(ass_data.decode('utf-8')))  # python2使用StringIO.StringIO
df = pd.read_csv('./railway.csv',encoding='utf-8')

data = np.array(df['铁路客运量_当期值(万人)'])
# normalize
normalized_data = (data - np.mean(data)) / np.std(data)
print(normalized_data)
plt.figure()
plt.plot(data)
plt.show()