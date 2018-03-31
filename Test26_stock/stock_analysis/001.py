'''
1.csv数据的读取

2.利用常用函数获取均值、中位数、方差、标准差等统计量

3.利用常用函数分析价格的加权均值、收益率、年化波动率等常用指标

4.处理数据中的日期

虽然PD非常方便，可以只用numpy来实现
'''
import numpy as np

c, v = np.loadtxt('../SDHJ_stock_data1.csv', delimiter=',', usecols=(3, 5), unpack=True)

print(c)

print(v)

vwap = np.average(c, weights=v)

print(vwap)

print(np.max(c))

print(np.min(c))

print(np.ptp(c))  #最大值和最小值差值

print(np.median(c))  #中间值

print(np.var(c)) #方差，反应股票的波动

print(np.mean((c - c.mean())**2))  #方差计算方法
#每天收益率

returns = -np.diff(c) / c[1:]

print(returns)
print(np.std(returns))  #收益标准差

print(np.where(returns>0)) #收益为正的天数

#波动率
logreturns = -np.diff(np.log(c))

volatility = np.std(logreturns) / np.mean(logreturns)

annual_volatility = volatility / np.sqrt(1. / 252.)

print(volatility)

print(annual_volatility)