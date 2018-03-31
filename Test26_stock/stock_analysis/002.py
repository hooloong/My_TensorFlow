import numpy as np
import datetime


def datestr2num(bytedate):
    return datetime.datetime.strptime(

        bytedate.decode('utf-8'), '%Y-%m-%d').date().weekday()


dates, c = np.loadtxt('../SDHJ_stock_data1.csv', delimiter=',', usecols=(0, 3),

                      converters={0: datestr2num}, unpack=True)

averages = np.zeros(5)

for i in range(5):
    index = np.where(dates == i)

    prices = np.take(c, index)

    avg = np.mean(prices)

    averages[i] = avg
    print("Day {} prices: {},avg={}".format(i, prices, avg))

top = np.max(averages)

top_index = np.argmax(averages)

bot = np.min(averages)

bot_index = np.argmin(averages)

print('highest:{}, top day is {}'.format(top, top_index))

print('lowest:{},bottom day is {}'.format(bot, bot_index))

