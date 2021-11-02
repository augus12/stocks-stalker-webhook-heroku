import asyncio

import pandas as pd
import yfinance
import mplfinance as mpf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from aiogram.types import ParseMode

matplotlib.use('Agg')
import logging


swing_long = []
swing_long_risky = []
swing_long_exit = []


def getLongSignals(long, long_risk, long_exit_small, long_exit, price1, price2, factor1=1.0, factor2=1.0):
    long_signal = []        #1
    long_risk_signal = []   #2
    long_exit_small_signal = [] #3
    long_exit_signal = []    #4
    prev = 0
    for date,value in long.iteritems():
        if  long[date] and prev != 1:
            long_signal.append(price1[date] * factor1)
            long_risk_signal.append(np.nan)
            long_exit_small_signal.append(np.nan)
            long_exit_signal.append(np.nan)
            prev = 1
        elif long_risk[date] and prev != 2:
            long_risk_signal.append(price1[date] * factor1)
            long_signal.append(np.nan)
            long_exit_small_signal.append(np.nan)
            long_exit_signal.append(np.nan)
            prev = 2
        elif long_exit[date] and prev != 4:
            long_exit_signal.append(price2[date] * factor2)
            long_signal.append(np.nan)
            long_risk_signal.append(np.nan)
            long_exit_small_signal.append(np.nan)
            prev = 4
        elif long_exit_small[date] and prev != 3 and prev != 4:
            long_exit_small_signal.append(price2[date] * factor2)
            long_signal.append(np.nan)
            long_risk_signal.append(np.nan)
            long_exit_signal.append(np.nan)
            prev = 3
        else:
            long_signal.append(np.nan)
            long_risk_signal.append(np.nan)
            long_exit_small_signal.append(np.nan)
            long_exit_signal.append(np.nan)

    return long_signal, long_risk_signal, long_exit_small_signal, long_exit_signal


async def plot_chart(bot, chat_id, stock):
    try:
        plt.rc('font', size=14)
        df = await HA(stock, period='1y')
        strategy = await trades(df)
        df['Date'] = range(df.shape[0])
        df = df.loc[:, ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'near']]
        strategy['Date'] = range(strategy.shape[0])
        df['long'] = strategy['long']
        df['long_risky'] = strategy['long_risky']
        df['long_exit'] = strategy['long_exit']
        df['long_exit_soft'] = strategy['long_exit_soft']

        l_arr, lr_arr, les_arr, le_arr = getLongSignals(df.iloc[-110:]['long'], df.iloc[-110:]['long_risky'], df.iloc[-110:]['long_exit_soft'], df.iloc[-110:]['long_exit'],
                       df.iloc[-110:]['Low'], df.iloc[-110:]['High'], 0.98, 1.01)
        go_long = mpf.make_addplot(l_arr, type='scatter', markersize=10, marker='^', color = 'green')
        exit_long = mpf.make_addplot(le_arr, type='scatter', markersize=10, marker='v', color='red')
        exit_long_soft = mpf.make_addplot(les_arr, type='scatter', markersize=10, marker='v', color='gray')
        risky_long = mpf.make_addplot(lr_arr, type='scatter', markersize=10, marker='^', color='blue')

        # setup = dict(type='candle', volume=True, mav=(7, 15, 22))
        # mpf.plot(df.iloc[0: 40], **setup)
        mpf.plot(df.iloc[-110:], type='candle', style='charles', mav=(50, 100, 200), volume=True, addplot=[go_long, exit_long, exit_long_soft, risky_long])

        output = BytesIO()
        plt.gcf().savefig(output, format="png")
        image_as_string = output.getvalue()
        logging.info("Sending ha plot..")
        await bot.send_photo(chat_id=chat_id, photo=image_as_string, caption="#" + stock)
    except:
        logging.warning("error in plot_chart")
    # candlestick_ohlc(ax,df.values,width=0.6, colorup='green', colordown='red', alpha=0.8)
    # fig.show()


# for i in range(df_ha.shape[0]):
#     if i > 0:
#         df_ha.loc[df_ha.index[i], 'Open'] = (df['Open'][i - 1] + df['Close'][i - 1]) / 2
#
#     df_ha.loc[df_ha.index[i], 'Close'] = (df['Open'][i] + df['Close'][i] + df['Low'][i] + df['High'][i]) / 4
# df_ha = df_ha.iloc[1:, :]
#
# plot_chart(df_ha)

async def HA(stock, period):
    name = stock + '.NS'
    ticker = yfinance.Ticker(name)
    df = ticker.history(interval="1d", period=period)
    df['Act_Close'] = df['Close']
    df["SMA_100"] = df['Close'].rolling(window=100).mean()
    df["SMA_50"] = df['Close'].rolling(window=50).mean()
    df['HA_Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4

    idx = df.index.name
    df.reset_index(inplace=True)

    for i in range(0, len(df)):
        if i == 0:
            df.loc[df.index[i], 'HA_Open'] = (df.loc[df.index[i], 'Open'] + df.loc[df.index[i], 'Close']) / 2
        else:
            df.loc[df.index[i], 'HA_Open'] = (df.loc[df.index[i - 1], 'HA_Open'] + df.loc[df.index[i - 1], 'HA_Close']) / 2
        df.loc[df.index[i], 'MA_Diff'] = df.loc[df.index[i], 'Close'] - df.loc[df.index[i], 'SMA_50']
        df.loc[df.index[i], 'Percent'] = df.loc[df.index[i], 'MA_Diff'] / df.loc[df.index[i], 'Close']
        df.loc[df.index[i], 'near'] = df.loc[df.index[i], 'Percent'] <= 0.05

    if idx:
        df.set_index(idx, inplace=True)

    df['High'] = df[['HA_Open', 'HA_Close', 'High']].max(axis=1)
    df['Low'] = df[['HA_Open', 'HA_Close', 'Low']].min(axis=1)

    df['Open'] = df['HA_Open']
    df['Close'] = df['HA_Close']

    # plot_chart(df)
    return df


# def trades(HA, oldO, oldC):
#    Heikin Ashi Reversal Strategy
#    ------------- Entry ----------------
#    Buying
#    latest HA candle is bearish, HA_Close < HA_Open
#     long1 = HA[0,:] < HA[1,:]
# #    current candle body is longer than previous candle body
#     long2 = numpy.abs(HA[0,:] - HA[1,:]) > numpy.abs(oldC - oldO)
# #    previous candle was bearish
#     long3 = oldC < oldO
# #    latest candle has no upper wick HA_Open == HA_High
#     long4 = HA[1,:] == HA[2,:]
#     long = long1 & long2 & long3 & long4

async def trades(df):
    current = df[1:]
    previous = df.shift(1)[1:]
    previous_prev = df.shift(2)[1:]

    current_close_to_ma = df['near']
    current_vol_increase = previous['Volume'] < current['Volume']
    previous_prev_bearish = previous_prev['Close'] < previous_prev['Open']

    latest_bearish = current['Close'] < current['Open']
    previous_bearish = previous['Close'] < previous['Open']

    previous_reversal = (previous['Open'] < previous['High']) & (previous['Low'] < previous['Open'])
    current_reversal = (current['Open'] < current['High']) & (current['Low'] < current['Open'])

    current_candle_longer = (np.abs(current['Close'] - current['Open']) > np.abs(previous['Close'] - previous['Open']))

    current_candle_shorter = ( np.abs(current['Close'] - current['Open']) < np.abs(previous['Close'] - previous['Open']))

    # print(np.divide(np.abs(current['Close'] - previous['Close']), previous['Close'])[-1])
    current_candle_size = np.divide(np.abs(current['Close'] - previous['Close']), previous['Close']) <= 0.05
    current_open_eq_low = current['Open'] == current['Low']
    current_open_eq_high = current['Open'] >= 0.99 * current['High']
    # current_open_eq_high_soft = current['Open'] >= 0.985 * current['High']

    current_width = np.divide(abs(current['Open'] - current['Close']), current['Open']) >= 0.01
    long = (
            ~latest_bearish
            & current_close_to_ma
            # & current_vol_increase
            # & current_candle_size
            & current_candle_longer
            # & (((previous_reversal & previous_bearish) | (
            #     previous_prev_bearish & previous_reversal)) | previous_bearish)
            & current_open_eq_low)

    long_risky = (
            ~latest_bearish
            & ~current_close_to_ma
            # & current_vol_increase
            # & current_candle_size
            & current_candle_longer
            # & (((previous_reversal & previous_bearish) | (
            #     previous_prev_bearish & previous_reversal)) | previous_bearish)
            & current_open_eq_low)

    long_exit = (
            latest_bearish
            #  & ~previous_bearish
            & (current_open_eq_high & current_width)

    )
    long_exit_soft = (
            latest_bearish
            #      & ~previous_bearish
            & ~(current_open_eq_high & current_width)
            & (current_open_eq_high | current_width)
    )
    return pd.DataFrame(
        {'long': long,
         'long_risky': long_risky,
         'long_exit_soft': long_exit_soft,
         #  'short': short,
         'long_exit': long_exit
         #  'short_exit': short_exit
         },
        index=current.index)


async def find_swing(bot, chat_id, stocks, time=0, all=0, client_loop=None):

    global swing_long
    global swing_long_risky
    global swing_long_exit

    if all != 2:
        swing_long = []
        swing_long_risky = []
        swing_long_exit = []

    result = []
    result_risky_buy = []
    result_exit = []
    date = ""
    for stock in stocks:
        try:
            df_ha = await HA(stock, '3mo')

            date = str(df_ha.index[-1 + time]).split(" ")[0]
            prev = (time - 1) if time != 0 else -2
            up = u'\u2B06'
            down = u'\u2B07'

            strategy = await trades(df_ha)

            df_ha['long'] = strategy['long']
            df_ha['long_risky'] = strategy['long_risky']
            df_ha['long_exit'] = strategy['long_exit']
            df_ha['long_exit_soft'] = strategy['long_exit_soft']
            index = -1 + time
            l_arr, lr_arr, les_arr, le_arr = getLongSignals(df_ha['long'],
                                                            df_ha['long_risky'],
                                                            df_ha['long_exit_soft'],
                                                            df_ha['long_exit'],
                                                            df_ha['Low'], df_ha['High'], 0.98, 1.01)
            runner = u'\U0001F3C3'
            if l_arr[index] is not np.nan:
                logging.info("Found HA buy: " + stock)
                curr = await get_exit_index(1, le_arr, l_arr, lr_arr, prev)
                running = 1 if curr == 0 else 0
                if curr == 0:
                    curr = -1
                change = round((df_ha['Act_Close'][curr] - df_ha['Act_Close'][prev]) * 100 / df_ha['Act_Close'][prev], 2)
                change_str = "   " + str(abs(change)) + "% " + (up if change > 0 else down)
                change_str = change_str + (" "+runner if running == 1 else "")
                result.append('*' + stock + '*' + change_str)
            if le_arr[index] is not np.nan:
                logging.info("Found HA exit: " + stock)
                # curr = await get_exit_index(0, le_arr, l_arr, lr_arr, prev)
                # running = 1 if curr == 0 else 0
                # if curr == 0:
                #     curr = -1
                # change = round((df_ha['Act_Close'][curr] - df_ha['Act_Close'][prev]) * 100 / df_ha['Act_Close'][prev], 2)
                # change_str = "   " + str(abs(change)) + "% " + (up if change > 0 else down)
                # change_str = change_str + (" "+runner if running == 1 else "")
                # result_exit.append('*' + stock + '*' + change_str)
                result_exit.append('*' + stock + '*')
            if lr_arr[index] is not np.nan:
                logging.info("Found HA risky buy: " + stock)
                curr = await get_exit_index(1, le_arr, l_arr, lr_arr, prev)
                running = 1 if curr == 0 else 0
                if curr == 0:
                    curr = -1
                change = round((df_ha['Act_Close'][curr] - df_ha['Act_Close'][prev]) * 100 / df_ha['Act_Close'][prev], 2)
                change_str = "   " + str(abs(change)) + "% " + (up if change > 0 else down)
                change_str = change_str + (" "+runner if running == 1 else "")
                result_risky_buy.append('*' + stock + '*' + change_str)
            # await plot_chart(df_ha, chat_id)


        except:
            logging.warning("error in " + stock)

    if all != -1:
        swing_long.extend(result)
        swing_long_risky.extend(result_risky_buy)
        swing_long_exit.extend(result_exit)

        swing_long = list(set(swing_long))
        swing_long_risky = list(set(swing_long_risky))
        swing_long_exit = list(set(swing_long_exit))

    if all != 1 and all != -1:
        await swing_list(bot, chat_id, all, swing_long, swing_long_risky, swing_long_exit, date, client_loop)
    elif all == -1:
        await swing_list(bot, chat_id, all, result, result_risky_buy, result_exit, date, client_loop)


async def get_exit_index(buy_or_sell, le_arr, l_arr, lr_arr, prev):
    for i in range(prev, -1):
        if buy_or_sell == 1 and (le_arr[i] is not np.nan):
            return i
        if buy_or_sell == 0 and ((l_arr[i] is not np.nan) or (lr_arr[i] is not np.nan)):
            return i
    return 0


async def some_callback(bot, chat_id, text):
    await bot.send_message(chat_id, text, parse_mode=ParseMode.MARKDOWN)


async def between_callback(bot, chat_id, text):
    loop = asyncio.new_event_loop()
    task = loop.create_task(some_callback(bot, chat_id, text))
    await asyncio.gather(task)
    loop.close()


async def swing_list(bot, chat_id, all, swing_long, swing_long_risky, swing_long_exit, date, client_loop):
    try:
        if len(swing_long) > 0 or len(swing_long_exit) > 0 or len(swing_long_risky) > 0:
            message = "*Swing watchlist  #" + date + "*"
            if len(swing_long) > 0:
                message = message + "\n\n*Buy:*\n"
                message = message + '\n'.join(swing_long)
            if len(swing_long_risky) > 0:
                message = message + "\n\n*Risky Buy:*\n"
                message = message + '\n'.join(swing_long_risky)
            if len(swing_long_exit) > 0:
                message = message + "\n\n*Exit:*\n"
                message = message + '\n'.join(swing_long_exit)

            send_fut = asyncio.create_task(some_callback(bot, chat_id, message))
            # wait for the coroutine to finish
            await send_fut.result()
            # await between_callback(bot, chat_id, message)
            # thread = Thread(target=between_callback, kwargs={'bot': bot, 'chat_id': chat_id, 'text': message})
            # thread.start()
            # await bot.send_message(chat_id, text=message, parse_mode=ParseMode.MARKDOWN)
        else:
            if all != 1:
                await bot.send_message(chat_id, text='Swing watchlist, none found.')
    except Exception as e:
        logging.warning(str(e))