import asyncio
import logging
import re
from datetime import datetime, timedelta
from io import BytesIO
from threading import Thread

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import trendln
import yfinance as yf
from aiogram import Bot, types
from aiogram.contrib.middlewares.logging import LoggingMiddleware
from aiogram.dispatcher import Dispatcher
from aiogram.utils.executor import start_webhook
from sklearn.linear_model import LinearRegression

matplotlib.use('Agg')
from bot.settings import (BOT_TOKEN, WEBHOOK_URL, WEBHOOK_PATH, WEBAPP_HOST, WEBAPP_PORT)
from bot.plot_graph import plot_chart, find_swing
from time import sleep

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(bot)
dp.middleware.setup(LoggingMiddleware())
recommendations = []

codes = ['HINDUNILVR', 'POLYCAB', 'BRITANNIA', 'NAVINFLUOR', 'WHIRLPOOL', 'VIPIND',
         'AMBER', 'AMRUTANJAN', 'APLAPOLLO', 'AUBANK', 'BATAINDIA', 'BORORENEW', 'CREDITACC',
         'HDFCLIFE', 'ICICIGI', 'KPITTECH', 'LAURUSLABS', 'PIDILITIND', 'POLYMED', 'SASTASUNDR', 'SIRCA',
         'TATACONSUM', 'TATAELXSI', 'TRENT', 'NAUKRI', 'INFOBEAN', 'HAPPSTMNDS', 'IEX', 'JUBLINGREA']

end_date = datetime.now()

count = 0
count_bse = 0
start_date = end_date - timedelta(days=200)
buy_ma = []
sell_ma = []

# codes = ['NESTLEIND', 'HINDUNILVR', 'POLYCAB', 'DIXON', 'HNDFDS', 'BRITANNIA', 'NAVINFLUOR', 'WHIRLPOOL', 'VIPIND',
#          'AMBER', 'AMRUTANJAN', 'APLAPOLLO', 'ARMANFIN', 'AUBANK', 'BATAINDIA', 'BORORENEW', 'CEATLTD', 'CREDITACC',
#          'HDFCLIFE', 'ICICIGI', 'KPITTECH', 'LAURUSLABS', 'PIDILITIND', 'POLYMED', 'SASTASUNDR', 'SIRCA',
#          'TATACONSUM', 'TATAELXSI', 'TRENT', 'NAUKRI', 'BALKRISIND']

codes_for_ma = ["ACC", "AUBANK", "AARTIIND", "ABBOTINDIA", "ADANIENT", "ADANIGREEN", "ADANIPORTS", "ATGL", "ADANITRANS",
                "ABCAPITAL", "ABFRL", "AJANTPHARM", "APLLTD", "ALKEM", "AMARAJABAT", "AMBUJACEM", "APOLLOHOSP",
                "APOLLOTYRE",
                "ASHOKLEY", "ASIANPAINT", "AUROPHARMA", "DMART", "AXISBANK", "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV",
                "BAJAJHLDNG", "BALKRISIND", "BANDHANBNK", "BANKBARODA", "BANKINDIA", "BATAINDIA", "BERGEPAINT", "BEL",
                "BHARATFORG", "BHEL", "BPCL", "BHARTIARTL", "BIOCON", "BBTC", "BOSCHLTD", "BRITANNIA", "CESC",
                "CADILAHC",
                "CANBK", "CASTROLIND", "CHOLAFIN", "CIPLA", "CUB", "COALINDIA", "COFORGE", "COLPAL", "CONCOR",
                "COROMANDEL",
                "CROMPTON", "CUMMINSIND", "DLF", "DABUR", "DALBHARAT", "DEEPAKNTR", "DHANI", "DIVISLAB", "DIXON",
                "LALPATHLAB",
                "DRREDDY", "EICHERMOT", "EMAMILTD", "ENDURANCE", "ESCORTS", "EXIDEIND", "FEDERALBNK", "FORTIS", "GAIL",
                "GMRINFRA", "GLENMARK", "GODREJAGRO", "GODREJCP", "GODREJIND", "GODREJPROP", "GRASIM", "GUJGASLTD",
                "GSPL",
                "HCLTECH", "HDFCAMC", "HDFCBANK", "HDFCLIFE", "HAVELLS", "HEROMOTOCO", "HINDALCO", "HAL", "HINDPETRO",
                "HINDUNILVR", "HINDZINC", "HDFC", "ICICIBANK", "ICICIGI", "ICICIPRULI", "ISEC", "IDFCFIRSTB", "ITC",
                "IBULHSGFIN", "INDIAMART", "INDHOTEL", "IOC", "IRCTC", "IGL", "INDUSTOWER", "INDUSINDBK", "NAUKRI",
                "INFY",
                "INDIGO", "IPCALAB", "JSWENERGY", "JSWSTEEL", "JINDALSTEL", "JUBLFOOD", "KOTAKBANK", "L&TFH", "LTTS",
                "LICHSGFIN", "LTI", "LT", "LAURUSLABS", "LUPIN", "MRF", "MGL", "M&MFIN", "M&M", "MANAPPURAM", "MARICO",
                "MARUTI", "MFSL", "MINDTREE", "MOTHERSUMI", "MPHASIS", "MUTHOOTFIN", "NATCOPHARM", "NMDC", "NTPC",
                "NAVINFLUOR", "NESTLEIND", "NAM-INDIA", "OBEROIRLTY", "ONGC", "OIL", "PIIND", "PAGEIND", "PETRONET",
                "PFIZER",
                "PIDILITIND", "PEL", "POLYCAB", "PFC", "POWERGRID", "PRESTIGE", "PGHH", "PNB", "RBLBANK", "RECLTD",
                "RELIANCE",
                "SBICARD", "SBILIFE", "SRF", "SANOFI", "SHREECEM", "SRTRANSFIN", "SIEMENS", "SBIN", "SAIL", "SUNPHARMA",
                "SUNTV", "SYNGENE", "TVSMOTOR", "TATACHEM", "TCS", "TATACONSUM", "TATAELXSI", "TATAMOTORS", "TATAPOWER",
                "TATASTEEL", "TECHM", "RAMCOCEM", "TITAN", "TORNTPHARM", "TORNTPOWER", "TRENT", "UPL", "ULTRACEMCO",
                "UNIONBANK", "UBL", "MCDOWELL-N", "VGUARD", "VBL", "VEDL", "IDEA", "VOLTAS", "WHIRLPOOL", "WIPRO",
                "YESBANK",
                "ZEEL"]


async def get_stocks_on_support(stocks_to_analyze=None):
    # nse = Nse()
    # codes = nse.get_stock_codes().keys()

    if stocks_to_analyze is None:
        stocks_to_analyze = codes
    logging.info("Starting the process.")

    coros = [find_stock(i) for i in stocks_to_analyze]
    response = await asyncio.gather(*coros)
    stocks = [stock for stock in response if stock is not None]
    logging.info(stocks)
    return stocks


async def find_stock(i):
    try:
        tick = yf.Ticker(i + '.NS')
        today_data = tick.history(period='3d', interval='5m', rounding=False)
        hist = tick.history(period='1y', rounding=False)
        fig = trendln.plot_support_resistance(hist[-1000:].Low, None, accuracy=1)
        ax = fig.gca()
        labels = list(map(lambda l: l._label, ax.lines))
        resistance = labels.index('Resistance')
        support = labels.index('Support')
        for j in range(support, resistance):
            line = ax.lines[j]
            x = line.get_xdata().tolist()
            y = line.get_ydata().tolist()

            model = LinearRegression()
            model.fit(np.array(x).reshape(-1, 1), np.array(y).reshape(-1, 1))
            # print(tick.info['regularMarketPrice'])
            X_predict = np.array([251])  # put the dates of which you want to predict kwh here
            y_predict = model.predict(X_predict.reshape(-1, 1))
            # print(i, ":", y_predict[0][0], scrip['lastPrice'])
            if abs(today_data['Close'][-1] - y_predict[0][0]) <= 0.02 * today_data['Close'][-1]:
                return i
                # print("Found :", i)
                # c = plt.plot(x, y)
                # plt.show()
    except Exception as e:
        logging.warning(e)
        pass


async def plot_stocks(chat_id, stocks_to_plot=None):
    # nse = Nse()
    # codes = nse.get_stock_codes().keys()

    if stocks_to_plot is None:
        stocks_to_plot = []
    logging.info("Starting the process.")

    coros = [plot_stock(chat_id, i) for i in stocks_to_plot]
    await asyncio.gather(*coros)


async def plot_stock(chat_id, i):
    try:
        tick = yf.Ticker(i + '.NS')
        hist = tick.history(period='1y', rounding=False)
        hist["EMA_100"] = hist.iloc[:, 3].ewm(span=100, adjust=False).mean()
        trendln.plot_support_resistance(hist[-1000:].Low, None, accuracy=1)
        plt.plot([ind for ind in range(0, 251)], hist['EMA_100'], color="seagreen")

        output = BytesIO()
        plt.gcf().savefig(output, format="png")
        image_as_string = output.getvalue()
        await bot.send_photo(chat_id=chat_id, photo=image_as_string, caption="#" + i)

    except Exception as e:
        logging.warning(e)
        pass


async def start(chat_id):
    message = 'Welcome to stock stalker.'
    await bot.send_message(chat_id, text=message)


async def echoMessage(chat_id, text):
    await bot.send_message(chat_id, text)


async def recommend(chat_id):
    global recommendations
    if len(recommendations) > 0:
        stocks = recommendations
        return await send_stocks_list(stocks, chat_id)
    else:
        return await force_recommend(chat_id)


async def force_recommend(chat_id):
    global recommendations
    await bot.send_message(chat_id, text="Process running, please wait.")
    stocks = await get_stocks_on_support()
    recommendations = stocks
    await send_stocks_list(stocks, chat_id)


async def analyze(chat_id, stocks_to_analyze):
    await bot.send_message(chat_id, text="Process running, please wait.")
    stocks = await get_stocks_on_support(stocks_to_analyze)
    await send_stocks_list(stocks, chat_id)


async def send_stocks_list(stocks, chat_id):
    if len(stocks) > 0:
        message = "Stocks found! :)\n"
        stocks_list = message + '\n'.join(stocks)
        await send_message_parts(chat_id, text=stocks_list)
    else:
        await send_message_parts(chat_id, text="Could not find any. :(")


async def morning(chat_id):
    global recommendations
    message = "Good Morning! Have a nice day!"
    await bot.send_message(chat_id, text=message)
    stocks = await get_stocks_on_support()
    recommendations = stocks
    # return await send_stocks_list(stocks, chat_id)


async def some_callback(bot, chat_id, defaultstocks, client_loop, time, all):
    await find_swing(bot, chat_id, defaultstocks, time, all, client_loop)


def between_callback(bot, chat_id, defaultstocks, time, all):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    loop.run_until_complete(some_callback(bot, chat_id, defaultstocks, loop, time, all))
    loop.close()


@dp.message_handler(commands=['run', 'force_run', 'trigger', 'start', 'swing', 'analyze', 'plot', 'plot_ha'])
async def echo(message: types.Message):
    logging.info(f'Received message from {message.from_user}')
    try:
        logging.info(message.text)
    except:
        logging.info('Cannot log message')
    if message.text == '/run' or message.text == '/run@stocks_stalker_bot':
        return await recommend(message.chat.id)
    elif message.text == '/force_run' or message.text == '/force_run@stocks_stalker_bot':
        return await force_recommend(message.chat.id)
    elif message.text == '/trigger' or message.text == '/trigger@stocks_stalker_bot':
        return await morning(message.chat.id)
    elif message.text == '/start' or message.text == '/start@stocks_stalker_bot':
        return await start(message.chat.id)
    elif "/analyze" in message.text:
        elements = message.text.split(" ")
        stocks = message.text.split(" ")[1:min(15, len(elements))]
        return await analyze(message.chat.id, stocks)
    elif "/plot_ha" in message.text:
        stock = message.text.split(" ")[1]
        logging.info("Plotting HA for " + stock)
        return await plot_chart(bot, message.chat.id, stock)
    elif "/plot" in message.text:
        elements = message.text.split(" ")
        stocks = message.text.split(" ")[1:min(15, len(elements))]
        return await plot_stocks(message.chat.id, stocks)
    elif '/swing' in message.text:
        elements = re.findall(r'\S+', message.text)

        defaultstocks = ['HINDUNILVR', 'POLYCAB', 'BHARTIARTL', 'NAVINFLUOR', 'VIPIND', 'AUBANK',
                         'AMBER', 'AMRUTANJAN', 'APLAPOLLO', 'BATAINDIA', 'CREDITACC', 'HDFCLIFE', 'ICICIGI',
                         'KPITTECH', 'LAURUSLABS', 'PIDILITIND', 'POLYMED', 'GLS', 'SIRCA', 'TATACONSUM',
                         'TATAELXSI', 'INFOBEAN', 'HAPPSTMNDS', 'IEX', 'JUBLINGREA', 'EXIDEIND', 'HDFCBANK',
                         'RELIANCE', 'SBICARD', 'SHREEPUSHK']

        extra = ['VINATIORGA', 'ASTRAL', 'HCLTECH', 'TCS', 'NEULANDLAB', 'AFFLE', 'SYNGENE', 'SEQUENT', 'MARICO',
                 'ATUL', 'LTTS', 'INFY', 'MINDTREE', 'RELAXO', 'AFFLE', 'TATACHEM', 'KOTAKBANK', 'TRENT', 'NAUKRI',
                 'ASIANPAINT', 'DEEPAKNTR', 'WHIRLPOOL', 'AWHCL', 'DIVISLAB', 'AMARAJABAT', 'BORORENEW', 'CLEAN',
                 'ULTRACEMCO', 'DEEPAKFERT', 'RALLIS']

        if len(elements) >= 2 and elements[1].isdigit():
            time = 0
            if len(elements) == 3:
                time = int(elements[-1])
            if int(elements[1]) == 1:
                await find_swing(bot, message.chat.id, defaultstocks, time, 1)
                return
            if int(elements[1]) == 2:
                return await find_swing(bot, message.chat.id, extra, time, 2)
            if int(elements[1]) == 0:
                defaultstocks.extend(extra)
                thread = Thread(target=between_callback,
                                kwargs={'bot': bot, 'chat_id': message.chat.id, 'defaultstocks': defaultstocks,
                                        'time': time, 'all': 0})
                thread.start()
                sleep(20)
                return 'started'
        elif len(elements) >= 2:
            return await find_swing(bot, message.chat.id, elements[1:], -1)
        return await find_swing(bot, message.chat.id, defaultstocks)


async def on_startup(dp):
    logging.warning('Starting connection. ')
    await bot.set_webhook(WEBHOOK_URL, drop_pending_updates=True)


async def on_shutdown(dp):
    logging.warning('Bye! Shutting down webhook connection')


MAX_MESSAGE_LENGTH = 1000


async def send_message_parts(chat_id, text: str):
    global MAX_MESSAGE_LENGTH
    if len(text) <= MAX_MESSAGE_LENGTH:
        await bot.send_message(chat_id, text)
        return

    parts = []
    while len(text) > 0:
        if len(text) > MAX_MESSAGE_LENGTH:
            part = text[:MAX_MESSAGE_LENGTH]
            first_lnbr = part.rfind('\n')
            if first_lnbr != -1:
                parts.append(part[:first_lnbr])
                text = text[(first_lnbr + 1):]
            else:
                parts.append(part)
                text = text[MAX_MESSAGE_LENGTH:]
        else:
            parts.append(text)
            break

    for part in parts:
        await bot.send_message(chat_id, part)


def main():
    logging.basicConfig(level=logging.INFO)
    start_webhook(
        dispatcher=dp,
        webhook_path=WEBHOOK_PATH,
        skip_updates=True,
        on_startup=on_startup,
        host=WEBAPP_HOST,
        port=WEBAPP_PORT
    )
