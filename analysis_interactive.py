# interactive analysis 
# 
# %%
from IPython import get_ipython


# %%
# ANALYSIS of CRYPTO
# https://matplotlib.org/3.2.2/api/_as_gen/matplotlib.pyplot.plot.html


# %%
import numpy as np
import pandas as pd
import os
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import rc
from pandas.plotting import register_matplotlib_converters
import seaborn as sns
from pylab import rcParams
import analysis_ta as analysis
import analysis_lambda as a
import analysis_lib as alib
from matplotlib import gridspec
import math
import analysis_science

# import ta as tanewlib

# PLOTTING SETUP
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")
register_matplotlib_converters()
sns.set(style='whitegrid', palette='muted', font_scale=1.5)
rcParams['figure.figsize'] = 22, 10

# COMMODITY
dateTimeObj = dt.datetime.now()
today = dateTimeObj.strftime("%Y-%m-%d")

offline = False
symbol = "BTC-USD"
# symbol = "ETH-USD"
# symbol = "XRP-USD"
# symbol = "AMZN"
# symbol = "^NDX"
# symbol = "WORK"
# symbol = "^GSPC"
# symbol = "NIO"
# symbol = "TSLA"
# symbol = "FIT"
# symbol = "NVDA"
date = today
filename = '/data/%s/Yahoo_BTCUSD_d.csv.ta.csv' % symbol

# amount of days to display
range_days = 30*12
# %%
def ta(df):
    return df

# %%
# offline = False
print("reading " + symbol)
# if offline:
#     path = os.getcwd()
#     # source = path + '/data/BTC/Bitstamp_BTCUSD_d.csv.ta.csv'
#     source = path + filename
#     # source = path + '/data/XRP-USD/Yahoo_XRPUSD_d.csv.ta.csv'
#     # source = path + '/data/^IXIC/Yahoo_Nasdaq_d.csv.ta.csv'
#     df_source = pd.read_csv(
#     source
#     )

# else:

signal_list = ["vol_flow",
                "NORMALIZE",
                "ALLIGATOR",
                "DPO",
                "SUPERTREND",
                "TREND_SIGNAL",
                'PriceDiffPerc',
                'PriceLevel',
                'CHANDELIER',
                'TSF_Price',
                'BBANDS',
                'MACD',
                'CHOP',
                'ROC',
                'ADX',
                'EMA21Volume',
                'HMA200',
                'EMA100',
                'EMA21',
                'uptrend_EMA',
                'uptrend_EMAPrev',
                'uptrend_MA50MA200',
                'uptrend_MA50MA200Prev',
                'HMAPriceXOverPerc',
                'RSI',"ATR",
                'AROON','variance','pivot','STRATEGY','CHOP',
                'TREND']


if date is None:
    date = dt.datetime.now().strftime("%Y-%m-%d")
# default: it will download just the last year
df_source = a.download(symbol, date, 365*5)

print("generating indicators")
df_source = a.process(df_source, signal_list)
df_source = ta(df_source)
# the dataset
df_source["Date"] = pd.to_datetime(df_source["Date"], infer_datetime_format=True)
df_source.set_index("Date")
# print(df_source.head(5))

# the dataset subrange
df = df_source.loc[df_source.Date <= today].tail(365*2)
df.set_index("Date")

print("generating resistance and supports")
volume = "Volume" 
global_support_resistance = df.groupby(['PriceLevel']).agg({volume: ['sum'],'PriceLevel': ['count']})
global_support_resistance.columns = ['volume', 'count']
global_support_resistance = global_support_resistance.loc[global_support_resistance["count"] > 1]

# %%
def daily_analysis(df, date):

    df_analysis = df.loc[df.Date <= date].tail(60)

    print("***** Analysis - ",date)

    result = analysis.analysis_trend_pulse(df_analysis,"")
    print("")
    print(result['description'])
    # print(result)

# dateTimeObj = dt.datetime.now()
# nowStr = dateTimeObj.strftime("%Y-%m-%d")
# daily_analysis(df_source,"2020-07-31")
daily_analysis(df_source,today)



# %%

def generate_support_resistance(df, value):

    filter = df[(0.9 * value < df.index) & (df.index < value * 1.1)]
    criteria = "volume"
    _average = filter[[criteria]].mean()
    filter = filter[(filter[criteria] > np.int(_average))]
    # print(filter.tail(10))
    return filter

def generate_support_resistance_lines(ax, df, local_support_resistance, measure):
    return
    # print(df['Close'].iat[len(df)-1])
    support_resistance = generate_support_resistance(local_support_resistance, df[measure].iat[len(df)-1])
    support_resistance['level'] = support_resistance.index
    # print(len(df))
    for i in range (0,len(support_resistance)-1):
        # print(support_resistance['level'].iat[i])
        ax.axhline( support_resistance['level'].iat[i], color="grey", alpha=support_resistance['count'].iat[i]/10)

def generate_cross(df,measure, label, start=1):
    df[label] = 0
    for i in range(start, len(df)):
        df[label].iat[i] = np.sign(df[measure].iat[i]) if np.sign(df[measure].iat[i]) != np.sign(df[measure].iat[i-1]) else 0
    return df  

def generate_dots(ax, df, measure_to_calc, measure_to_show, start=1, greenDot='gD', redDot='rD', greenLabel = "long", redLabel="short", offset=0):
    label = "flag"
    df = generate_cross(df,measure_to_calc,label)

    df["long" ] = df.apply (lambda row: None if math.isnan(row[label]) or row[label]!= 1  else row[measure_to_show] * (1+offset), axis=1)
    df["short"] = df.apply (lambda row: None if math.isnan(row[label]) or row[label]!=-1  else row[measure_to_show] * (1-offset), axis=1)
    ax.plot(df['long'], greenDot, markevery=None, label=greenLabel)
    ax.plot(df['short'], redDot, markevery=None, label=redLabel)
    df = df.drop(columns=[label,"long","short"])

# def generate_minmax(df, measure, label, start=2):
#     der_label = "der"
#     df["xx"] = df[measure].calc.derivative()
#     df[der_label] = df["xx"].calc.derivative()
#     df[label] = 0
#     for i in range(start, len(df)):
#         if np.isnan(df[der_label].iat[i]) or np.isnan(df[der_label].iat[i-1]) or np.isnan(df[der_label].iat[i-2]):
#             # print(df[label].iat[i-2],df[label].iat[i-1],df[label].iat[i])
#             continue
#         if np.sign(df[der_label].iat[i-2]) != np.sign(df[der_label].iat[i-1]) and np.sign(df[der_label].iat[i-1]) != np.sign(df[der_label].iat[i]):
#             if np.sign(df[der_label].iat[i-2]) == np.sign(df[der_label].iat[i]):
#                 # derivative changed sign
#                 print(df[der_label].iat[i-2],df[der_label].iat[i-1],df[der_label].iat[i])
#                 df[label ].iat[i-1] = np.sign(df[der_label].iat[i-1])
#         # df[label].iat[i] = np.sign(df[measure].iat[i]) if np.sign(df[measure].iat[i]) != np.sign(df[measure].iat[i-1]) else 0
#     print(df[der_label].tail(30))
#     df = df.drop(columns=[der_label])
#     return df  

# def generate_minmax_dots(ax, df, measure_to_calc, measure_to_show, start=2, greenDot='gv', redDot='r^', greenLabel = "max", redLabel="min"):
#     label = "derivative"
#     df = generate_minmax(df,measure_to_calc,label, start=start)

#     df["down" ] = df.apply (lambda row: None if math.isnan(row[label]) or row[label]!= 1  else row[measure_to_show] , axis=1)
#     # df["zero" ] = df.apply (lambda row: None if math.isnan(row[label]) or row[label]!= 0  else row[measure_to_show] , axis=1)
#     df["up"] = df.apply (lambda row: None if math.isnan(row[label]) or row[label]!=-1  else row[measure_to_show] , axis=1)
#     ax.plot(df['up'], greenDot, markevery=None, label=greenLabel)
#     # ax.plot(df['zero'], "kx", markevery=None, label="zero")
#     ax.plot(df['down'], redDot, markevery=None, label=redLabel)
#     df = df.drop(columns=[label,"up","down"])

def generate_minmax_dots(ax,df,measure_to_calc, measure_to_show, period=5, greenDot='go', redDot='ro', greenLabel = "min", redLabel="max"):
    FlowMinMax = analysis_science.minmax(df,measure_to_calc,"FlowMinMax",period)
    df = FlowMinMax
    # # min max
    df["min"] = df.apply (lambda row: None if math.isnan(row['FlowMinMax']) or row['FlowMinMax']!=-1 else row[measure_to_show], axis=1)
    df["max"] = df.apply (lambda row: None if math.isnan(row['FlowMinMax']) or row['FlowMinMax']!=1 else row[measure_to_show], axis=1)
    ax.plot(df['min'], greenDot, markevery=None, label=greenLabel)
    ax.plot(df['max'], redDot, markevery=None, label=redLabel)
    df = df.drop(columns=["FlowMinMax","min","max"])

# %%
def ax_chart(df, ax, log=True, ema=True, sar=True, superTrend=True, chandelier=False, support=True, bbands=False):
    if log:
        ax.set_yscale('log')

    ax.plot(df['Close'], label='close', color='black')
    ax.plot(df['HMA200'], label='hma200', color = 'blue')
    ax.plot(df['SMA200'], label='sma200', color="crimson")

    if ema:
        ax.plot(df['EMA21'], label='ema21', color="orchid")
        ax.plot(df['EMA100'], label='ema100', color="deeppink")
        ax.plot(df['SMA200'], label='sma200', color="crimson")

    if sar:
        ax.plot(df['SAR'], 'k|', markevery=None, label='SAR')
    if superTrend:
        ax.plot(df['SuperTrend'], label='supertrend', color="slategrey")

    if chandelier:
        ax.plot(df['CHANDELIER_EXIT'], label='supertrend', color="green")
        # ax.plot(df['CHANDELIER_SHORT'], label='supertrend', color="red")

    if support:
        generate_support_resistance_lines(ax, df, global_support_resistance, "Close")
        # # print(df['Close'].iat[len(df)-1])
        # support_resistance = generate_support_resistance(global_support_resistance, df['Close'].iat[len(df)-1])
        # support_resistance['level'] = support_resistance.index
        # # print(len(df))
        # for i in range (0,len(support_resistance)-1):
        #     # print(support_resistance['level'].iat[i])
        #     ax.axhline( support_resistance['level'].iat[i], color="grey", alpha=0.5)
    if bbands:
        ax.plot(df['BBANDS_HI'], label='bb hi', color="green",alpha=0.5)
        ax.plot(df['BBANDS_MID'], label='bb mid', color="blue",alpha=0.5)
        ax.plot(df['BBANDS_LOW'], label='bb low', color="red",alpha=0.5)

    ax.legend(loc = 'upper left');

def ax_bband_ratio(df, ax):

    ax.plot(df["BBANDS_OSC"], label='bb%', color="black")
    ax.plot(df['BBANDS_SQUEEZE'], label='bb squeeze', color='blue')
    # ax.plot(df['BBANDS_OSC_MA7'], label='ma7', color='blue', alpha=0.7)
    ax.axhline(   1, color="green" ,alpha=0.5)
    ax.axhline( 0.8, color="green" ,alpha=0.5, linestyle='--')
    ax.axhline( 0.5, color="yellow",alpha=0.5)
    ax.axhline( 0.2, color="red"   ,alpha=0.5, linestyle='--')
    ax.axhline( 0.08, color="red"  ,alpha=0.5, linestyle='dotted')
    ax.axhline(   0, color="red"   ,alpha=0.5)

    ax.legend(loc = 'upper left')

def ax_rsi(df, ax):
    ax.plot(df['RSI'], label='RSI')
    ax.axhline( 70, color="red", alpha=0.5)
    ax.axhline( 30, color="green", alpha=0.5)
    ax.legend(loc = 'upper left');

def ax_adx(df, ax):
    ax.plot(df['ADX'], label='ADX')
    ax.axhline( 25, color="green", alpha=0.5)
    ax.legend(loc = 'upper left');

def ax_macd(df, ax):
    ax.plot(df['MACD'], label='MACD')
    ax.plot(df['MACD_SIGNAL'], label='MACD_SIGNAL')

    ax.legend(title = "Vol Flow Strategy", loc = 'upper left');

# Vol flow strategy
def ax_volflow(df,ax):
    ax.plot(df['vol_flow'], label='vol flow')

    FlowMinMax = analysis_science.minmax(df,"vol_flow","FlowMinMax",5)
    df = FlowMinMax
    # # min max
    df["min"] = df.apply (lambda row: None if math.isnan(row['FlowMinMax']) or row['FlowMinMax']!=-1 else row['vol_flow'], axis=1)
    df["max"] = df.apply (lambda row: None if math.isnan(row['FlowMinMax']) or row['FlowMinMax']!=1 else row['vol_flow'], axis=1)
    ax.plot(df['min'], 'go', markevery=None, label='min')
    ax.plot(df['max'], 'ro', markevery=None, label='max')

    PlotTrendMin = df[['min']].dropna()
    PlotTrendMax = df[['max']].dropna()
    ax.plot(PlotTrendMin['min'])
    ax.plot(PlotTrendMax['max'])

    df["long" ] = df.apply (lambda row: None if math.isnan(row['vol_signaltopbottom']) or row['vol_signaltopbottom']!= 1  else row['vol_flow'], axis=1)
    df["short"] = df.apply (lambda row: None if math.isnan(row['vol_signaltopbottom']) or row['vol_signaltopbottom']!=-1  else row['vol_flow'], axis=1)
    ax.plot(df['long'], 'gx', markevery=None, label='long')
    ax.plot(df['short'], 'rx', markevery=None, label='short')   
    ax.legend(title = "Vol Flow Strategy", loc = 'upper left');

    df = df.drop(columns=['long',"short","min","max"])

def ax_volTrendStrategy(df,ax, strategy="strategy"):
    ax.plot(df['Close'], label='close',color='black')
    ax.plot(df['HMA200'], label='hma200')
    ax.plot(df['SAR'], 'k|', markevery=None, label='SAR')
    ax.plot(df['SuperTrend'], label='supertrend')

    df["strategy_sell" ] = df.apply (lambda row: None if  row[strategy +'_helper'] != "sell" else row['Close'], axis=1)
    df["strategy_buy"  ] = df.apply (lambda row: None if  row[strategy +'_helper'] != "buy"  else row['Close'], axis=1)

    df["strategy_short_tp" ] = df.apply (lambda row: None if  row[strategy +'_helper'] != "short tp" else row['Close'], axis=1)
    df["strategy_long_tp"  ] = df.apply (lambda row: None if  row[strategy +'_helper'] != "long tp"  else row['Close'], axis=1)

    df["strategy_sell_tp" ] = df.apply (lambda row: None if  row[strategy +'_helper'] != "end short" else row['Close'], axis=1)
    df["strategy_buy_tp"  ] = df.apply (lambda row: None if  row[strategy +'_helper'] != "end long"  else row['Close'], axis=1)

    df["strategy_start_buy"   ] = df.apply (lambda row: None if  row[strategy +'_helper'] != "weak buy" else row['Close'], axis=1)
    df["strategy_start_sell"  ] = df.apply (lambda row: None if  row[strategy +'_helper'] != "weak sell"  else row['Close'], axis=1)

    ax.plot(df['strategy_short_tp'], 'k^', markevery=None, label='tp')
    ax.plot(df['strategy_long_tp'], 'bv', markevery=None, label='tp')

    ax.plot(df['strategy_sell_tp'], 'rX', markevery=None, label='end short')
    ax.plot(df['strategy_buy_tp'], 'gX', markevery=None, label='end long')

    ax.plot(df['strategy_buy'], 'go', markevery=None, label='long')
    ax.plot(df['strategy_sell'], 'ro', markevery=None, label='short')

    ax.plot(df['strategy_start_buy'], 'kv', markevery=None, label='weak buy')
    ax.plot(df['strategy_start_sell'], 'b^', markevery=None, label='weak sell')

    ax.legend(title = "Vol Trend Strategy", loc = 'upper left');

def ax_alligator(df,ax):
    ax.plot(df['Close'], label='close', color='black')
    ax.plot(df['jaw'], label='jaw')
    ax.plot(df['teeth'], label='teeth')
    ax.plot(df['lips'], label='lips')
    ax.legend(title = "Alligator", loc = 'upper left')

def ax_voltrend(df,ax):
    ax.plot(df['TREND'], label='vol trend')
    ax.legend(title = "Vol Trend", loc = 'upper left')

def ax_chop(df,ax):
    ax.plot(df['CHOP'], label='chop')
    ax.axhline( 61.8, color="red", alpha=0.5)
    ax.axhline( 38.2, color="green", alpha=0.5)
    ax.legend(title = "Chop", loc = 'upper left')

def ax_volumeStrategy(df,ax):
    df["vol_tmp"] = df.apply (lambda row: np.sign(row["TREND"]) if np.sign(row["TREND"]) == np.sign(row["vol_flow"]) else 0 , axis=1)
    ax.plot(df['vol_tmp'], label='vol bull/bear', color="blue")
    ax.plot(np.sign(df['DPO']),   label='DPO bull/bear', color = "black")

    df["bullbear"] = 0
    for i in range(2, len(df)):
        # if   df["volume_signal"].iat[i] == 0 and df["volume_signal"].iat[i-1] == -1:
        #     df["bullbear"].iat[i] = 0.5
        # elif df["volume_signal"].iat[i] == 0 and df["volume_signal"].iat[i-1] ==  1:
        #     df["bullbear"].iat[i] = -0.5

        if df["DPO"].iat[i] > 0 and df["DPO"].iat[i-1] > 0 and df["DPO"].iat[i-2] <= 0 and df["vol_tmp"].iat[i] > 0:
            df["bullbear"].iat[i] = 1
        elif df["DPO"].iat[i] <= 0 and df["DPO"].iat[i-1] <= 0 and df["DPO"].iat[i-2] > 0 and df["vol_tmp"].iat[i] < 0:
            df["bullbear"].iat[i] = -1

    df["bull"] = df.apply (lambda row: None if math.isnan(row['bullbear']) or row['bullbear']!= 1  else -1, axis=1)
    df["bear"] = df.apply (lambda row: None if math.isnan(row['bullbear']) or row['bullbear']!=-1  else  1, axis=1)
    ax.plot(df['bull'], 'gD', markevery=None, label='bull')
    ax.plot(df['bear'], 'rD', markevery=None, label='bear')

    # df["down"] = df.apply (lambda row: -1 if row['bullbear']==-0.5  else None, axis=1)
    # df["up"  ] = df.apply (lambda row:  1 if row['bullbear']== 0.5  else None, axis=1)
    # ax.plot(df['down'], 'gD', markevery=None, label='down')
    # ax.plot(df['up'  ], 'rD', markevery=None, label='up')

    ax.legend(title = "Vol Strategy", loc = 'upper left')
    df = df.drop(columns=['bullbear',"bull","bear","vol_tmp"])


def ax_dpo(df,ax):
    ax.plot(df['DPO'], label='DPO', color='black')
    ax.plot(df['DPO-HMA200'], label='DPO-HMA200', color='blue')
    ax.plot(df['DPO-SMA200'], label='DPO-SMA200', color='red')
    ax.legend(title = "DPO", loc = 'upper left')

def ax_prob(df,ax):
    df_prob = df[['pNeg', 'pPos', 'pZer']].dropna()
   
    # ax.bar(df_prob.index,  df_prob['pPos'], width, label='Pos', color='green')
    # ax.bar(df_prob.index,  df_prob['pZer'], width, label='Zer', color='yellow', bottom=df_prob['pPos'])
    # ax.bar(df_prob.index,  df_prob['pNeg'], width, label='Neg', color='red', bottom=df_prob['pPos']) 

    
    # ax.plot(df['pNeg'], label='p(-)', color='red')
    # ax.axhline( 0.5, color="green", alpha=0.5)
    # ax.plot(df['pPos']-df["pNeg"], label='p(+)', color='blue')
    # ax.plot(df['pPos']-df["pNeg"], label='p(+)', color='black')

    df_prob["prob_up"  ] = df_prob.apply (lambda row: None if (row['pPos'] is None) or math.isnan(row['pPos']) or row['pPos'] <  0.7  else  1, axis=1)
    df_prob["prob_down"] = df_prob.apply (lambda row: None if (row['pNeg'] is None) or math.isnan(row['pNeg']) or row['pNeg'] <  0.7  else -1, axis=1)

    width = dt.timedelta(days=1)
    ax.bar(df_prob.index,  df_prob['prob_up']  , width, label='Pos', color='green')
    ax.bar(df_prob.index,  df_prob['prob_down'], width, label='Neg', color='red')

    # ax.plot(df['pDiff'], label='p(+)', color='black')
    # ax.axhline( 0.70, color="green", alpha=0.5)
    # ax.axhline( 0, color="yellow", alpha=0.5)
    # ax.axhline( -0.7, color="red", alpha=0.5)

    # ax.legend(title = "Prob", loc = 'upper left')

def ax_volume(df,ax):
    df_prob = df[['Open','Close','Volume','EMA21Volume']].dropna()
    width = dt.timedelta(days=1)

    df_prob["vol_pos"] = df.apply (lambda row: None if math.isnan(row['Volume']) or row['Open'] <= row['Close']  else row['Volume'], axis=1)
    df_prob["vol_neg"] = df.apply (lambda row: None if math.isnan(row['Volume']) or row['Open'] >= row['Close']  else row['Volume'], axis=1)
    ax.bar(df_prob.index,  df_prob['vol_pos'], width, label='Volume', color='green')
    ax.bar(df_prob.index,  df_prob['vol_neg'], width, label='Volume', color='red')
    ax.plot(df['EMA21Volume'], label='EMA21 Vol', color='black')
    ax.legend(title = "DPO", loc = 'upper left')
    # ax.ylim(top=1.5)
    # ax2.bar(df_prob.index,  df_prob['pZer'], width, label='Zer', color='yellow', bottom=df_prob['pPos'])
    # ax.bar(df_prob.index,  df_prob['pNeg'], width, label='Neg', color='red', bottom=df_prob['pPos']) 

def ax_hma_n(df,ax):
    ax.plot(df['HMA200-N'], label='HMA200-N', color='blue')
    ax.plot(df['HMA200-N-SMA200'], label='SMA200-R', color='red')
    ax.legend(title = "HMA", loc = 'upper left');

def ax_sma_n(df,ax):
    ax.plot(df['SMA200-N'], label='SMA200-N', color='red')
    # ax.plot(df['HMA200-N'], label='HMA200-N', color='black')
    ax.plot(df['SMA200-N-HMA200'], label='HMA200-R', color='blue')
    ax.legend(title = "SMA", loc = 'upper left')

def ax_alligator_oscillator(df,ax):
    ax.plot(df['alligator_oscillator'], label='alligator')

    generate_minmax_dots(ax,df,"alligator_oscillator", "alligator_oscillator", period=5, greenDot='go', redDot='ro', greenLabel = "min", redLabel="max")

    ax.legend(title = "alligator", loc = 'upper left')

def ax_aroon_oscillator(df,ax):
    ax.plot(df['AROON_OSC'], label='Aroon')
    ax.axhline( 50, color="green") 
    ax.axhline(-50, color="red") 
    generate_dots(ax, df, 'AROON_OSC',  'AROON_OSC', start=1, greenDot='gD', redDot='rD', greenLabel = "long", redLabel="short")
    ax.legend(title = "aroon osc", loc = 'upper left')

def ax_macd_history(df,ax):
    ax.plot(df['MACD_HISTORY'], label='macd')

    df = generate_cross(df,'MACD_HISTORY',"flag")

    # df["long" ] = df.apply (lambda row: None if math.isnan(row['flag']) or row['flag']!= 1  else row['MACD_HISTORY'], axis=1)
    # df["short"] = df.apply (lambda row: None if math.isnan(row['flag']) or row['flag']!=-1  else row['MACD_HISTORY'], axis=1)
    # ax.plot(df['long'], 'gD', markevery=None, label='long')
    # ax.plot(df['short'], 'rD', markevery=None, label='short')
    # df = df.drop(columns=['flag',"long","short"])

    generate_dots(ax, df, 'MACD_HISTORY', 'MACD_HISTORY', start=1, greenDot='gD', redDot='rD', greenLabel = "long", redLabel="short")

    generate_minmax_dots(ax,df,"MACD_HISTORY", "MACD_HISTORY", period=5, greenDot='gx', redDot='rx', greenLabel = "min", redLabel="max")


    # df["derivative"] = df['MACD_HISTORY'].calc.derivative()
    # df["flex"] = df['derivative'].calc.derivative()
    # generate_dots(ax, df, 'derivative', "flag", 'MACD_HISTORY', start=1, greenDot='gx', redDot='rx', greenLabel = "up", redLabel="down")

    ax.legend(title = "macd history", loc = 'upper left')


# %%
def plot_chart(df):

    fig = plt.figure()

    gs = gridspec.GridSpec(ncols=1, nrows=3, width_ratios=[1], height_ratios=[3,1,1]) 
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[1,0], sharex=ax1)
    ax3 = fig.add_subplot(gs[2,0], sharex=ax1)

    ax_chart(df, ax1, log=True, ema=False, sar=False, superTrend=False, chandelier=False)
    ax_dpo(df,ax2)
    ax_rsi(df, ax3)
    # ax_macd(df, ax3)


# plot_chart(df.loc[df.Date <= today].tail(range_days))
plot_chart(df_source)
# %%
def plot_strategy(df, strategy = "strategy"):

    fig = plt.figure()
    gs = gridspec.GridSpec(ncols=1, nrows=3, width_ratios=[1], height_ratios=[3,1,1]) 
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[1,0], sharex=ax1)
    ax3 = fig.add_subplot(gs[2,0], sharex=ax1)

    ax_chart(df, ax1, log=True, ema=False, sar=True, superTrend=False, chandelier=False)
    # ax_chop(df,ax2)
    ax_aroon_oscillator(df,ax2)
    ax_macd_history(df,ax3)
    # ax3.plot(df['AROON_UP'], label='Aroon Up')
    # ax3.plot(df['AROON_DOWN'], label='Aroon Down')
    # ax_volflow(df,ax2)

plot_strategy(df.loc[df.Date <= today].tail(range_days))

# %%

def plot_alligator(df, strategy = "strategy"):
    # import pandas_helper_calc

    fig = plt.figure()
    gs = gridspec.GridSpec(ncols=1, nrows=3, width_ratios=[1], height_ratios=[1,1,1]) 
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[1,0], sharex=ax1)
    ax3 = fig.add_subplot(gs[2,0], sharex=ax1)

    ax_alligator(df,ax1)
    # < 0 price is falling
    # ax2.plot(df['alligator_oscillator'], label='direction', color='blue')
    ax_chop(df,ax2)
    # > 0 oscillator start trending up
    # ax3.plot(np.sign(df['alligator_oscillator'].calc.derivative()), label='direction', color='red')
    ax_alligator_oscillator(df,ax3)

    # ax_volumeStrategy(df,ax2)

# plot_alligator(df.loc[df.Date <= today].tail(range_days))


# %%
def plot_averages(df):

    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)

    # ax_chart(df, ax1, log=True, ema=False, sar=True, superTrend=False, chandelier=True)
    ax_dpo(df,ax1)
    ax_hma_n(df,ax2)
    ax_sma_n(df,ax3)

plot_averages(df.loc[df.Date <= today].tail(range_days))
# %%
def plot_chartvolume(df):

    fig = plt.figure()

    gs = gridspec.GridSpec(ncols=1, nrows=2, width_ratios=[1], height_ratios=[3,1]) 
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[1,0], sharex=ax1)


    ax_chart(df, ax1, log=False, ema=False, sar=False, superTrend=False)
    # put bullets on supertrend and sar
    df["temp"] =  df["Close"] - df["SAR"]
    generate_dots(ax1, df, 'temp', 'Close', start=1, greenDot='gD', redDot='rD', greenLabel = "SAR Up", redLabel="SAR Down", offset = 0.01)
    df["temp"] = df["Close"] - df["SuperTrend"]
    generate_dots(ax1, df, 'temp', 'Close', start=1, greenDot='g*', redDot='r*', greenLabel = "STrend Up", redLabel="STrend Down", offset = 0.01)
    # generate_dots(ax1, df, 'MACD_HISTORY', 'Close', start=1, greenDot='gX', redDot='rX', greenLabel = "MACD up", redLabel="MACD down", offset = 0.01)
    generate_minmax_dots(ax1,df,"MACD_HISTORY", "Close", period=5, greenDot='gX', redDot='rX', greenLabel = "MACD up", redLabel="MACD down")
    generate_dots(ax1, df, 'vol_flow', 'Close', start=1, greenDot='g+', redDot='r+', greenLabel = "Vol bull", redLabel="Vol bear", offset = 0.01)
    generate_dots(ax1, df, 'alligator_oscillator', 'Close', start=1, greenDot='g^', redDot='rv', greenLabel = "All up", redLabel="All down", offset = 0)

    ax1.legend(title = "Strategies", loc = 'upper left')
    # ax_volume(df,ax2)

plot_chartvolume(df.loc[df.Date <= today].tail(range_days))
# %%

#  PROBABILITY
def probability(df_source, period=7, Price = "Close", Value = 'HMAPriceXOverPerc', multiplier=1):

    df_source["pPos"] = None
    df_source["pNeg"] = None
    df_source["pZer"] = None

    m = alib.build_probability_matrix(Value,  df_source, Price,  period=period, multiplier=multiplier)
    match = 0
    for i in range(1, len(df_source)):
        # Close = df_source[Price].iat[i]
        df_source["pNeg"].iat[i],df_source["pPos"].iat[i],df_source["pZer"].iat[i] = alib.updown_probability(df_source[Value].iat[i], m)
               
        # df_source["pDiff"].iat[i] = df_source["pPos"].iat[i] - (df_source["pNeg"].iat[i])
        # +df_source["pZer"].iat[i]
        # 
    return df_source

def prob_plot(ax1,ax2, subset, df, value, label, threshold, marker, multiplier=1):
    
    df_prob = probability(subset[["Close",value]], period=2, Price = "Close", Value = value, multiplier=multiplier)
    df_prob = df_prob.tail(len(df))
    # reduce df_prob to the values contained in the index of df (that is shorter)

    df_prob["prob_up"  ] = df_prob.apply (lambda row: None if (row['pPos'] is None) or math.isnan(row['pPos']) or row['pPos'] < threshold  else row["Close"] * 1.005, axis=1)
    df_prob["prob_down"] = df_prob.apply (lambda row: None if (row['pNeg'] is None) or math.isnan(row['pNeg']) or row['pNeg'] < threshold  else row["Close"] * 0.995, axis=1)
    ax1.plot(df_prob['prob_up'  ], 'b' + marker, markevery=None, label=label + ' p(+)')
    ax1.plot(df_prob['prob_down'], 'r' + marker, markevery=None, label=label + ' p(-)') 
    # print(df_prob.tail(10))
    df_prob = df_prob.drop(columns=['prob_up',"prob_down","pPos","pNeg","pZer"])

def plot_probability(df_training, df):

    fig = plt.figure()

    gs = gridspec.GridSpec(ncols=1, nrows=1, width_ratios=[1], height_ratios=[3]) 
    ax1 = fig.add_subplot(gs[0,0])
    # ax2 = fig.add_subplot(gs[1,0], sharex=ax1)
    # ax3 = fig.add_subplot(gs[2,0], sharex=ax1)
    ax2 = None
    # subset = df_training.loc[df_training.Date <= today].tail(365*2)
    ax_chart(df, ax1, log=False, ema=False, sar=False, superTrend=False, bbands=False, support=False)
    prob_plot(ax1, ax2, df_training, df, 'HMAPriceXOverPerc', "HMA", 0.7, "o")
    prob_plot(ax1, ax2, df_training, df, 'MACD_HISTORY', "MACD", 0.97, "X")
    prob_plot(ax1, ax2, df_training, df, 'RSI', "RSI", 0.7, "s")
    prob_plot(ax1, ax2, df_training, df, 'ROC', "ROC", 0.7, "p")
    df_training["bb"] = df_training["BBANDS_OSC"] * 100
    prob_plot(ax1, ax2, df_training, df, 'bb', "BB%", 0.7, "*")
    df_training["bb"] = 100 * df_training["SMA200-N"]
    prob_plot(ax1, ax2, df_training, df, 'bb', "SMA-N", 0.7, "D")
    df_training["bb"] = 100* df_training["BBANDS_SQUEEZE"]
    prob_plot(ax1, ax2, df_training, df, 'bb', "BB Sq", 0.55, "H")
    # ax_prob(df,ax2)
    # ax_adx(df,ax3)
    ax1.legend(title = "Probabilities", loc = 'upper left')

plot_probability(df_source.loc[df_source.Date <= today].tail(356*2), df.loc[df.Date <= today].tail(60))

# %%
def plot_volume(df, strategy = "strategy"):

    fig = plt.figure()

    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)

    ax_voltrend(df,ax2)
    ax_volflow(df,ax1)
    ax_volumeStrategy(df,ax3)

# plot_volume(df.loc[df.Date <= today].tail(360))
# %%
def plot_pivot(df, strategy = "strategy"):

    # fig = plt.figure()
    fig = plt.figure()
    gs = gridspec.GridSpec(ncols=1, nrows=2, width_ratios=[1], height_ratios=[3,1]) 
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[1,0], sharex=ax1)

    df["pivot"] = df["pivot"].shift(1)
    df["R1"] = df["R1"].shift(1)
    df["R2"] = df["R2"].shift(1)
    df["S1"] = df["S1"].shift(1)
    df["S2"] = df["S2"].shift(1)

    ax1.plot(df['pivot'], label='pivot', color='yellow')
    ax1.plot(df['R1'], label='R1', color='darkred', alpha=0.5)
    ax1.plot(df['R2'], label='R2', color='red', alpha=0.5)
    ax1.plot(df['S1'], label='S1', color='green', alpha=0.5)
    ax1.plot(df['S2'], label='S2', color='darkgreen', alpha=0.5)

    ax_chart(df, ax1, log=False, ema=False, sar=False, superTrend=False, bbands=False)
    ax2.plot(df['variance_perc'], label='variance', color='green')
    # ax_bband_ratio(df, ax3)

plot_pivot(df.loc[df.Date <= today].tail(range_days))

# %%
def plot_bb(df, strategy = "strategy"):

    fig = plt.figure()
    gs = gridspec.GridSpec(ncols=1, nrows=3, width_ratios=[1], height_ratios=[3,1,1]) 
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[1,0], sharex=ax1)
    ax3 = fig.add_subplot(gs[2,0], sharex=ax1)

    ax_chart(df, ax1, log=False, ema=False, sar=False, superTrend=False, bbands=True)
    # ax2.plot(df['BBANDS_HI']/df['BBANDS_LOW'], label='variance', color='green')

    ax_bband_ratio(df, ax2)
    ax3.plot(df['vol_flow'], label='vol flow')
    

# plot_bb(df.loc[df.Date <= today].tail(365))

# %%
def plot_everything(df, strategy = "strategy"):

    fig = plt.figure(constrained_layout=True)
    gs = gridspec.GridSpec(ncols=2, nrows=5, width_ratios=[1,1], height_ratios=[2,1,1,1,1]) 
    ax1 = fig.add_subplot(gs[0,:])
    # ax2 = fig.add_subplot(gs[0,1])
    ax3  = fig.add_subplot(gs[1,0])
    ax4  = fig.add_subplot(gs[1,1])
    ax5  = fig.add_subplot(gs[2,0])
    ax6  = fig.add_subplot(gs[2,1])
    ax7  = fig.add_subplot(gs[3,0])
    ax8  = fig.add_subplot(gs[3,1])
    ax9  = fig.add_subplot(gs[4,0])
    ax10 = fig.add_subplot(gs[4,1])
    # ax_chart(df, ax1, log=False, ema=False, sar=False, superTrend=True, bbands=False, support=False)
    ax_chart(df, ax1, log=False, ema=True, sar=True, superTrend=False, bbands=False)
    # ax2.plot(df['BBANDS_HI']/df['BBANDS_LOW'], label='variance', color='green')

    ax_bband_ratio(df, ax3)
    ax5.plot(df['vol_flow'], label='vol flow')
    ax_alligator_oscillator(df,ax7)
    ax_rsi(df, ax4)
    # ax_macd(df, ax6)
    ax_aroon_oscillator(df,ax8)
    ax_macd_history(df,ax6)
    ax_volume(df,ax9)
    ax_adx(df,ax10)

plot_everything(df.loc[df.Date <= today].tail(60))

# %%
def plot_kissofdeath(df):

    fig = plt.figure()

    gs = gridspec.GridSpec(ncols=1, nrows=3, width_ratios=[1], height_ratios=[3,1,1]) 
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[1,0], sharex=ax1)
    ax3 = fig.add_subplot(gs[2,0], sharex=ax1)


    ax1.plot(df['Close'], label='close', color='black')
    ax1.plot(df['SMA50'], label='sma50', color = 'blue')
    ax1.plot(df['SMA200'], label='sma200', color="crimson")

    # ax_chart(df, ax1, log=True, ema=False, sar=False, superTrend=False, chandelier=False)
    ax_dpo(df,ax2)
    ax_rsi(df, ax3)
    # ax_macd(df, ax3)


# plot_chart(df.loc[df.Date <= today].tail(range_days))
plot_kissofdeath(df_source)

# %%
def plot_priceaction(df):

    fig = plt.figure()

    gs = gridspec.GridSpec(ncols=1, nrows=3, width_ratios=[1], height_ratios=[3,1,1]) 
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[1,0], sharex=ax1)
    ax3 = fig.add_subplot(gs[2,0], sharex=ax1)


    ax_chart(df, ax1, log=False, ema=False, sar=False, superTrend=False)
    # put bullets on supertrend and sar
    # df["temp"] =  df["Close"] - df["SAR"]
    # generate_dots(ax1, df, 'temp', 'Close', start=1, greenDot='gD', redDot='rD', greenLabel = "SAR Up", redLabel="SAR Down", offset = 0.01)
    # df["temp"] = df["Close"] - df["SuperTrend"]
    # generate_dots(ax1, df, 'temp', 'Close', start=1, greenDot='g*', redDot='r*', greenLabel = "STrend Up", redLabel="STrend Down", offset = 0.01)
    # # generate_dots(ax1, df, 'MACD_HISTORY', 'Close', start=1, greenDot='gX', redDot='rX', greenLabel = "MACD up", redLabel="MACD down", offset = 0.01)
    # generate_minmax_dots(ax1,df,"MACD_HISTORY", "Close", period=5, greenDot='gX', redDot='rX', greenLabel = "MACD up", redLabel="MACD down")
    # generate_dots(ax1, df, 'vol_flow', 'Close', start=1, greenDot='g+', redDot='r+', greenLabel = "Vol bull", redLabel="Vol bear", offset = 0.01)
    # generate_dots(ax1, df, 'alligator_oscillator', 'Close', start=1, greenDot='g^', redDot='rv', greenLabel = "All up", redLabel="All down", offset = 0)

    ax1.legend(title = "Price Action", loc = 'upper left')
    ax_volume(df,ax2)

    def ax_atr(df,ax):
        ax.plot(df['ATR'], label='ATR', color='black')
        df['ATR-AVG'] = pd.Series(ta.EMA(df["ATR"], timeperiod=200), index=df.index)
        ax.plot(df['ATR-AVG'], label='ATR-AVG', color='red')
        ax.legend(title = "ATR", loc = 'upper left')

    ax_atr(df,ax3)


plot_priceaction(df.loc[df.Date <= today].tail(range_days))



# %%
# # PROBABILITY
# df_prob = df.loc[df.Date <= today].tail(365*2)
# most_recent = df_prob.tail(1)
# print(most_recent[['Close']])
# period = 2

# Value = 'HMAPriceXOverPerc'
# Price='Close'
# df_source = df_prob[[Price,Value]]
# df_source["pPos"] = None
# df_source["pNeg"] = None
# df_source["pZer"] = None
# m = alib.build_probability_matrix(Value, df_source, Price, period=period, multiplier=1)
# # n,p,z = alib.updown_probability(most_recent.iloc[0][Value], m)
# print(most_recent.iloc[0][Value])
# n,p,z = alib.updown_probability(most_recent.iloc[0][Value], m)
# print(Value)
# print(p,n,z)

# Value = 'MACD_HISTORY'
# Price='Close'
# df_source = df_prob[[Price,Value]]
# df_source["pPos"] = None
# df_source["pNeg"] = None
# df_source["pZer"] = None
# m = alib.build_probability_matrix(Value, df_source, Price, period=period, multiplier=1)
# # n,p,z = alib.updown_probability(most_recent.iloc[0][Value], m)
# print(most_recent.iloc[0][Value])
# n,p,z = alib.updown_probability(most_recent.iloc[0][Value], m)
# print(Value)
# print(p,n,z)

# Value = 'RSI'
# Price='Close'
# df_source = df_prob[[Price,Value]]
# df_source["pPos"] = None
# df_source["pNeg"] = None
# df_source["pZer"] = None
# m = alib.build_probability_matrix(Value, df_source, Price, period=period, multiplier=1)
# # n,p,z = alib.updown_probability(most_recent.iloc[0][Value], m)
# print(most_recent.iloc[0][Value])
# n,p,z = alib.updown_probability(most_recent.iloc[0][Value], m)
# print(Value)
# print(p,n,z)
# %%
