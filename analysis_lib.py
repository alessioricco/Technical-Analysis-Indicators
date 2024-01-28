# import pandas as pd
import numpy as np

# **************************************************************************
# **************************************************************************
def days_ago(df, date, num):
    return df.loc[df.Date <= date].tail(num)

def days_range(df, date_min, date_max):
    return df.loc[date_min <= df.Date].loc[df.Date <= date_max]

# **************************************************************************
# **************************************************************************
def percentage(lo, hi):
    return (hi - lo)/lo

def format_percent(value):
    return "{:.2%}".format(value)

def format_money(value):
    return '{0:.5g}'.format(value)

def format_number(value):
    return '{0:.3g}'.format(value)

def isIncreasing(df):
    # least recent
    least_recent = df.head(1)
    # print(least_recent.iloc[0])
    # most recent
    most_recent = df.tail(1)
    # print(most_recent.iloc[0])
    # return least_recent.iloc[0] < most_recent.iloc[0]

    perc = percentage(least_recent.iloc[0],most_recent.iloc[0])
    # print(least_recent.iloc[0])
    # print(most_recent.iloc[0])
    # print(perc)
    return perc

# **************************************************************************
# **************************************************************************
def slope(df, column,period):
    lw_df = df[[column]].tail(period).copy()
    lw_df.reset_index( drop=True, inplace=True)
    lw_df['x'] = lw_df.index
    x = lw_df['x'].tolist()
    y = lw_df[column].to_list()
    return np.polyfit(x, y, 1)

def slopedirection(df, column, period):
    try:
        s = slope(df, column,period)
        # y = s[0]x + s[1]
        sign = s[0] 
        if sign == 0:
            return 0
        if sign > 0:
            return 1
        return -1
    except:
        return 0

# **************************************************************************
# **************************************************************************
def rule_of_seven_bullish(low, high):
    result1 = ((high - low) * 1.75) + (low)
    result2 = ((high - low) * 2.33) + (low)
    result3 = ((high - low) * 3.5) + (low)
    return (result1, result2, result3)

def rule_of_seven_bearish(low, high):
    result1 = high - ((high - low) * 1.4)
    result2 = high - ((high - low) * 1.75)
    result3 = high - ((high - low) * 2.33)
    return (result1, result2, result3)

# **************************************************************************
# **************************************************************************

def marketFee(spending):
    # if spending <= 10:
    #     return 0.99
    # if spending <= 25:
    #     return 1.49      
    # if spending <= 50:
    #     return 1.99  
    # if spending <= 200:
    #     return 2.99  
    # return spending * 0.0399
    return spending * 0.005
# sellBreakEvenPoint(qty, amount, col) if action=="long" else buyBreakEvenPoint(amount+fee, qty, col)

# -----------------------------------------------   
# 
# ----------------------------------------------- 
def buy(price,moneytoinvest):
    amount = moneytoinvest
    fee = marketFee(amount)
    qty = (amount-fee)/price 
    return qty, fee

# -----------------------------------------------   
# 
# ----------------------------------------------- 
def sell(price,cryptotoinvest):
    amount = cryptotoinvest * price
    fee = marketFee(amount)
    return amount-fee, fee

def sellBreakEvenPoint(cryptotoinvest, amount, price):
    # breakeven point for long positions
    return 1+(amount+2*marketFee(cryptotoinvest * price))/cryptotoinvest

def buyBreakEvenPoint(moneytoinvest, qty):
    return ((moneytoinvest-2*marketFee(moneytoinvest))/qty)-1

def WhatIfBuy(money, price):
    qty, fee = buy(price, money)
    return sellBreakEvenPoint(qty, money, price)

def WhatIfSell(crypto, price):
    amount, fee = sell(price, crypto)
    return buyBreakEvenPoint(amount, crypto)

# -----------------------------------------------   
#  PROBABILITY
# ----------------------------------------------- 

MAXRANGE = 300
HALFRANGE = int(MAXRANGE/2)

def isInRange(r,c):
    # return -HALFRANGE < r and r < HALFRANGE and  -HALFRANGE < c and c < HALFRANGE
    return  0 <= r and r < MAXRANGE and 0 <= c and c < MAXRANGE

def build_probability_matrix(VALUE, df, PRICE = 'Close', multiplier=1, period=7):
    # given the spread between price and hma200 want likely with happens in the next 7 days?

    m = np.zeros( (MAXRANGE, MAXRANGE), dtype=int )
    # period = 7
    # VALUE = "HMAPriceXOverPerc"

    # df_range = df.loc[df.Date <= "2021-01-01"].tail(365*2)
    # df_range.set_index("Date")
    # df_range = df_range[['Close',VALUE]]
    df_range = df[[PRICE,VALUE]]

    for i in range(0, len(df_range)-period):
        # current value of spread in percentage
        c = HALFRANGE + np.int(np.round(df_range[VALUE].iat[i] * multiplier,0))
        
        # current value of spread n days later
        # r = HALFRANGE + np.int(np.round(df_range[VALUE].iat[i+period] * multiplier,0))
        # current increase or drop in price n days later
        perc = np.int(np.round(percentage(df_range[PRICE].iat[i],df_range[PRICE].iat[i+period]) * 100,0))
        # if perc == 14:
        #     print(1)

        r = HALFRANGE + perc
        if np.isnan(c) or np.isnan(r):
            continue
        if isInRange(r,c):
            m[r,c] += np.int(1)
    df_range = None
    return m

def find_probability(current_value, m):

    a = []
    c = HALFRANGE + np.int(np.round(current_value,0))
    if c > MAXRANGE:
        return a

    for r in range(0, len(m[0])):
        if isInRange(r,c): 
            if m[r,c] != 0:
                a.append([(r-HALFRANGE)/100,m[r,c]])
    return a

def updown_probability(current_value, m):
    a = find_probability(current_value, m)
    tot = 0
    for i in range(0,len(a)):
        tot += a[i][1]

    if tot == 0:
        return 0,0,0

    # s = "probability of price direction\n"
    pos = 0
    neg = 0
    for i in range(0,len(a)):
        if a[i][0] < 0:
            neg += a[i][1]
        if a[i][0] > 0:
            pos += a[i][1]  
    return neg/tot, pos/tot, (tot-(pos+neg))/tot


def show_probability(a):
    if len(a) == 0:
        return "no history"
    tot = 0
    for i in range(0,len(a)):
        tot += a[i][1]

    s = "probability of price direction\n"
    pos = 0
    neg = 0
    for i in range(0,len(a)):
        if a[i][0] < 0:
            neg += a[i][1]
        if a[i][0] > 0:
            pos += a[i][1]  
        s += "p(%s) = %s\n" % (format_percent(a[i][0]),format_percent(a[i][1]/tot))
 
    s += "p(%s) = %s\n" % ("<0",format_percent(neg/tot))
    s += "p(%s) = %s\n" % ("=0",format_percent((tot-(pos+neg))/tot))
    s += "p(%s) = %s\n" % (">0",format_percent(pos/tot))

    return s
