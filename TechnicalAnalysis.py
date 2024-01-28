import pandas as pd
# https://github.com/scls19fr/pandas-helper-calc
import numpy as np
# https://github.com/bukosabino/ta
# https://technical-analysis-library-in-python.readthedocs.io/en/latest/index.html
# import ta as tanewlib
# http://mrjbq7.github.io/ta-lib/
# https://cryptotrader.org/talib
import talib as ta
import datetime as dt
# from config import Config
# from utilities import Utilities
# from scipy.signal import argrelextrema
import math


# utilities = Utilities()
VALUE = 'Close'
VOLUME = 'Volume'

class TechnicalAnalysis:

    def __init__(self):
        return

# **************************************************************************
# **************************************************************************
    def golden_cross(self,df):
        if df['SMA50'] > df['SMA200']:
            # golden cross is meaningful only if both lines are ascending
            return 1 
        if df['SMA200'] > df['SMA50']:
            # kiss of death is valid only if both lines are declining
            return -1
        return 0

# **************************************************************************
# **************************************************************************
    def ema100_buy_signal(self,df):
        if df['High'] >= df['EMA100'] and df["EMA100"] >= df['Low']:
            return 1 
        return 0

# **************************************************************************
# **************************************************************************
    def uptrend(self,df):
        if df[VALUE] > df['MA50'] and df['MA50'] > df['MA200']:
            return 1 
        if df['MA200'] > df['MA50'] and df['MA50'] > df[VALUE]:
            return -1 
        return 0

# # **************************************************************************
# # **************************************************************************
#     def macd_buy_signal(self,df):
#         if df['MACD'] >= df['MACD_SIGNAL']:
#             return 1 
#         if df['MACD'] <= df['MACD_SIGNAL']:
#             return -1 
#         return 0

# # **************************************************************************
# # **************************************************************************
#     def macd_uptrend(self, df):
#         if df['MACD_HISTORY'] > 0:
#             return 1
#         if df['MACD_HISTORY'] > 0:
#             return 1
#         return 0    

# **************************************************************************
# **************************************************************************
    def macd_strategy(self, df):
        # # # buy
        # if df['MACD'] >= df['MACD_SIGNAL'] and np.sign(df['MACD']) < 0 and np.sign(df['MACD_SIGNAL']) < 0 and df["EMA200"] < df[VALUE]:
        #     return 1
        # # sell 
        # if df['MACD'] <= df['MACD_SIGNAL'] and np.sign(df['MACD']) > 0 and np.sign(df['MACD_SIGNAL']) > 0 and df["EMA200"] > df[VALUE]:
        #     return -1 
        # return 0 
        # buy
        if df['MACD'] >= df['MACD_SIGNAL'] and np.sign(df['MACD']) < 0 :
            return 1
        # sell 
        if df['MACD'] <= df['MACD_SIGNAL'] and np.sign(df['MACD']) > 0 :
            return -1 
        return 0 

# **************************************************************************
# **************************************************************************
    def uptrend2(self,df):
        if df[VALUE] > df['EMA21'] and df['EMA21'] > df['EMA55'] and df['EMA55'] > df['EMA100'] and df['EMA100'] > df['SMA200']:
            return 1
        if df[VALUE] < df['EMA21'] and df['EMA21'] < df['EMA55'] and df['EMA55'] < df['EMA100'] and df['EMA100'] < df['SMA200']:
            return -1
        return 0

# **************************************************************************
# **************************************************************************
    def obv(self,df):
        obv = ta.OBV(df[VALUE],df[VOLUME])
        return obv



# **************************************************************************
# **************************************************************************
    def low_high_perc(self,df):
        return 0 if df['High'] == 0 else 1 - (df['Low'] / df['High']) 

# **************************************************************************
# **************************************************************************
    def hma_uptrend(self,df):
        # =IFS(AND(AB17>0,AB18<0),"down",AND(AB17<0,AB18>0),"up",TRUE(),"")

        if df["HMA200"] > df[VALUE]:
            return -1
        if df["HMA200"] < df[VALUE]:
            return 1
        return 0 

    def ema200_uptrend(self,df):
        # =IFS(AND(AB17>0,AB18<0),"down",AND(AB17<0,AB18>0),"up",TRUE(),"")

        if df["EMA200"] > df[VALUE]:
            return -1
        if df["EMA200"] < df[VALUE]:
            return 1
        return 0 

# **************************************************************************
# **************************************************************************
    # examine score and previous values and
    # offer suggestions using the derivative
    # to evaluate if we are at the end of uptrend
    # and downtrend 
    def null(self,df):
        return " "

# **************************************************************************
# **************************************************************************
    def priceDistance(self,df):
        return 100 * self.normalise(df, VALUE, "EMA100")
        # return 100 * (df[VALUE] - df['EMA100'])/df['EMA100']

# **************************************************************************
# **************************************************************************
    def HMAPriceXOverPerc(self,df):
        return 100 * self.normalise(df, VALUE, "SMA200")
        # return 100 * (df[VALUE] - df['HMA200'])/df['HMA200']

# **************************************************************************
# **************************************************************************
    # def SMAPriceXOverPerc(self,df):
    #     return (df[VALUE] - df['SMA200'])/df['SMA200']

    # def long200(self,df):
    #     hma200Perc = (df[VALUE] - df['HMA200'])/df['HMA200']
    #     # sma200Perc = (df[VALUE] - df['SMA200'])/df['SMA200']
    #     return (hma200Perc)

    def normalise(self,df, price, indicator):            
        return (df[price]/df[indicator]) - 1

# **************************************************************************
# **************************************************************************
    def sign(self,x):
        return 1 if x > 0 else -1 if x<0 else 0

# **************************************************************************
# **************************************************************************
    def suggestion(self,df):
        def sign(x):
            return 1 if x > 0 else -1 if x<0 else 0

        if sign(df['HMAPriceXOverPerc']) > sign(df['HMAPriceXOverPercPrev']):
            return "xover uptrend"
        if sign(df['HMAPriceXOverPerc']) < sign(df['HMAPriceXOverPercPrev']):
            return "xover downtrend"
        if df['HMAPriceXOverPerc'] > 20:
            return "hot"
        if df['HMAPriceXOverPerc'] < -20:
            return "cold"
        return " "

# **************************************************************************
# **************************************************************************
    def aroon_helper(self, df):
        # =ifs(AND(AD1055>80,AE1055<20),"Bear",AND(AD1055<50,AE1055<50),"consolidate",AND(AE1055>80,AD1055<20),"Bull",AND(AD1055<AE1055),"up",AND(AD1055>AE1055),"down",True(),"")
        up = df["AROON_UP"]
        down = df["AROON_DOWN"]
        if up>80 and down<20:
            return "Bull"
        if up>95 and down<5:
            return "Strong Bull (-)"
        if up<50 and down<50:
            return "consolidate"
        if down>80 and up<20:
            return "Bear"
        if down>95 and up<5:
            return "Strong Bear (+)"
        if up>down:
            return "up"
        if up<down:
            return "down"
        return ""

# **************************************************************************
# **************************************************************************
    def chopiness(self, df, window=14):
        atrsum = df['ATR'].rolling(window).sum()
        highs = df['High'].rolling(window).max()
        lows = df['Low'].rolling(window).min()
        return 100 * np.log10(atrsum / (highs - lows)) / np.log10(window)

# **************************************************************************
# **************************************************************************
    def bollinger(self, df):
        # df['BBANDS_HI'],df['BBANDS_MID'],df['BBANDS_LOW']
        if df[VALUE] > df['BBANDS_HI']:
            return "above hi"
        if df[VALUE] < df['BBANDS_LOW']:
            return "below lo"
        if df[VALUE] > df['BBANDS_MID']:
            return "above mid"
        if df[VALUE] < df['BBANDS_MID']:
            return "below mid"
        return " "

# **************************************************************************
# **************************************************************************
    def bollinger_squeze(self, df):
        return (df['BBANDS_HI']-df['BBANDS_LOW'])/df[VALUE]

# **************************************************************************
# **************************************************************************
    def bollinger_osc(self, df):
        diff = (df['BBANDS_HI']-df['BBANDS_LOW'])
        if diff == 0:
            return None
        # return 200*(1-((df['BBANDS_HI']-df[VALUE])/diff))-100
        return ( (df[VALUE]-df['BBANDS_LOW'])/diff )

# **************************************************************************
# **************************************************************************
    def chop_label(self, df):
        if df["CHOP"] > 61.8:
            return "choppy"
        if df["CHOP"] < 38.2:
            return "trendy"
        return " "

# **************************************************************************
# **************************************************************************
    def chop_signal(self, df):
        # range
        if df["CHOP"] > 61.8:
            return -1
        #  trending
        if df["CHOP"] < 38.2:
            return 1
        return 0

# **************************************************************************
# **************************************************************************
    def pumpdump(self, df):
        percent_delta = 0.15
        change = (df['High'] - df['Low'])
        if (df['Close'] > df['Open']) and (change / df['Low']) > percent_delta:
            return 1
        if (df['Close'] < df['Open']) and (change / df['Low']) > percent_delta:
            return -1
        return 0

# **************************************************************************
# **************************************************************************
    # def resistanceAndSupport(self, df, row, support=True):
    #     volume = VOLUME
    #     grouped_multiple = df[df['Date'] <= row["Date"]].groupby(['PriceLevel']).agg({volume: ['sum'],'PriceLevel': ['count']})
    #     grouped_multiple.columns = ['volume', 'count']
    #     # grouped_multiple.to_csv(source + '.supports.csv')
        
    #     # 
    #     count_mean = grouped_multiple["count"].mean() 
    #     grouped_multiple = grouped_multiple[grouped_multiple['count'] > count_mean]
    #     grouped_multiple['PriceLevel'] = grouped_multiple.index
    #     # grouped_multiple = grouped_multiple.loc[grouped_multiple["count"] > 1]

    #     price = row[VALUE]
    #     if support:
    #         return grouped_multiple[grouped_multiple['PriceLevel'] < price][['PriceLevel']].max()['PriceLevel']

    #     return grouped_multiple[grouped_multiple['PriceLevel'] > price][['PriceLevel']].min()['PriceLevel']
    #     # return support['PriceLevel']

# **************************************************************************
# **************************************************************************
    def averagevolumes(self, df, col, upvalues):
        period = 14
        open,close,volume = "Open", "Close", VOLUME
        set= df[[open,close,volume]].loc[df.Date <= col["Date"]].tail(period)
        if len(set) < 14:
            return None
        # set = df.tail(14)[[open,close,volume]]
        set_volume = set[volume]

        set_up = set_volume.loc[ set[close]>set[open]]
        cumulative_up = set_up.sum()
        count_up = set_up.count()
        averagevolume_up = (cumulative_up / count_up)

        set_down = set_volume.loc[ set[close]<set[open]]
        cumulative_down = set_down.sum()
        count_down = set_down.count()        
        averagevolume_down = -(cumulative_down / count_down) 
        if upvalues:
            return averagevolume_up
        else:
            return averagevolume_down

# **************************************************************************
# **************************************************************************
    def vol_set(self, df, col, upvalues):
        period = 14
        open,close,volume = "Open", "Close", VOLUME
        # set= df[[open,close,volume]].loc[df.Date <= col["Date"]].tail(period)
        set= df.loc[df.Date <= col["Date"]].tail(period)
        if len(set) < 14:
            return pd.DataFrame(columns = set.columns)[volume]
        # set = df.tail(14)[[open,close,volume]]
        # set_volume = set[volume]

        if upvalues:
            return set[volume].loc[ set[close]>set[open]]
        else:
            return set[volume].loc[ set[close]<set[open]]

# **************************************************************************
# **************************************************************************
    def volplot(self,row):

        if (not np.isnan(row["vol_avgup"])) and row["Close"]>row["Open"]:
            if row["vol_avgup"]<=row[VOLUME]:
                    # signal up
                    return 2
            if row["vol_avgdown"]>= row[VOLUME]:
                # regression up
                return 1
        if  (not np.isnan(row["vol_avgdown"])) and row["Close"]<row["Open"]:
            if  row["vol_avgdown"]>= -row[VOLUME]:
                    # signal down
                    return -2
            if row["vol_avgdown"]<= -row[VOLUME]:
                # regression down
                return -1
        return 0    

# **************************************************************************
# **************************************************************************
    def vol_localtop(self, row):
        if (not np.isnan(row["vol_avgup"])) and row["Close"]>row["Open"]:
            # signal up
            if row["vol_avgup"]<=row[VOLUME]:
                if row["vol_flow"] < 0 and row["vol_flowPrev"] < 0:
                    return 1

        if  (not np.isnan(row["vol_avgdown"])) and row["Close"]<row["Open"]:
            # signal down
            if  row["vol_avgdown"]>= -row[VOLUME]:
                if row["vol_flow"] > 0 and row["vol_flowPrev"] > 0:
                    return -1

        return 0    

# **************************************************************************
# **************************************************************************
    def volrank(self,df):
        # # barc = ebc ? ((flow > 0 and not sigdown) ? lime : (flow < 0 and not sigup) ? red : purgatory ? silver : purgatory1 ? silver : silver) : na
        # purgatory = (flow > 0 and flow[1] > 0) and sigdown
        # purgatory1 = (flow < 0 and flow[1] < 0) and sigup
        if df["vol_flow"] < 0 and not df["sig_up"]:
            return -1
        if df["vol_flow"] > 0 and not df["sig_down"]:
            return 1
        return 0   

# **************************************************************************
# **************************************************************************
    def SupportResistance(self, low, hi):
        try:
            l = min(low,hi)
            h = max(low,hi)
            filter = self.support[l <= self.support['Close']]
            filter = filter[self.support['Close'] <= h]
            # utilities.log(filter)
            # filter = filter[self.support['Support'] == 1]
            value_support = filter['Close'].iloc[0]
            value_resistance = filter['Close'].iloc[len(filter)-1]
            return value_support, value_resistance
        except:
            return -1, -1

# **************************************************************************
# **************************************************************************
    def getSupport(self, value):
        try:
            filter = self.support[value > self.support['Close']]
            # print(filter)
            return filter['Close'].iloc[len(filter)-1]
        except:
            return -1
        
# **************************************************************************
# **************************************************************************
    def getResistance(self, value):
        try:
            filter = self.support[value < self.support['Close']]
            # print(filter)
            return filter['Close'].iloc[0]
        except:
            return -1
    
# **************************************************************************
# **************************************************************************
    def getBestSupport(self, value):
        try:
            filter = self.support[value > self.support['Close']]
            filter = filter[self.support['Support'] == 1]
            # print(filter)
            return filter['Close'].iloc[len(filter)-1]
        except:
            return -1

# **************************************************************************
# **************************************************************************
    def getBestResistance(self, value):
        try:
            filter = self.support[value < self.support['Close']]
            filter = filter[self.support['Support'] == 1]
            # print(filter)
            return filter['Close'].iloc[0]
        except:
            return -1

# **************************************************************************
# **************************************************************************
    def getStrongestSupport(self, value):
        try:
            filter = self.support[value > self.support['Close']].tail()
            filter = filter[filter['Volume']==filter['Volume'].max()]
            # print(filter)
            return filter['Close'].iloc[len(filter)-1]
        except:
            return -1

# **************************************************************************
# **************************************************************************
    def getStrongestResistance(self, value):
        try:
            filter = self.support[value < self.support['Close']].head()
            filter = filter[filter['Volume']==filter['Volume'].max()]
            # print(filter)
            return filter['Close'].iloc[0]
        except:
            return -1
# **************************************************************************
# **************************************************************************
    def Supertrend(self,df,multiplier,period): #df is the dataframe, n is the period, f is the factor; f=3, n=7 are commonly used.

        df['ATR'] = pd.Series(ta.ATR(df["High"],df["Low"],df["Close"], timeperiod=period), index=df.index)
        #Calculation of SuperTrend
        
        df['basic_ub']=(df['High']+df['Low'])/2 + (multiplier*df['ATR'])
        df['basic_lb']=(df['High']+df['Low'])/2 - (multiplier*df['ATR'])

        # Compute final upper and lower bands
        df['final_ub'] = 0.00
        df['final_lb'] = 0.00
        for i in range(period, len(df)):
            df['final_ub'].iat[i] = df['basic_ub'].iat[i] if df['basic_ub'].iat[i] < df['final_ub'].iat[i - 1] or df["Close"].iat[i - 1] > df['final_ub'].iat[i - 1] else df['final_ub'].iat[i - 1]
            df['final_lb'].iat[i] = df['basic_lb'].iat[i] if df['basic_lb'].iat[i] > df['final_lb'].iat[i - 1] or df["Close"].iat[i - 1] < df['final_lb'].iat[i - 1] else df['final_lb'].iat[i - 1]
        
        # Set the Supertrend value
        st = "SuperTrend"
        df[st] = 0.00
        for i in range(period, len(df)):
            df[st].iat[i] = df['final_ub'].iat[i] if df[st].iat[i - 1] == df['final_ub'].iat[i - 1] and df["Close"].iat[i] <= df['final_ub'].iat[i] else \
                            df['final_lb'].iat[i] if df[st].iat[i - 1] == df['final_ub'].iat[i - 1] and df["Close"].iat[i] >  df['final_ub'].iat[i] else \
                            df['final_lb'].iat[i] if df[st].iat[i - 1] == df['final_lb'].iat[i - 1] and df["Close"].iat[i] >= df['final_lb'].iat[i] else \
                            df['final_ub'].iat[i] if df[st].iat[i - 1] == df['final_lb'].iat[i - 1] and df["Close"].iat[i] <  df['final_lb'].iat[i] else 0.00 
                    
        # Mark the trend direction up/down
        stx = "SuperTrend_Indicator"
        df[stx] = np.where((df[st] > 0.00), np.where((df["Close"] < df[st]), -1,  1), 0)


        # Remove basic and final bands from the columns
        df.drop(['ATR','basic_ub', 'basic_lb', 'final_ub', 'final_lb'], inplace=True, axis=1)
        
        df.fillna(0, inplace=True)

        return df

# **************************************************************************
# **************************************************************************

    def SuperTrend_Strategy(self,row):
        stx = "SuperTrend_Indicator"
        if math.isnan(row['uptrend_HMA200']):
            return 0
        if row["CHOP_SIGNAL"] != 1:
            return 0
        if row[stx] == 1 and row['uptrend_HMA200'] == 1:
            return row[stx]
        if row[stx] == -1 and row['uptrend_HMA200'] == -1:
            return row[stx]
        
        return 0

# **************************************************************************
# **************************************************************************

    def trend_strength(self,df):
        # measures the trend 
        # > 0 trending
        # < 0 ranging
        # df["ari_trend"] = 0
        # df["ari_range"] = 0

        trend = 0
        sideways = 0
        uncertain = 0
        if df['CHOP_SIGNAL'] == 1:
            trend += 1
        if df['CHOP_SIGNAL'] == -1:
            sideways +=1
        if df['CHOP_SIGNAL'] == 0:
            uncertain +=1

        up = df["AROON_UP"]
        down = df["AROON_DOWN"]

        if np.abs(up-down) < 25:
            sideways +=1
        if np.abs(up-down) > 60:
            trend += 1

        if df['ADX'] > 25:
            trend += 1
        else:
            sideways +=1
        return trend-sideways-uncertain

    def trend_direction(self,df):
        up = 0
        down = 0
        uncertain = 0

        aup = df["AROON_UP"]
        adown = df["AROON_DOWN"]

        if aup>80 and adown<20:
            up +=1
        elif adown>80 and aup<20:
            down += 1
        else:
            uncertain +=1

        if df["HMA200"] < df["Close"]:
            up +=1
        elif df["HMA200"] > df["Close"]:
            down +=1
        else:
            uncertain +=1

        # if df["uptrend_EMA"] > 0:
        #     up +=1
        # elif df["uptrend_EMA"] < 0:
        #     down +=1
        # else:
        #     uncertain +=1

        if df["TREND"] > 0:
            up +=1
        elif df["TREND"] < 0:
            down +=1
        else:
            uncertain +=1

        # if df["RSI"] > 80:
        #     up +=1
        # elif df["RSI"] < 20:
        #     down +=1
        # else:
        #     uncertain +=1

        # if df['uptrend_MACD'] > 0:
        #     up += 1
        # elif df["uptrend_MACD"] < 0:
        #     down +=1
        # else:
        #     uncertain +=1

        return up - down - uncertain

# **************************************************************************
# **************************************************************************
    def createARStrategy(self, df, strategy_name = "strategy", uptrend_name = "uptrend_HMA200", strong_trend_name = "CHOP_SIGNAL"):
        df[strategy_name] = 0
        df[strategy_name + "_helper"] = ""
        for i in range(1, len(df)):

            uptrend = np.sign(df[uptrend_name].iat[i])
            strong_trend = np.sign(df[strong_trend_name].iat[i])

            sar_buy = (df["SAR"].iat[i] < df["Close"].iat[i])

            score = 0    
            if uptrend==1:
                # uptrend, 
                # then if strategy indicator was at 0 and now the trend is strong
                # or the strategy indicator was already above 0 then
                if (df[strategy_name].iat[i-1] <= 0 and strong_trend == 1) or df[strategy_name].iat[i-1] > 0:
                    if df["SuperTrend_Indicator"].iat[i] == 1:
                        score += 1
                    if sar_buy:
                        score += 1

            elif uptrend==-1:
                # downtrend,
                # then if the strategy indicator was at 0 and now the trend is strong
                # or the strategy indicator was already below 0
                if (df[strategy_name].iat[i-1] >= 0 and strong_trend == 1) or df[strategy_name].iat[i-1]<0:
                    if df["SuperTrend_Indicator"].iat[i] == -1:
                        score -= 1
                    if not sar_buy:
                        score -= 1

            df[strategy_name].iat[i] = score 

        # for i in range(1, len(df)):

            if df[strategy_name].iat[i] > 0 and df[strategy_name].iat[i-1] <= 0 :            
                df[strategy_name + "_helper"].iat[i] = "buy"

            elif df[strategy_name].iat[i] < 0 and df[strategy_name].iat[i-1] >= 0 :            
                df[strategy_name + "_helper"].iat[i] = "sell"

            elif df[strategy_name].iat[i] == 0 and df[strategy_name].iat[i-1] > 0 :            
                df[strategy_name + "_helper"].iat[i] = "end long"

            elif df[strategy_name].iat[i] == 0 and df[strategy_name].iat[i-1] < 0 :            
                df[strategy_name + "_helper"].iat[i] = "end short"

            elif (df["TREND"].iat[i]<-4 and df["TREND_BULL"].iat[i]<0.25):
                df[strategy_name + "_helper"].iat[i] = "short tp"
                # df[strategy_name].iat[i] = - 0.5

            elif (df["TREND"].iat[i]>4 and df["TREND_BEAR"].iat[i]<0.25):
                df[strategy_name + "_helper"].iat[i] = "long tp"
                # df[strategy_name].iat[i] = 0.5

            elif np.sign(df["SuperTrend_Indicator"].iat[i-1]) != np.sign(df["SuperTrend_Indicator"].iat[i]):
                if df["SuperTrend_Indicator"].iat[i] == 1:
                    df[strategy_name + "_helper"].iat[i] = "start buy"
                elif df["SuperTrend_Indicator"].iat[i] == -1:
                    df[strategy_name + "_helper"].iat[i] = "start sell"

            elif np.abs(df["HMAPriceXOverPerc"].iat[i]) < 5:
                if (df["SAR"].iat[i] < df["Close"].iat[i]) != (df["SAR"].iat[i-1] < df["Close"].iat[i-1]):
                    if (df["SAR"].iat[i] < df["Close"].iat[i]):
                        df[strategy_name + "_helper"].iat[i] = "weak buy"
                elif (df["SAR"].iat[i] > df["Close"].iat[i]) != (df["SAR"].iat[i-1] > df["Close"].iat[i-1]):
                    if (df["SAR"].iat[i] > df["Close"].iat[i]):
                        df[strategy_name + "_helper"].iat[i] = "weak sell"

        return df

# **************************************************************************
# **************************************************************************
    def SMMA(self, df, src, period, adjust=True):
        # return src.ewm(alpha = 1/period, adjust=adjust).mean()
        sma = pd.Series(ta.SMA(df[src], timeperiod=period), index=df.index)
        smma = sma
        for i in range(period, len(df)):
            smma.iat[i] = (smma.iat[i-1] * (period-1) + df[src].iat[i])/period
        # sma(src, length) : (smma[1] * (length - 1) + src) / length
        return smma

# **************************************************************************
# **************************************************************************
    def alligator_strategy(self, row):
        if row["jaw"]>row["teeth"] and row["teeth"] > row["lips"]:
            return -(row["jaw"]-row["lips"])/row["lips"]
                
        if row["jaw"]<row["teeth"] and row["teeth"] < row["lips"]:
            return (row["lips"]-row["jaw"])/row["jaw"]
                
        return 0


# **************************************************************************
# **************************************************************************
    def HMA(self,df, VALUE, timeperiod=200):

        def WMAtmp(self,df):
            # needed for calculation of HMA
            return  2*df['WMA100']-df['WMA200']

        try:
            WMA100 = pd.Series(ta.WMA(df[VALUE], timeperiod=np.int(timeperiod/2)), index=df.index)
            WMA200 = pd.Series(ta.WMA(df[VALUE], timeperiod=timeperiod), index=df.index)
            WMAtmp = 2*WMA100-WMA200
            # WMAtmp = df.apply (lambda row: self.WMAtmp(row), axis=1)
            return pd.Series(ta.WMA(WMAtmp, timeperiod=14), index=df.index)
            # df['uptrend_HMA200Prev'] = df['uptrend_HMA200'].shift(1)
            # df = df.drop(columns=['WMA100','WMA200','WMAtmp'])
        except:
            return math.nan




# **************************************************************************
# **************************************************************************
    def ApplyTechnicalAnalysis(self, df, list=None):
        
        def inList(value, list):
            if list is None:
                return True
            else:
                return value in list


        # def minmax(df,order,column):
        #     # import numpy as np
        #     # https://stackoverflow.com/questions/31070563/find-all-local-maxima-and-minima-when-x-and-y-values-are-given-as-numpy-arrays
        #     from scipy.signal import argrelextrema
        #     # import matplotlib.pyplot as plt

        #     x = np.array(df["Date"].values)
        #     y = np.array(df[VALUE].values)

        #     # sort the data in x and rearrange y accordingly
        #     sortId = np.argsort(x)
        #     x = x[sortId]
        #     y = y[sortId]

        #     # this way the x-axis corresponds to the index of x
        #     maxm = argrelextrema(y, np.greater, order=order)  # (array([1, 3, 6]),)
        #     minm = argrelextrema(y, np.less, order=order)  # (array([2, 5, 7]),)
        #     for elem in maxm[0]:
        #         df.iloc[elem, df.columns.get_loc(column)] = 'max'
        #     for elem in minm[0]:
        #         df.iloc[elem, df.columns.get_loc(column)] = 'min'

        
        # https://uk.tradingview.com/script/MoRmwV5U-Pump-Doctor-Trends/
        if inList("vol_flow", list):
            df["vol_avgsumup"] = df.apply (lambda row: self.vol_set(df,row, True).sum(), axis=1)
            df["vol_avgsumdown"] = df.apply (lambda row: self.vol_set(df,row, False).sum(), axis=1)
            df["vol_avgcountup"] = df.apply (lambda row: self.vol_set(df,row, True).count(), axis=1)
            df["vol_avgcountdown"] = df.apply (lambda row: self.vol_set(df,row, False).count(), axis=1)
            df["vol_avgup"] = df["vol_avgsumup"]/df["vol_avgcountup"]
            df["vol_avgdown"] = -df["vol_avgsumdown"]/df["vol_avgcountdown"]
            df["vol_flow"] = df["vol_avgsumup"]-df["vol_avgsumdown"]
            df["vol_flowPrev"] = df["vol_flow"].shift(1)
            # df["vol_flowDer"] = df["vol_flow"].calc.derivative()
            # df["vol_signalshort"] = df.apply (lambda row: self.volplot(row), axis=1)
            # df["vol_signallong"] = df.apply (lambda row: 1 if row["vol_flow"] > 0 else -1, axis=1)
            df["vol_signaltopbottom"] = df.apply (lambda row: self.vol_localtop(row), axis=1)
            df = df.drop(columns=['vol_avgsumup','vol_avgsumdown','vol_avgcountup','vol_avgcountdown','vol_avgup','vol_avgdown', "vol_flowPrev"])      
        
        if inList("TREND", list):
            df['ATR5'] = pd.Series(ta.ATR(df["High"],df["Low"],df["Close"], timeperiod=5), index=df.index)
            df['LOWEST50'] = df['Low'].rolling(50).min()
            df['HIGHEST50'] = df['High'].rolling(50).max()
            df["TREND_BULL"] = df.apply (lambda row: (row["Close"] - row["LOWEST50"])/row["ATR5"], axis=1)
            df["TREND_BEAR"] = df.apply (lambda row: (row["HIGHEST50"]-row["Close"])/row["ATR5"], axis=1)
            # df["TREND_BEAR2"] = -df["TREND_BEAR"]
            df["TREND"] = df["TREND_BULL"]-df["TREND_BEAR"]
            df = df.drop(columns=['ATR5','LOWEST50','HIGHEST50'])

        # VOLUME EMA 21
        if inList("EMA21Volume", list):
            df['EMA21Volume'] = pd.Series(ta.EMA(df[VOLUME], timeperiod=21), index=df.index)

        # ON BALANCE VOLUME INDICATOR
        if inList("OBV", list):
            df['OBV'] = pd.Series(ta.OBV(df[VALUE],df[VOLUME]) , index=df.index)

        # ACCUMULATION/DISTRIBUTION INDICATOR
        if inList("AD", list):
            df['AD'] = pd.Series(ta.AD(df['High'], df['Low'], df['Close'], df[VOLUME]), index=df.index)

        if inList("ADOSC", list):
            df['ADOSC'] = ta.ADOSC(df["High"],df["Low"], df["Close"], df[VOLUME])

        # Price Rate of Change Indicator
        if inList("ROC", list):
            df['ROC'] = pd.Series(ta.ROCR100(df[VALUE], timeperiod=7), index=df.index)

        # relative strength index
        if inList("RSI", list):
            df['RSI'] = pd.Series(ta.RSI(df[VALUE],14), index=df.index)

        # AVERAGES
        df['MA50'] = pd.Series(ta.MA(df[VALUE], timeperiod=50, matype=0), index=df.index)
        df['MA200'] = pd.Series(ta.MA(df[VALUE], timeperiod=200, matype=0), index=df.index)
        df['EMA21'] = pd.Series(ta.EMA(df[VALUE], timeperiod=21), index=df.index)
        df['EMA55'] = pd.Series(ta.EMA(df[VALUE], timeperiod=55), index=df.index)
        df['EMA100'] = pd.Series(ta.EMA(df[VALUE], timeperiod=100), index=df.index)
        df['SMA100'] = pd.Series(ta.SMA(df[VALUE], timeperiod=100), index=df.index)
        df['EMA200'] = pd.Series(ta.EMA(df[VALUE], timeperiod=100), index=df.index)
        df['SMA50'] = pd.Series(ta.SMA(df[VALUE], timeperiod=50), index=df.index)
        df['SMA200'] = pd.Series(ta.SMA(df[VALUE], timeperiod=200), index=df.index)
        df['SMA20'] = pd.Series(ta.SMA(df[VALUE], timeperiod=20), index=df.index)

        dfw = df[["Close"]].apply(lambda x: x.resample('7D').mean())
        dfw['SMA50W'] = pd.Series(ta.SMA(dfw['Close'], timeperiod=50), index=df.index)
        dfw['SMA200W'] = pd.Series(ta.SMA(dfw['Close'], timeperiod=200), index=df.index)
        dfw['SMA20W'] = pd.Series(ta.SMA(dfw['Close'], timeperiod=20), index=df.index)
        df["CloseW"] = dfw["Close"]
        df["SMA50W"] = dfw["SMA50W"]
        df["SMA200W"] = dfw["SMA200W"]
        df["SMA20W"] = dfw["SMA20W"]

#     df['week_avg'] = df[["Close"]].apply(lambda x: x.resample('7D').mean())
#     df['week_avg'] = df["week_avg"].fillna(method='ffill')
#     df['avg_perc'] = 100*(df["Close"]-df["week_avg"])/df["week_avg"]

        # df['CloseW'] = df[[VALUE]].apply(lambda x: x.resample('7D').mean())
        # df["CloseW"] = df["CloseW"].fillna(method='ffill')
        # # # df['SMA20W'] = df[["SMA20"]].apply(lambda x: x.resample('7D').mean()).fillna(method='ffill')
        # # # df['SMA50W'] = df[["SMA50"]].apply(lambda x: x.resample('7D').mean()).fillna(method='ffill')
        # # # df['SMA200W'] = df[["SMA200"]].apply(lambda x: x.resample('7D').mean()).fillna(method='ffill')
        # df['SMA50W'] = pd.Series(ta.SMA(df['CloseW'], timeperiod=50), index=df.index)
        # df['SMA200W'] = pd.Series(ta.SMA(df['CloseW'], timeperiod=200), index=df.index)
        # df['SMA20W'] = pd.Series(ta.SMA(df['CloseW'], timeperiod=20), index=df.index)


        # ACCUMULATION/DISTRIBUTION INDICATOR
        if inList("PI", list):
            df['MA111'] = pd.Series(ta.MA(df[VALUE], timeperiod=111, matype=0), index=df.index)
            df['MA350x2'] = 2 * pd.Series(ta.MA(df[VALUE], timeperiod=350, matype=0), index=df.index)

        # df['wClose'] = pd.Series(ta.WCLPRICE(df["High"], df["Low"],df[VALUE]), index=df.index)
        df['HL2'] = (df["High"]+df["Low"])/2

        # DPO
        if inList("DPO", list):
            dpo_period = 21
            dpo_value = "Close"
            df["DPO_SMA"] = pd.Series(ta.SMA(df[dpo_value], timeperiod=dpo_period), index=df.index)
            df["DPO"] = math.nan
            dpo_backperiod = np.int(np.round(dpo_period/2 + 1,0))
            for i in range(dpo_period, len(df)):
                # df["DPO"].iat[i] = df[dpo_value].iat[i-dpo_backperiod] - df["DPO_SMA"].iat[i]
                df["DPO"].iat[i] = df[dpo_value].iat[i] - df["DPO_SMA"].iat[i-dpo_backperiod]

            df = df.drop(columns=['DPO_SMA'])

        # ALLIGATOR
        if inList("ALLIGATOR", list):
            alligator_value = "HL2"
            # df['jaw'] = SMMA(df[alligator_value],13).shift(8)
            # df['teeth'] = SMMA(df[alligator_value],8).shift(5)
            # df['lips'] = SMMA(df[alligator_value],5).shift(3)
            df['jaw'] = self.SMMA(df,alligator_value,13)
            df['teeth'] = self.SMMA(df,alligator_value,8)
            df['lips'] = self.SMMA(df,alligator_value,5)
            df['alligator_oscillator'] = df.apply (lambda row: self.alligator_strategy(row), axis=1)

        if inList("signal_golden_cross", list):
            df['signal_golden_cross'] = df.apply (lambda row: self.golden_cross(row), axis=1)

        if inList("signal_ema100", list):
            df['signal_ema100'] = df.apply (lambda row: self.ema100_buy_signal(row), axis=1)

        if inList("uptrend_MA50MA200", list) or inList("uptrend_MA50MA200Prev", list):
            df['uptrend_MA50MA200'] = df.apply (lambda row: self.uptrend(row), axis=1)

        if inList("uptrend_MA50MA200Prev", list):
            df['uptrend_MA50MA200Prev'] = df['uptrend_MA50MA200'].shift(1)

        if inList("uptrend_EMA", list) or inList("uptrend_EMAPrev", list):
            df['uptrend_EMA'] = df.apply (lambda row: self.uptrend2(row), axis=1)
            df['uptrend_EMA200'] = df.apply (lambda row: self.ema200_uptrend(row), axis=1)

        if inList("uptrend_EMAPrev", list):
            df['uptrend_EMAPrev'] = df['uptrend_EMA'].shift(1)

        if inList("HMA200", list) or inList("HMAPriceXOverPerc", list) or inList("HMAPriceXOverPercPrev", list):
            # df['SMA50Der'] = df['SMA50'].calc.derivative()
            # df['SMA200Der'] = df['SMA200'].calc.derivative()
            df['HMA200'] = self.HMA(df,VALUE,200)
            df['uptrend_HMA200'] = df.apply (lambda row: self.hma_uptrend(row), axis=1)

        # percent distance from HMA200
        if inList("HMAPriceXOverPerc", list) or inList("HMAPriceXOverPercPrev", list):
            df['HMAPriceXOverPerc'] = df.apply (lambda row: self.HMAPriceXOverPerc(row), axis=1)
            # df["SMAPriceXOverPerc"] = df.apply (lambda row: self.SMAPriceXOverPerc(row), axis=1)

        if inList("NORMALIZE", list):
            # normalized price using hma200 as baseline
            df['HMA200-N'] = df.apply (lambda row: self.normalise(row,VALUE,"HMA200"), axis=1)
            df['HMA200-N-SMA200'] = pd.Series(ta.SMA(df['HMA200-N'], timeperiod=200), index=df.index)

            # normalised price using sma200 as baseline 
            df['SMA200-N'] = df.apply (lambda row: self.normalise(row,VALUE,"SMA200"), axis=1)
            df['SMA200-N-HMA200'] = self.HMA(df,"SMA200-N", 200)

            df['EMA21-N'] = df.apply (lambda row: self.normalise(row,VALUE,"EMA21"), axis=1)
            df['EMA55-N'] = df.apply (lambda row: self.normalise(row,VALUE,"EMA55"), axis=1)

            if inList("NORMALIZE", list):
                # moving average using DPO as price
                df['DPO-HMA200'] = self.HMA(df,"DPO", 200)
                df['DPO-SMA200'] = pd.Series(ta.SMA(df["DPO"], timeperiod=200), index=df.index)

                # normalized DPO using its sma200 as baseline
                df['DPO-HMA200-N'] = df.apply (lambda row: self.normalise(row,"DPO","DPO-HMA200"), axis=1)
                # df['DPO-N-HMA200'] = self.HMA(df,"DPO-N", 200)

        # percent distance from HMA200 (previous value)
        # if inList("HMAPriceXOverPercPrev", list):
        #     df['HMAPriceXOverPercPrev'] = df['HMAPriceXOverPerc'].shift(1)

        # price level for resistance and support
        if inList("PriceLevel", list):
            df['PriceLevel'] = df[VALUE].apply(lambda x: round(x, -int(math.log10(x)-2))) 

        # percentage difference in price
        if inList("PriceDiffPerc", list):
            df['PriceDiffPerc'] = df[VALUE].pct_change()

        # time series forecast of price
        if inList("TSF_Price", list):
            df['TSF_Price'] = pd.Series(ta.TSF(df[VALUE], timeperiod=7), index=df.index)
        # if inList("TSF_Price", list):
            # df['TSF_Price_14'] = pd.Series(ta.TSF(df[VALUE], timeperiod=14), index=df.index)
            # df['TSF_Price_21'] = pd.Series(ta.TSF(df[VALUE], timeperiod=21), index=df.index)

        # percent distance from EMA 100
        if inList("PriceDistance", list):
            df['PriceDistance'] = df.apply (lambda row: self.priceDistance(row), axis=1)

        # MACD       
        if inList("MACD", list):
            df['MACD'],df['MACD_SIGNAL'],df['MACD_HISTORY'] = ta.MACD(df[VALUE])
            df['MACD_strategy'] = df.apply (lambda row: self.macd_strategy(row), axis=1) 
            # df['uptrend_MACD'] = df.apply (lambda row: self.macd_uptrend(row), axis=1)
            # df['signal_MACD'] = df.apply (lambda row: self.macd_buy_signal(row), axis=1) 

        # AVERAGE TRUE RANGE
        if inList("ATR", list) or inList("CHOP", list):
            df['ATR'] = pd.Series(ta.ATR(df["High"],df["Low"],df["Close"], timeperiod=22), index=df.index)

        # chandelier_multiplier
        if inList("CHANDELIER", list):
            chand_mult = 3
            chand_period = 22
            df['rollhi'] = df["Close"].rolling(chand_period).max() 
            df['rolllo'] = df["Close"].rolling(chand_period).min()
            df['CHANDELIER_LONG'] = df['rollhi'] - df["ATR"] * chand_mult
            df['CHANDELIER_SHORT'] = df['rolllo'] - df["ATR"] * chand_mult
            df['CHANDELIER_EXIT'] = df.apply (lambda row: row['CHANDELIER_LONG'] if row['Close'] >= row['CHANDELIER_LONG'] else row['CHANDELIER_SHORT'], axis=1)
            df = df.drop(columns=['rollhi','rolllo','CHANDELIER_SHORT','CHANDELIER_LONG'])

        # BOLLINGER BANDS
        if inList("BBANDS", list):
            df['BBANDS_HI'],df['BBANDS_MID'],df['BBANDS_LOW'] = ta.BBANDS(df[VALUE], timeperiod=20,nbdevup=2, nbdevdn=2,matype=0)
            df['BBANDS_SQUEEZE'] = df.apply (lambda row: self.bollinger_squeze(row), axis=1)
            # df['BBANDS_SQUEEZE_MA7'] = pd.Series(ta.MA(df["BBANDS_SQUEEZE"], timeperiod=3, matype=0), index=df.index)
            df['BBANDS_OSC'] = df.apply (lambda row: self.bollinger_osc(row), axis=1)
            df['BBANDS_OSC_MA7'] = pd.Series(ta.MA(df["BBANDS_OSC"], timeperiod=7, matype=0), index=df.index)
        
        # DONCHIAN BARS
        # donchian = tanewlib.volatility.DonchianChannel(df["High"], df["Low"], df["Close"])
        # df["DONCHIAN_HI"] = donchian.donchian_channel_hband() 
        # df["DONCHIAN_LO"] = donchian.donchian_channel_lband() 
        # df['DONCHIAN_HI_DiffPerc'] = df['DONCHIAN_HI'].pct_change()
        # df['DONCHIAN_LO_DiffPerc'] = df['DONCHIAN_LO'].pct_change()

        # CHOPPINESS INDICATOR
        if inList("CHOP", list):
            window = 14
            df['CHOP'] = 100 * np.log10(df['ATR'].rolling(window).sum() / (df['High'].rolling(window).max() - df['Low'].rolling(window).min())) / np.log10(window)
            df["CHOP_SIGNAL"] = df.apply (lambda row: self.chop_signal(row) , axis=1) 

        # AVERAGE DIRECTIONAL INDEX
        if inList("ADX", list) or inList("ADX_signal", list):
            df['ADX'] = pd.Series(ta.ADX(df['High'], df['Low'], df['Close'], timeperiod=14))
            df['ADX_signal'] = df.apply (lambda row: -1 if math.isnan(row['ADX']) else int(row['ADX']/25), axis=1)

        # AARON INDICATOR
        if inList("AROON", list):
            df['AROON_DOWN'],df['AROON_UP'] = ta.AROON(df["High"],df["Low"])
            df['AROON_OSC'] = ta.AROONOSC(df["High"],df["Low"])
            # df["AROON_HELPER"] = df.apply (lambda row: self.aroon_helper(row), axis=1) 

        # PARABOLIC SAR
        df['SAR'] = pd.Series(ta.SAR(df['High'], df['Low'], acceleration=0.02, maximum=0.2))

        # SUPERTREND
        # https://github.com/arkochhar/Technical-Indicators/blob/master/indicator/indicators.py
        if inList("SUPERTREND", list):
            period = 10
            df = self.Supertrend(df,3,period)
           
        # minmax(df,7,"priceMinMax")
        # df['support'] = df.apply (lambda row: self.resistanceAndSupport(df,row), axis=1)
        # df['resistance'] = df.apply (lambda row: self.resistanceAndSupport(df,row, support=False), axis=1)

        # FEATURES
        # df['label_RSI'] = df['RSI'].apply(lambda x: 'oversold' if x<35 else 'overbought' if x>85 else " ")
        # df['label_xover'] = df.apply (lambda row: self.suggestion(row), axis=1)
        # df["label_aroon"] = df.apply (lambda row: self.aroon_bull(row), axis=1)   
        # df["label_chop"] = df.apply (lambda row: self.chop_label(row), axis=1) 
        # df["label_BBAND"] = df.apply (lambda row: self.bollinger(row), axis=1) 
        
        if inList("STRATEGY", list):
            df = self.createARStrategy(df, "strategy")
        # df['trend_strength'] = df.apply(lambda row: np.sign(self.trend_strength(row)), axis=1)
        # df['trend_direction'] = df.apply(lambda row: np.sign(self.trend_direction(row)), axis=1)
        # df = self.createARStrategy(df, "strategy2","uptrend_EMA200")
        # df = self.createARStrategy(df, "strategy3","uptrend_EMA")
        if inList("signal_PumpDump", list):
            df['signal_PumpDump'] = df.apply(lambda row: self.pumpdump(row), axis=1)

        if inList("pivot", list):
            df["pivot"] = (df["High"]+df["Low"]+df["Close"])/3
            df["pivot"] = df["pivot"].shift(-1)
            df["R1"] = 2*df["pivot"]-df["Low"]
            df["R2"] = df["pivot"]+(df["High"]-df["Low"])
            df["S1"] = 2*df["pivot"]-df["High"]
            df["S2"] = df["pivot"]-(df["High"]-df["Low"])
            df["pivot"] = df["pivot"].shift(1)
            df["R1"] = df["R1"].shift(1)
            df["R2"] = df["R2"].shift(1)
            df["S1"] = df["S1"].shift(1)
            df["S2"] = df["S2"].shift(1)

        if inList("variance", list):
            df["variance"] = pd.Series(ta.STDDEV(df["Close"], timeperiod=5, nbdev=1), index=df.index) 
            df["variance_perc"] = df["variance"]/df["Close"]
        return df