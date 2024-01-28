# **************************************************************************
# daily analysis of a stock
# **************************************************************************
# import pandas as pd
import numpy as np
import analysis_lib

def aroon_helper(df):
    # =ifs(AND(AD1055>80,AE1055<20),"Bear",AND(AD1055<50,AE1055<50),"consolidate",AND(AE1055>80,AD1055<20),"Bull",AND(AD1055<AE1055),"up",AND(AD1055>AE1055),"down",True(),"")
    up = df["AROON_UP"]
    down = df["AROON_DOWN"]

    if up>80 and down<20:
        return "Bull"
    if up>95 and down<5:
        return "Strong Bull (-)"
    # if up<50 and down<50:
    #     return "consolidate"

    if down>80 and up<20:
        return "Bear"
    if down>95 and up<5:
        return "Strong Bear (+)"

    if np.abs(up-down) < 25:
        return "range"
    if np.abs(up-down) > 60:
        return "trending"

    if up>down:
        return "up"
    if up<down:
        return "down"
    return ""

# **************************************************************************
# **************************************************************************
def analysis_trend_pulse(df, symbol):
    score = 0
    suggestion = None
    s = "\n"

    buy = 0
    sell = 0
    hold = 0

    most_recent = df.tail(1)
    date = most_recent.iloc[0]['Date']
    five_days = analysis_lib.days_ago(df,date,5)
    ten_days = analysis_lib.days_ago(df,date,10)

    s += "Analysis %s %s\n" % (symbol,date)

    diff_5d = analysis_lib.isIncreasing(five_days['Close'])
    diff_10d = analysis_lib.isIncreasing(ten_days['Close'])
    tmp = analysis_lib.format_percent(most_recent.iloc[0]['PriceDiffPerc']) 
    tmp1 = analysis_lib.format_percent(diff_5d)
    tmp2 = analysis_lib.format_percent(diff_10d)
    s += "Price   : %s (%s/%s/%s 1/5/10d)\n" % (analysis_lib.format_money(most_recent.iloc[0]['Close']), tmp, tmp1, tmp2)
    s += "Forecast: %s\n" % analysis_lib.format_money(most_recent.iloc[0]['TSF_Price'])

    s += "EMA21: %s\n" % analysis_lib.format_money(most_recent.iloc[0]['EMA21'])
    s += "EMA55: %s\n" % analysis_lib.format_money(most_recent.iloc[0]['EMA55'])
    s += "EMA100: %s\n" % analysis_lib.format_money(most_recent.iloc[0]['EMA100'])
    s += "HMA200: %s\n" % analysis_lib.format_money(most_recent.iloc[0]['HMA200'])
 
    # price_diff_5d = analysis_lib.isIncreasing(five_days['Close'])
    # # s = s + "Price moved %s perc since the last 5 days.\n" % (price_diff_5d)
    # s += " (" + analysis_lib.format_percent(price_diff_5d) + " 5d)"

    # price_diff_10d = analysis_lib.isIncreasing(ten_days['Close'])
    # # s = s + "Price moved %s perc since the last 10 days.\n" % (price_diff_10d)
    # s += analysis_lib.format_percent(price_diff_10d) + " 10d\n"

    s += "\n* TREND/RANGE (Strong/Weak)\n"
    if most_recent.iloc[0]['CHOP_SIGNAL'] == 1:
        s = s + "(S) Strong Trend\n" 
    if most_recent.iloc[0]['CHOP_SIGNAL'] == -1:
        s = s + "(W) Sideway Trend\n" 
        hold +=1
    if most_recent.iloc[0]['CHOP_SIGNAL'] == 0:
        s = s + "(W) Choppy Trend\n" 
        hold +=1
    # if most_recent.iloc[0]['ADX_signal'] > 0:
    #     s = s + "ADX over 25. This is a potential trend consolidation signal\n" 

    diff_10d = analysis_lib.isIncreasing(ten_days['ADX'])
    tmp2 = " (" + analysis_lib.format_percent(diff_10d) + " 10d)"
    s += "%s ADX: %s %s\n" % (("(S)" if most_recent.iloc[0]['ADX'] > 25 else "(W)"), analysis_lib.format_money(most_recent.iloc[0]['ADX']), tmp2)

    if most_recent.iloc[0]['EMA21Volume'] < most_recent.iloc[0]['Volume']:
        s = s + "(S) Volumes above the average\n" 
        # buy +=1
    else:
        s = s + "(W) Volumes below the average\n" 
        hold +=1

    tmp_score = "    "
    if most_recent.iloc[0]['AROON_OSC'] > 85:
        tmp_score = "(S)"
        buy +=1
    if most_recent.iloc[0]['AROON_OSC'] < -85:
        tmp_score = "(W)"
        sell +=1
    if -50 < most_recent.iloc[0]['AROON_OSC'] and most_recent.iloc[0]['AROON_OSC'] < 50:
        tmp_score = "(=)"
        hold +=1

    tmp1 = ""
    tmp2 = ""
    s += "%s AROON OSC: %s\n" % (tmp_score,analysis_lib.format_money(most_recent.iloc[0]['AROON_OSC']))


    s += "\n* AVERAGES\n"
    if most_recent.iloc[0]['uptrend_HMA200'] == 1:
        s = s + "(+) Close > HMA200\n"
        buy +=1
    if most_recent.iloc[0]['uptrend_HMA200'] == -1:
        s = s + "(-) Close < HMA200\n"
        sell +=1

    if most_recent.iloc[0]['EMA100'] <= most_recent.iloc[0]['Close']:
        s = s + "(+) Close > EMA100\n"  
        buy +=1 
    if most_recent.iloc[0]['EMA100'] >= most_recent.iloc[0]['Close']:
        s = s + "(-) Close < EMA100\n"  
        sell +=1 

    if most_recent.iloc[0]['EMA21'] <= most_recent.iloc[0]['Close']:
        s = s + "(+) Close > EMA21\n"  
        buy +=1
    if most_recent.iloc[0]['EMA21'] >= most_recent.iloc[0]['Close']:
        s = s + "(-) Close < EMA21\n"  
        sell +=1

    # df[VALUE] > df['EMA21'] and df['EMA21'] > df['EMA55'] and df['EMA55'] > df['EMA100'] and df['EMA100'] > df['SMA200']
    if most_recent.iloc[0]['uptrend_EMA'] == 1:
        s = s + "(+) Close > EMA:21:55:100 > SMA200\n"
        buy += 1    
    if most_recent.iloc[0]['uptrend_EMA'] == -1:
        s = s + "(-) Close < EMA21:55:100 < SMA200\n"
        sell += 1   

    # 
    if most_recent.iloc[0]['uptrend_EMA'] != 1 and most_recent.iloc[0]['uptrend_EMAPrev'] == 1:
        s = s + "(=): EMA ribbon now not aligned\n"
        # sell += 1     
    if most_recent.iloc[0]['uptrend_EMA'] == 1 and most_recent.iloc[0]['uptrend_EMAPrev'] != 1:
        s = s + "(+): EMA ribbon now aligned\n"
        # buy += 1 

    if most_recent.iloc[0]['uptrend_EMA'] != -1 and most_recent.iloc[0]['uptrend_EMAPrev'] == -1:
        s = s + "(=): EMA ribbon now not aligned\n"
        # buy += 1     
    if most_recent.iloc[0]['uptrend_EMA'] == -1 and most_recent.iloc[0]['uptrend_EMAPrev'] != -1:
        s = s + "(-): EMA ribbon now aligned\n"
        # sell += 1 


    if most_recent.iloc[0]['uptrend_MA50MA200'] == 1:
        s = s + "(+) Close > MA50:200\n"
        buy += 1
    if most_recent.iloc[0]['uptrend_MA50MA200'] == -1:
        s = s + "(-) Close < MA200:50\n"
        sell +=1

    if most_recent.iloc[0]['uptrend_MA50MA200'] != 1 and most_recent.iloc[0]['uptrend_MA50MA200Prev'] == 1:
        s = s + "(=) Close > MA50:200 Invalidated\n"

    if most_recent.iloc[0]['uptrend_MA50MA200'] == 1 and most_recent.iloc[0]['uptrend_MA50MA200Prev'] != 1:
        s = s + "(+) Close > MA50:200 Validated\n"
        # buy += 1    

    if most_recent.iloc[0]['uptrend_MA50MA200'] != -1 and most_recent.iloc[0]['uptrend_MA50MA200Prev'] == -1:
        s = s + "(=) Close > MA200:50 Invalidated\n"

    if most_recent.iloc[0]['uptrend_MA50MA200'] == -1 and most_recent.iloc[0]['uptrend_MA50MA200Prev'] != -1:
        s = s + "(-) Close > MA200:50 Validated\n"
        # sell += 1  

    # if most_recent.iloc[0]['HMAPriceXOverPerc'] >= 50:
    #     s = s + "Price and HMA200 are very far. This could lead to a reversal\n"
    #     # score = score + 1
    # if most_recent.iloc[0]['HMAPriceXOverPerc'] <= -50:
    #     s = s + "Price and HMA200 are very far. This could lead to a reversal\n"

    diff_5d = analysis_lib.isIncreasing(five_days['HMAPriceXOverPerc'])
    diff_10d = analysis_lib.isIncreasing(ten_days['HMAPriceXOverPerc'])
    tmp1 = " (" + analysis_lib.format_percent(diff_5d) + " 5d)"
    tmp2 = " (" + analysis_lib.format_percent(diff_10d) + " 10d)"
    s += "Close/HMA200: %s %s %s\n" % (analysis_lib.format_percent(most_recent.iloc[0]['HMAPriceXOverPerc']/100), tmp1, tmp2)

    s += "\n* MOMENTUM\n"
    # rsi_perc = round(analysis_lib.isIncreasing(five_days['RSI']) * 100,1)
    # s = s + "%s RSI value is %s and (%s perc) since the last 5 days.\n" % (round(most_recent.iloc[0]['RSI'],1), rsi_perc)

    tmp_score = "    "
    if most_recent.iloc[0]['RSI'] > 70:
        tmp_score = "(OB)"
        buy +=1
    if most_recent.iloc[0]['RSI'] < 30:
        tmp_score = "(OS)"
        sell +=1

    # diff_5d = analysis_lib.isIncreasing(five_days['RSI'])
    # diff_10d = analysis_lib.isIncreasing(ten_days['RSI'])
    # tmp1 = " (" + analysis_lib.format_percent(diff_5d) + " 5d)"
    # tmp2 = " (" + analysis_lib.format_percent(diff_10d) + " 10d)"
    tmp1 = ""
    tmp2 = ""
    s += "%s RSI      : %s %s %s\n" % (tmp_score, analysis_lib.format_money(most_recent.iloc[0]['RSI']), tmp1, tmp2)

    # tmp_score = "    "
    # if most_recent.iloc[0]['AROON_OSC'] > 85:
    #     tmp_score = "(S)"
    #     buy +=1
    # if most_recent.iloc[0]['AROON_OSC'] < -85:
    #     tmp_score = "(W)"
    #     sell +=1
    # if -50 < most_recent.iloc[0]['AROON_OSC'] and most_recent.iloc[0]['AROON_OSC'] < 50:
    #     tmp_score = "(=)"
    #     hold +=1

    # # diff_5d = analysis_lib.isIncreasing(five_days['AROON_OSC'])
    # # diff_10d = analysis_lib.isIncreasing(ten_days['AROON_OSC'])
    # # tmp1 = " (" + analysis_lib.format_percent(diff_5d) + " 5d)"
    # # tmp2 = " (" + analysis_lib.format_percent(diff_10d) + " 10d)"
    # tmp1 = ""
    # tmp2 = ""
    # s += "%s AROON OSC: %s %s %s\n" % (tmp_score,analysis_lib.format_money(most_recent.iloc[0]['AROON_OSC']), tmp1, tmp2)

    tmp1 = aroon_helper(most_recent.iloc[0])
    if tmp1 != "":
        s += "     AROON IND: %s\n" % tmp1

    # aroon_oscillator_perc = round(analysis_lib.isIncreasing(five_days['AROON_OSC']) * 100,1)
    # s = s + "AROON OSC value is %s and (%s perc) since the last 5 days.\n" % (round(most_recent.iloc[0]['AROON_OSC'],1), aroon_oscillator_perc)
    # if most_recent.iloc[0]['AROON_OSC'] > 85:
    #     s = s + "AROON Oscillator near the top. This could lead to a reversal\n"
    #     sell +=1
    # if most_recent.iloc[0]['AROON_OSC'] < -85:
    #     s = s + "AROON Oscillator near the top. This could lead to a reversal\n"
    #     buy +=1

    # if most_recent.iloc[0]['signal_PumpDump'] == 1:
    #     s += "Pump - this could lead to a fast dump"
    #     sell += 1
    # if most_recent.iloc[0]['signal_PumpDump'] == -1:
    #     s += "Dump - this could lead to a fast pump"
    #     buy += 1        

    signal_score = buy-sell-hold
    tmp1 = analysis_lib.format_number(signal_score) 
    tmp2 = analysis_lib.format_number(buy)
    tmp3 = analysis_lib.format_number(sell)
    tmp4 = analysis_lib.format_number(hold)
    s += "\nBot Score:\n %s (+: %s -: %s =: %s)\n" % (tmp1, tmp2, tmp3, tmp4)
    if signal_score > 4 and sell == 0:
            s += " strong bullish\n"
    else:
        if signal_score > 4:
            s += " strong bullish\n"
        if signal_score > 2:
            s += " bullish\n"

    if signal_score < -4 and buy == 0:
            s += " strong bearish\n"
    else:
        if signal_score < -4:
            s += " strong bearish\n"
        if signal_score < -2:
            s += " bearish\n"

    if -2 < signal_score and signal_score < 2:
        s += " uncertain\n"


    s += "\n"
    # s += "TREND BULL value: %s (%s perc 5d %s perc 10d)\n" % (round(most_recent.iloc[0]["TREND_BULL"],2), round(analysis_lib.isIncreasing(five_days['TREND_BULL']),1),round(analysis_lib.isIncreasing(ten_days['TREND_BULL']),1))
    # s += "TREND BEAR value: %s (%s perc 5d %s perc 10d)\n" % (round(most_recent.iloc[0]["TREND_BEAR"],2), round(analysis_lib.isIncreasing(five_days['TREND_BEAR']),1),round(analysis_lib.isIncreasing(ten_days['TREND_BEAR']),1))
    # s += "TREND      value: %s (%s perc 5d %s perc 10d)\n" % (round(most_recent.iloc[0]["TREND"],2), round(analysis_lib.isIncreasing(five_days['TREND']),1),round(analysis_lib.isIncreasing(ten_days['TREND']),1))

    s += "STREND:\n (%s) SL:%s\n" % ("buy" if most_recent.iloc[0]["SuperTrend_Indicator"] == 1 else "sell", analysis_lib.format_money(most_recent.iloc[0]["SuperTrend"]))
    s += "SAR   :\n (%s) SL:%s\n" % ("buy" if most_recent.iloc[0]["SAR"] < most_recent.iloc[0]["Close"] else "sell", analysis_lib.format_money(most_recent.iloc[0]["SAR"]))

    trend_score = int(most_recent.iloc[0]["TREND"])
    threshold = 4
    suggestion = -1 if trend_score < -threshold else (1 if trend_score > threshold else None)
    if suggestion is None:
        score_day_before = ((df.tail(2)).head(1)).iloc[0]["TREND"]
        if np.sign(trend_score) != np.sign(score_day_before):
            # trend changed sign
            suggestion = 0
    
    tmp1 = analysis_lib.format_number(most_recent.iloc[0]["TREND"]) 
    tmp2 = analysis_lib.format_number(most_recent.iloc[0]["TREND_BULL"])
    tmp3 = analysis_lib.format_number(most_recent.iloc[0]["TREND_BEAR"])
    s += "\nVolume Trend:\n %s (+: %s -: %s)\n" % (tmp1, tmp2, tmp3)   
    # if suggestion is None:
    #     s += "volume trend score is keeping its direction\n" 
    if suggestion == 0:
        s += " changed sign.\n risk of reversal\n"
    if suggestion == 1:
        s += " strong bullish.\n risk of reversal\n"
    if suggestion == -1:
        s += " strong bearish.\n risk of reversal\n"

    if (most_recent.iloc[0]["TREND"]<-4 and most_recent.iloc[0]["TREND_BULL"]<0.2):
        s += " (+) bulls are weak\n"
    if (most_recent.iloc[0]["TREND"]>4 and most_recent.iloc[0]["TREND_BEAR"]<0.2):
        s += " (-) bears are weak\n"

    if (most_recent.iloc[0]["strategy_helper"] != ""):
        s += "\nARI strategy suggest: %s\n" % most_recent.iloc[0]["strategy_helper"]
    # if (most_recent.iloc[0]["TREND_IMPULSE"] != 0):
    #     s += "\nTrend pulse: %s \n" % analysis_lib.format_number(most_recent.iloc[0]["TREND_IMPULSE"])
    #     if (most_recent.iloc[0]["TREND_IMPULSE"] > 0):
    #         # suggested strategy is to find the top and sell before the end of the pulse
    #         s+= " bullish pulse\n"
    #     if (most_recent.iloc[0]["TREND_IMPULSE"] < 0):
    #         # suggested strategy is to find the bottom and buy before the end of the pulse
    #         s+= " bearish pulse\n"
    #     if (most_recent.iloc[0]["TREND_IMPULSE"] == 0):
            # s+= " no strong pulse\n"

    # if (most_recent.iloc[0]["TREND_SIGNAL"] != 0):
    #     s += "\nTrend signal %s \n" % analysis_lib.format_number(most_recent.iloc[0]["TREND_SIGNAL"])
    #     if (most_recent.iloc[0]["TREND_SIGNAL"] == 0.5):
    #         s+= " start of bearish strong pulse\n"
    #     if (most_recent.iloc[0]["TREND_SIGNAL"] == -0.5):
    #         s+= " start of bullish strong pulse\n"
    #     if (most_recent.iloc[0]["TREND_SIGNAL"] == 1):
    #         s+= " end of bearish pulse (+)\n"
    #     if (most_recent.iloc[0]["TREND_SIGNAL"] == -1):
    #         s+= " end of bullish pulse (-)\n"

    s += "\n"
    s += "buy  now breakeven: %s\n" % analysis_lib.format_money( analysis_lib.WhatIfBuy(100, most_recent.iloc[0]['Close']))
    s += "sell now breakeven: %s\n" % analysis_lib.format_money( analysis_lib.WhatIfSell(100, most_recent.iloc[0]['Close']))
    s += "\n"
    s += "Sell only if HMA200>Close and the trend is strong\n"
    s += "Buy  only if HMA200<Close and the trend is strong\n"
    s += "Take Profit if the trend is weak\n"
    return { 
        'buy': buy,
        'sell': sell,
        'hold': hold,
        'score': score, 
        'suggestion':suggestion, 
        'description':s
        } 
