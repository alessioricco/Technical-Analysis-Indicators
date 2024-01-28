#!/usr/bin/env python
# coding: utf-8

# **************************************************
#  Download the data and store in the right folder
# https://medium.com/@korniichuk/lambda-with-pandas-fd81aa2ff25e
# https://github.com/keithrozario/Klayers/blob/master/deployments/python3.8/arns/eu-west-2.csv
# https://github.com/Kooshini/ta-lib-aws-lambda
# **************************************************

# Modules
# import json
# import pandas as pd
# import numpy as np
# import datetime 
# import os
import datetime as dt
from pandas_datareader import data
from TechnicalAnalysis import TechnicalAnalysis
import analysis_ta as analysis

# **************************************************************************
# download from yahoo the daily charts
# **************************************************************************
def download(symbol, date, days=365):
    
        if date is None:    
            dateTimeObj = dt.datetime.now()
        else:
            dateTimeObj = dt.datetime.strptime(date, "%Y-%m-%d")

        date = dateTimeObj.strftime("%Y-%m-%d")
        date_start = (dateTimeObj - dt.timedelta(days=days)).strftime("%Y-%m-%d")

        # dt = datetime.today() - timedelta(days=days_to_subtract)
        #  date_time_obj = datetime. strptime(date_time_str, '%d/%m/%y %H:%M:%S')
        df_source = data.DataReader(symbol, 
                    start=date_start, 
                    end=date, 
                    data_source='yahoo')
        df_source['Date'] = df_source.index
        df_source = df_source.drop(columns=['Adj Close'])
        return df_source

# **************************************************************************
# Apply the TA module and return the dataframe
# **************************************************************************
def process(df_source, signal_list = None):
    technical_analysis = TechnicalAnalysis()
    # if signal_list is None:
    #     signal_list = ["ALLIGATOR","HMA200","SUPERTREND","TREND_SIGNAL",'PriceDiffPerc','TSF_Price','MACD','CHOP','ADX','EMA21Volume','HMA200','EMA100','EMA21','uptrend_EMA','uptrend_EMAPrev','uptrend_MA50MA200','uptrend_MA50MA200Prev','HMAPriceXOverPerc','RSI','AROON','TREND']

    df_source = technical_analysis.ApplyTechnicalAnalysis(df_source,signal_list)
    df_source = df_source.fillna(method='ffill')
    # df_source = df_source.dropna()

    return df_source


# **************************************************************************
# **************************************************************************
def daily_analysis(df, symbol, date):

    df_analysis = df.loc[df.Date <= date].tail(20)
    
    result = analysis.analysis_trend_pulse(df_analysis, symbol)
    return result


# **************************************************************************
# **************************************************************************
def go(symbol, date):

    signal_list = ["ALLIGATOR","HMA200","SUPERTREND","TREND_SIGNAL",'PriceDiffPerc','TSF_Price','MACD','CHOP','ADX','EMA21Volume','HMA200','EMA100','EMA21','uptrend_EMA','uptrend_EMAPrev','uptrend_MA50MA200','uptrend_MA50MA200Prev','HMAPriceXOverPerc','RSI','AROON','TREND']

    if date is None:
        date = dt.datetime.now().strftime("%Y-%m-%d")

    df = download(symbol, date)
    df = process(df, signal_list)
    return {
        'statusCode': 200,
        'body': daily_analysis(df, symbol, date)
    }


# **************************************************************************
# **************************************************************************
def lambda_handler(event, context):
    date = None
    if 'Date' in event.keys():
        date = event['Date']
    else:   
        date = dt.datetime.now().strftime("%Y-%m-%d")
    return go(event['Symbol'], date)
