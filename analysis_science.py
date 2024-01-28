from scipy.signal import argrelextrema
import numpy as np
# import math


# # **************************************************************************
# def _boolrelextrema(data, comparator, axis=0, order=1, mode='clip'):

#     if((int(order) != order) or (order < 1)):
#         raise ValueError('Order must be an int >= 1')

#     datalen = data.shape[axis]
#     locs = np.arange(0, datalen)

#     results = np.ones(data.shape, dtype=bool)
#     main = data.take(locs, axis=axis, mode=mode)
#     for shift in range(1, order + 1):
#         plus = data.take(locs + shift, axis=axis, mode=mode)
#         minus = data.take(locs - shift, axis=axis, mode=mode)
#         results &= comparator(main, plus)
#         results &= comparator(main, minus)
#         if(~results.any()):
#             return results
#     return results

# # **************************************************************************
# def argrelextrema(data, comparator, axis=0, order=1, mode='clip'):

#     results = _boolrelextrema(data, comparator,
#                               axis, order, mode)
#     return np.nonzero(results)

# **************************************************************************
# **************************************************************************
def minmax(df,measure,column, order=2):
    # import numpy as np
    # https://stackoverflow.com/questions/31070563/find-all-local-maxima-and-minima-when-x-and-y-values-are-given-as-numpy-arrays
    
    # import matplotlib.pyplot as plt

    # x = np.array(df["Date"].values)
    df['DateTmp'] = df.index
    x = np.array(df["DateTmp"].values)
    y = np.array(df[measure].values)

    # sort the data in x and rearrange y accordingly
    sortId = np.argsort(x)
    x = x[sortId]
    y = y[sortId]

    df[column] = 0

    # this way the x-axis corresponds to the index of x
    maxm = argrelextrema(y, np.greater, order=order)  # (array([1, 3, 6]),)
    minm = argrelextrema(y, np.less, order=order)  # (array([2, 5, 7]),)
    for elem in maxm[0]:
        # max 
        df.iloc[elem, df.columns.get_loc(column)] = 1 
    for elem in minm[0]:
        # min
        df.iloc[elem, df.columns.get_loc(column)] = -1
    return  df.drop(columns=['DateTmp'])

def minmaxTwoMeasures(df,measureMin,measureMax,column, order=2):
    # import numpy as np
    # https://stackoverflow.com/questions/31070563/find-all-local-maxima-and-minima-when-x-and-y-values-are-given-as-numpy-arrays
    
    # import matplotlib.pyplot as plt

    # x = np.array(df["Date"].values)
    df['DateTmp'] = df.index
    x = np.array(df["DateTmp"].values)
    y1 = np.array(df[measureMin].values)
    y2 = np.array(df[measureMax].values)

    # sort the data in x and rearrange y accordingly
    sortId = np.argsort(x)
    x = x[sortId]
    y1 = y1[sortId]
    y2 = y2[sortId]

    df[column] = 0

    # this way the x-axis corresponds to the index of x
    maxm = argrelextrema(y2, np.greater, order=order)  # (array([1, 3, 6]),)
    minm = argrelextrema(y1, np.less, order=order)  # (array([2, 5, 7]),)
    for elem in maxm[0]:
        # max 
        df.iloc[elem, df.columns.get_loc(column)] = 1 
    for elem in minm[0]:
        # min
        df.iloc[elem, df.columns.get_loc(column)] = -1
    return  df.drop(columns=['DateTmp'])
