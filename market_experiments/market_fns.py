import os
import pandas as pd
import json
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

def getStockPrices(sym,out_format):
    # pulls stock data from alpha vantage api

    # really smart and secure encoding of api key
    av_api_key = "GY!02!50P!R0B!LZ0DS!20ZK!0I1".replace('0','').replace('!','')
    ts = TimeSeries(key=av_api_key, output_format=out_format)
    data, meta_data = ts.get_daily_adjusted(symbol=sym, outputsize='full')
    return data,meta_data

def getLocalStock(sym,col_list = None):

	# include col_list to only get specific columns
	fpath = os.path.join('stock_data','sp500','{}.json'.format(sym))
	with open(fpath, 'r') as myf:
		dstr = myf.read()
	data_dict = json.loads(dstr)[0]
	df = pd.DataFrame.from_dict(data_dict).transpose().astype(float)
	df = df.reindex(index=df.index[::-1])
	
	#fix column names
	cols = {n:'{0} - {1}'.format(sym,n.split('. ')[1]) for n in df.columns}
	df = df.rename(columns=cols)
	if col_list is not None:
		col_list = ['{0} - {1}'.format(sym,c) for c in col_list]
		df = df[col_list]

	return df

def cloneDataSP500():
    # this is for writing the stock data from alpha vantage to json files
    # using SP500_file (specified below) as a list of stock tickers to use 

    sp500_file = os.path.join('stock_data','constituents_csv.csv')
    sp500df = pd.read_csv(sp500_file)

    # have to do this in pieces; limit api per day...
    # indicate last stock and start from there
    last_stock = 'MTD'
    toggle = False

    for symbol in sp500df.Symbol:

        if toggle:

            print('getting stock: {}'.format(symbol))
            jdata = getStockPrices(symbol.strip(),'json')
            data_str = json.dumps(jdata)
            with open(os.path.join('stock_data','sp500', '{}.json'.format(symbol)), 'w') as datafile:
                datafile.write(data_str)

            # dont exceed 5 requests per minute
            time.sleep(16)

        if symbol.strip() == last_stock:
            toggle = True


def getSymbolList():
	# gets list of tickers for which there is json data in stock_data/sp500
	data_dir = os.path.join('stock_data','sp500')
	return ['.'.join(f.split('.')[:-1]) for f in os.listdir(data_dir) if f.endswith('.json')]


def toMatFile(dflist):
	# converts pandas df to matlab matrix
	# saves column names as a strings in varlist

	# save prices
	df = pd.concat(dflist,axis=1,sort=True)
	df = df.dropna()
	dfmat = df.values
	scipy.io.savemat('marketdata.mat', mdict={'mktdata': dfmat})

	# save stock names
	varlist = list(df.columns)
	out_list = np.zeros((len(varlist),), dtype=np.object)
	out_list[:] = varlist
	scipy.io.savemat('varlist.mat', mdict={'varlist': out_list})


def nDimLegendre(origdf,n):
	# applys nth order legendre polynomial to each column of df
	# might be useful for polynomial regression
	df = origdf.copy()
	for col in df.columns:
		for k in range(1,n+1):
			lcoeffs = np.zeros(k)
			lcoeffs[-1] = 1
			ncol = "L_{0}({1})".format(k,col)
			print('adding column',ncol)
			df[ncol] = df[col].apply(np.polynomial.legendre.Legendre(lcoeffs))
	return df

def percentChange(df):
	# return df timeseries of percent change on each day
	# calculated as difference in sequential close prices, not difference between open and close
	diff_df = df.diff().shift(-1)
	percent_df = 100*diff_df.divide(df).shift(1).dropna()
	return percent_df

def gaussSmooth(df,n=5,std=2):
	# rolling average with gaussian weights on window
	# make n bigger for wider window
	# make std bigger for more evenly distruted weights in window
	return df.rolling(window=n,win_type='gaussian',center=True).mean(std=std).dropna()



if __name__ == "__main__":

	# TODO 
		# implement and solve ws = argmin_w{||y-[ldf]*w||_2 + (lam)*||w||_1}



	# stocks and which data
	stocklist = ['GOOGL','AAPL','ADP','ADBE','ATVI']
	column_list = ['adjusted close',]

	# build one df
	dfs = [getLocalStock(s,column_list) for s in stocklist]
	mdf = pd.concat(dfs,axis=1,sort=True)
	mdf = mdf.dropna()

	# get day percent change
	pdf = percentChange(mdf)

	# get smoothed version of data
	spdf = gaussSmooth(pdf)

	# plot original and smooth (of just google)
	googdf = pd.concat([pdf['GOOGL - adjusted close'],spdf['GOOGL - adjusted close']],axis=1,sort=True)
	googdf.plot()

	# polynomials for finding model
	# ldf = nDimLegendre(mdf,2)