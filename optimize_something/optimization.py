"""MC1-P2: Optimize a portfolio.  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		   	  			  	 		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		   	  			  	 		  		  		    	 		 		   		 		  
All Rights Reserved  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		   	  			  	 		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		   	  			  	 		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		   	  			  	 		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		   	  			  	 		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		   	  			  	 		  		  		    	 		 		   		 		  
or edited.  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		   	  			  	 		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		   	  			  	 		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		   	  			  	 		  		  		    	 		 		   		 		  
GT honor code violation.  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
-----do not edit anything above this line---
Student Name: Tucker Balch (replace with your name)
GT User ID: cfleisher3
GT ID: 903421975
"""
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import datetime as dt
from scipy.optimize import minimize
from util import get_data, plot_data


# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality
def optimize_portfolio(sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 1, 1),
                       syms=['GOOG', 'AAPL', 'GLD', 'XOM'], gen_plot=False):
    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices_all.fillna(method='ffill', inplace=True)
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # find optimal portfolio allocations
    default_alloc = 1/max(len(syms), 1)
    allocs = np.zeros((len(syms),))
    allocs.fill(default_alloc)

    if allocs.shape[0] > 1:
        allocs[-1] = 1.0 - np.sum(allocs[:-1])

    allocs, cr, adr, sddr, sr = calc_stats(prices.values, allocs)
    print_portfolio(prices, syms, allocs)

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # indexed daily portfolio values
        port_prices = (prices.values*allocs).sum(axis=1)
        port_val = pd.DataFrame(port_prices/port_prices[0])
        port_val = port_val.set_index(prices.index)

        idx_SPY = prices_SPY / prices_SPY.iloc[0]

        # plot indexed portfolio vs SPY
        fig, ax = plt.subplots()
        ax.set_title('Daily Portfolio Value and SPY')
        ax.plot(port_val.index, port_val.values, 'b', label='Portfolio',
                linewidth=0.5)
        ax.plot(idx_SPY.index, idx_SPY.values, 'g', label='SPY',
                linewidth=0.5)

        plt.xlim(port_val.index.values[0], port_val.index.values[-1])
        miny = math.floor(min(port_val.values.min(),
                              idx_SPY.values.min())*10)/10
        maxy = math.ceil(max(port_val.values.max(),
                             idx_SPY.values.max())*10)/10
        plt.ylim(miny, maxy)
        ax.grid(linestyle='--')

        fig.autofmt_xdate()  # rotates and right aligns x labels
        plt.xlabel('Date')
        plt.ylabel('Price')

        plt.legend()
        plt.savefig('fig1.png')

    return allocs, cr, adr, sddr, sr


def print_portfolio(px_df, syms, allocs):
    print(f'***** Optimal Portfolio Summary *****')
    print(f'trading days: {px_df.shape[0]}\n')

    # table header
    header = f"{'symbol':<10}"
    col_names = ['weight', 'px_start', 'px_end', 'cr', 'adr', 'std', 'sr']
    for name in col_names:
        header += f'{name:>10}'
    print(header)

    # table body
    def build_row(*args):
        key, weight, px0, px1, cr, adr, sddr, sr = args
        return f'{key:<10}' \
            f'{weight:>10.3f}' \
            f'{pxs[0]:>10.2f}' \
            f'{pxs[-1]:>10.2f}' \
            f'{cr:>10.3f}' \
            f'{adr:>10.3f}' \
            f'{sddr:>10.3f}' \
            f'{sr:>10.4f}'

    def pxs_to_stats(pxs):
        cr = pxs[-1] / pxs[0] - 1
        dr = pxs[1:] / pxs[:-1] - 1.
        adr = np.mean(dr)
        sddr = np.std(dr)
        sr = adr / sddr
        return cr, adr, sddr, sr

    rows = []
    for i, key in enumerate(syms):
        pxs = px_df[key].values
        cr, adr, sddr, sr = pxs_to_stats(pxs)
        row = build_row(key, allocs[i], pxs[0], pxs[1], cr, adr, sddr, sr)
        rows.append(row)

    # table footer
    port_val = np.matmul(px_df.values, allocs)
    port_pxs = np.squeeze(port_val)
    cr, adr, sddr, sr = pxs_to_stats(port_pxs)
    total_row = build_row('Total', np.sum(allocs), port_pxs[0],
                          port_pxs[1], cr, adr, sddr, sr)
    rows.append(total_row)

    # display msg
    for row in rows:
        print(row)


def calc_stats(PX, allocs):
    # calcs sharpe ration and relevant portfolio stats
    # returns opt_allocs, cr, adr, sddr, sr
    dr = PX[1:] / PX[:-1] - 1.  # position daily returns
    # dr = dr / dr[0]

    # Sharpe ratio objective function
    bnds = tuple([(0.0, 1.0) for _ in allocs])
    cons = ({'type': 'eq', 'fun': lambda x: 1.0 - np.sum(x)})
    res = minimize(sharpe_ratio,
                   allocs,
                   args=(dr,),
                   bounds=bnds,
                   constraints=cons,
                   options={'disp': True},
                   method='SLSQP')

    print(f'X: {res.x}, Y: {res.fun}')
    # calc stats for optimal allocs
    opt_allocs = res.x
    port_vals = (PX*opt_allocs).sum(axis=1)
    cr = port_vals[-1] / port_vals[0] - 1.
    pdr = port_vals[1:] / port_vals[:-1] - 1.

    adr = pdr.mean()
    sddr = pdr.std()
    sr = np.sqrt(252)*adr/sddr

    return opt_allocs, cr, adr, sddr, sr


def sharpe_ratio(allocs, DRS):
    return -np.dot(DRS, allocs).mean()/np.dot(DRS, allocs).std()


def test_code():
    # This function WILL NOT be called by the auto grader
    # Do not assume variables defined here are available to your function/code
    # It is only here to help you set up and test your code
    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!
    start_date = dt.datetime(2008, 6, 1)
    end_date = dt.datetime(2009, 6, 1)
    symbols = ['IBM', 'X', 'GLD', 'JPM']

    # Assess the portfolio
    allocs, cr, adr, sddr, sr = optimize_portfolio(
        sd=start_date,
        ed=end_date,
        syms=symbols,
        gen_plot=True)


if __name__ == "__main__":
    # This code WILL NOT be called by the auto grader
    # Do not assume that it will be called
    test_code()
