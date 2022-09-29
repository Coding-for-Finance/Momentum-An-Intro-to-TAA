#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun September 11

@author: danielkotas

Name: Momentum trading strategy
Inputs: Stock market data (closing prices) from Finance!Yahoo or Pickle file
Outputs: Summary statistics table and plot visualizing the risk/returns characteristics of the strategy

Universe: S&P500 Stocks as of September 2022
Intuition: Momentum strategy (buying winners, selling losers) is a well-documented phenomenon in finance
            known to generate excess returns, while being also prone to stark crashes. The explanation might be
             that investors transpose past returns into the future, keeping the momentum going, up until to the point
              of reversal. Momentum has also been established as one of the Fama & French-style risk-factors
               explaining stock returns.
Setup:
"""

# %% Imports

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import pickle
import seaborn as sns


# importing module, need to specify local path where module is located!
sys.path.insert(0, '/Users/danielkotas/Documents/Documents – Daniel’s MacBook Air/Important /Extra Learning/modules')
from portfolio_analytics import port_an
wd = r'/Users/danielkotas/Documents/Documents – Daniel’s MacBook Air/Important /Extra Learning'


# %% Functions

def selector(returns, n_long, n_short):
    cumprod_for_sorting = np.prod(1 + returns) - 1
    winners = list(cumprod_for_sorting.nlargest(n_long).keys())
    losers = list(cumprod_for_sorting.nsmallest(n_short).keys())


    loser_weights = cumprod_for_sorting[losers].div(cumprod_for_sorting[losers].sum(axis=0))
    winner_weights = cumprod_for_sorting[winners].div(cumprod_for_sorting[winners].sum(axis=0))
    return winners, losers, winner_weights, loser_weights

def column_index(df, query_cols):
    cols = df.columns.values
    sidx = np.argsort(cols)
    return sidx[np.searchsorted(cols,query_cols,sorter=sidx)]

#PUT PORT AN CLASS HERE AS WELL!!!




# %% Parameters

# set a parameter whether to load from Yahoo (True) or from pickle file (False). Our if condition in "Data loading"
# will decide based on this parameter which way to import data. Note that if loading from pickle file is selected,
# static data with fixed tickers and dates will be loaded
load_from_yahoo = False
load_from_pickle = True
load_from_excel = False

# define start date for Yahoo import
start_date = '2000-01-01'

# define a list of tickers we want to have prices for. Ticker has to adhere to Yahoo's convention (ticker in the bracket)
# https://finance.yahoo.com/quote/XLE?p=XLE&.tsrc=fin-srch . Tickers can be also loaded from an .xlsx
# NOTE: ^TNX is a 10-year treasury rate. We should be rather using a 1-month libor rate since we hold our investments monthly, but that is unavailable on Yahoo Finance, thus we're using this as a proxy
rf_string = '^TNX'
tickers = ['XLE', 'XLF', 'XLU', 'XLI', 'XLK', 'XLV', 'XLY', 'XLP', 'XLB', rf_string]

# define lookback period, i.e. how much back in time do we look to determine which stocks performed the best and worst
lookback = 6
# define holding period, i.e. how much into the future do we think the returns will be persisent
holding_period = 6

# define number of stocks in the long and short leg
n_long = 3
n_short = 3

own_capital = 1 # how much of our own capital we are using in decimal (1 = 100%)
# define short_cap used for scaling final weights in the short leg of the strategy
short_cap = -1
# define long_cap for scaling final weights in the long leg. We want the legs to sum up to 1 (defined above), i.e. we invest 100% of our capital
# and the short leg finances the rest of the long long leg. We manipulate equation " 1 = short_cap + long_cap "
long_cap = own_capital - short_cap

# we need to define onw-way transaction costs. This varies wildly from the asset, broker etc, but a rule of thumb is to set somewhere from 5 to 20bps
tc = 0.002 #20bps

# %% Data loading

if load_from_yahoo:
    df_prices = yf.download(tickers, start=start_date)['Adj Close']
if load_from_pickle:
    infile = open('df_prices.pkl', 'rb')
    df_prices = pickle.load(infile)
    infile.close()
if load_from_excel:
    df_prices = pd.read_excel('df_prices.xlsx',header=0, index_col = 0)

# %% Data pre-processing
## resample to monthly data
df_prices_reindex = df_prices.copy()
df_prices_reindex['Date'] = df_prices_reindex.index
df_prices_monthly = df_prices_reindex.loc[df_prices_reindex.groupby(pd.Grouper(key='Date', freq='1M')).Date.idxmax()].drop(['Date'],axis=1)

# calculate ordinary returns from prices as 1-period percentage change. As there is no period before the very first,
# this generates a NaN - we want to drop it to have a clean dataframe
df_returns_w_rf = df_prices_monthly.pct_change(1).dropna(axis=0)

# because we are using our own capital, we need to calculate SR while taking risk-free returns into account, i.e.
# taking into consideration what we would have earned if we invested into risk free asset and subtract it. If we modify how much
# of our own capital we use, let's say 80%, then we also need to subtract only 80% of risk free return

df_rf = pd.DataFrame(df_returns_w_rf[rf_string]) /  1200 # we need also divide by 100 because to express as decimals and by 12 to get from p.a. to p.m., assuming monthly periodicity.
df_rf_scaled = df_rf * own_capital
df_returns = df_returns_w_rf.drop([rf_string], axis = 1)

# %% Strategy, portfolio weights, returns calculation

# set up a list of dataframes names for each type of momentum
l_weights_names = list(['df_weights_pureTS', 'df_weights_TS_cumulative', 'df_weights_scaling'])
# set up a list of names of strategies for more intuitive reading, namely for plotting and summary statistics
strategy_names = list(['Momentum Pure TS', 'Momentum Cumulative TS', 'Momentum TS & CS'])


# we need to set up an empty dictionary, where all dataframes will be stored. Option No.1: loop through the names of strategies and write a new dataframe in each iteration of the loop
dict_weights = {}
for weights_name in l_weights_names:
    # we call the dictionary and subset it with the [] brackets to write a new object into it, in our case a dataframe of zeros, inheriting the index and columns names from our dataframe with title returns
    dict_weights[weights_name] = pd.DataFrame(data=0, index=df_returns.index, columns=df_returns.keys())

# Option No. 2: This line does exactly the same as the loop above, but is more neatly written ("pandas loop"). It helps when our dataframes are large, but it this case, the efficiency doesn't play a big role.
# dict_weights = {weights_name: pd.DataFrame(data=0, index=df_returns.index, columns=df_returns.keys()) for weights_name in l_weights_names}


for i in range(lookback, len(df_returns)):

    # slicing our returns to desired lookback period, which we then pass to our selector function
    lookback_returns = df_returns.iloc[i - lookback:i, ]
    # using selector function to determine winners and losers in the past "lookback" months. Will be used for weights assingning by subsetting
    winners, losers, winner_weights, loser_weights = selector(lookback_returns, n_long, n_short)

    # we determine the column index of winners/losers to assign weights, because we are looping through integer index using iloc and assignign to a row-wise sliced df and subset by columns. Note it isn't possible to use
    # df.iloc[i+1:i+1:holding_period][winners/loser], because this returns only a COPY of the desired subset and slice of the original df, but does not access it directly.
    winners_index = column_index(df_returns, winners)
    loser_index = column_index(df_returns, losers)

    # Momentum variation No.1: Pure Momentum. Every period, we determine the winners/losers and assign weights for the following holding period to them based on desired number of titles in long and short leg. Note that this
    # variation makes most sense when holding_period = 1, otherwise we are just overwriting the weights
    dict_weights['df_weights_pureTS'].iloc[i+1:i+1 + holding_period, winners_index] = 1 / n_long
    dict_weights['df_weights_pureTS'].iloc[i+1:i+1 + holding_period, loser_index] = -1 / n_short

    # Momentum variation No.2: Cumulative. Every period, we determine the winners/loser and assign weights for the following holding period, but add them to the weights which we determined previously. We are cumulating the weights which were determined in previous periods
    dict_weights['df_weights_TS_cumulative'].iloc[i+1:i+1 + holding_period, winners_index] = dict_weights['df_weights_TS_cumulative'].iloc[i + 1 :i + 1 + holding_period, winners_index] + 1 / n_long  # change looping through indexes and addition with datetime difference
    dict_weights['df_weights_TS_cumulative'].iloc[i+1:i+1 + holding_period, loser_index] = dict_weights['df_weights_TS_cumulative'].iloc[i + 1 :i + 1 + holding_period, loser_index] - 1 / n_short

    # Momentum variation No.3: Scaled cumulative. Until now, we have been assigning the weights in an equal weights fashion, i.e. if we had 3 "winners", each received a weights of 0.33. Now we want to scale the weights
    # such that the winners which won the most will have the highest weight and vice versa. It means that not only the highest performing stocks will receive positive weights, but we also take magnitude of the performance into account.
    # Practically, we already wrote the code for weights calculation in the "selector" function, therefore we only need to assing the weights
    dict_weights['df_weights_scaling'].iloc[i + 1:i + 1 + holding_period, winners_index] = winner_weights.values
    dict_weights['df_weights_scaling'].iloc[i+1:i + 1 + holding_period, loser_index] =  -1 * loser_weights.values

# Weights determined - need to scale them and multiply with returns to get returns and then sum them up to get strategy returns
dict_scaled_weights = {}
dict_strategy_returns = {}
dict_portfolio_returns = {}
dict_avg_turnover = {}

## scaling: We need to scale the weights such that the long leg and the short leg sum to 1 (we are using 100% of our capital). Also, we take into account long and short caps, meaning
# we limit how much can we actually go short (it would be possible to have long and short legs sum to 1, while the short leg would have a cumulative weight of say 100, meaning we would have extreme, i.e. unrealistically high leverage)
for weights_name, strategy in zip(dict_weights, strategy_names):
    print(weights_name, strategy)
    temp_scaled = dict_weights[weights_name].copy()
    temp_not_scaled = dict_weights[weights_name].copy()

    temp_scaled[temp_not_scaled > 0] = temp_not_scaled.div(temp_not_scaled[temp_not_scaled > 0].sum(axis=1), axis=0) * long_cap
    temp_scaled[temp_not_scaled < 0] = (temp_not_scaled.div(temp_not_scaled[temp_not_scaled < 0].sum(axis=1), axis=0)) * short_cap

    turnover = temp_scaled.diff(1).abs().sum(axis=1)
    dict_avg_turnover[strategy] = turnover.mean()
    transaction_costs = turnover * tc


    dict_scaled_weights[weights_name + '_scaled'] = temp_scaled
    dict_strategy_returns[strategy] = temp_scaled * df_returns

    # we sum across assets to get portfolio returns, accounting for transactions costs calculate above and also risk-free earned
    dict_portfolio_returns[strategy] = (temp_scaled * df_returns).sum(axis=1) - transaction_costs + (df_rf[rf_string] * (1-own_capital)) # if not fully invested and lending, we need to earn the proportion on risk-free


# adding an equal weights benchmark (1 divided by number of assets available or simply an arithmetic mean of CS returns)
df_BM = pd.DataFrame(df_returns.mean(axis=1))
df_BM.columns = ['Benchmark']
df_all_strategies = pd.DataFrame(dict_portfolio_returns)


# %% Summary statistics

# we defined a class with functions which calculate basic summary statistics. It is necessary to pass a labeled dataframe with returns observed in the strategies and the annualization factor (if monthly data = 12)
# for calculating the statistics. We have an optinonal argument "benchmark" for comparison of strategies to benchmark
summary = port_an(returns = df_all_strategies, ann_factor= 12,benchmark=df_BM ,rf = df_rf_scaled)

df_summary = summary.an_w_BM()
max_return = df_summary['Geometric Average - Excess (%)'].max()
max_return_name = df_summary['Geometric Average - Excess (%)'].idxmax()
print('Highest average return of', max_return ,'% was achieved by ', max_return_name,'strategy')

max_sharpe = df_summary['Sharpe Ratio'].max()
max_sharpe_name = df_summary['Sharpe Ratio'].idxmax()
print('Highest sharpe ratio of', max_sharpe ,'was achieved by',max_sharpe_name,'strategy')

# %% Visualizations

## NAV Plot
df_NAVs = summary.NAVs # saving a df of the NAVs we calculated in the class "port_an"
df_NAVs.plot() # calling a simple plot
plt.title(f"NAV Comparison of {len(strategy_names)} momentum strategies") # adding a title with a dynamic variable if one decides to add/reduce strategies
plt.xlabel('Date') # adding x and y-axes labels
plt.ylabel('Cumulative NAVs')
plt.savefig(f"plots/NAVs.png")
plt.show() #if we want to display the final plot, displays the plot with the propiertes we defined above. Note this has to be the last line to take all properties into account

## Weights stacked plot

for strategy, name in zip(dict_scaled_weights, strategy_names):
    fig, ax = plt.subplots()
    # split dataframe df into negative only and positive only values
    df_neg, df_pos = dict_scaled_weights[strategy].clip(upper=0), dict_scaled_weights[strategy].clip(lower=0)
    # stacked area plot of positive values
    df_pos.plot.area(ax=ax, stacked=True, linewidth=0.)
    # reset the color cycle
    ax.set_prop_cycle(None)
    # stacked area plot of negative values, prepend column names with '_' such that they don't appear in the legend
    df_neg.rename(columns=lambda x: '_' + x).plot.area(ax=ax, stacked=True, linewidth=0.)
    # rescale the y axis
    ax.set_ylim([df_neg.sum(axis=1).min(), df_pos.sum(axis=1).max()])
    plt.title(f"Weights of {name} strategy")
    plt.savefig(f"plots/{name}_weights.png")
    plt.show()
    # https://stackoverflow.com/questions/52872938/stacked-area-plot-in-python-with-positive-and-negative-values

## Scatter plot
for strategy, name in zip(dict_portfolio_returns, strategy_names):
    sns.regplot(x = df_BM, y = dict_portfolio_returns[strategy])
    plt.title(f"Scatter plot of {name} strategy against benchmark")
    # create a folder called "plots" in the working directory for saving the plots
    plt.savefig(f"plots/{name}_scatter.png")
    plt.show()



