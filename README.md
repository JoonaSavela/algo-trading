# Algorithmic trading project
by Joona Savela

## Introduction

This project consists of all the aspects of algorithmic trading: data collection, multiple strategy optimization based on the collected data, portfolio optimization of multiple strategies, and implementation of the optimized strategy portfolio as a fully functioning trading algorithm. The programming language python was used to code the entirety this project.

> Note: This project is never finished! There are many things which I'd like to implement, some of which are listed at the very end of this README.

The assets traded by this algorithm include some popular cryptocurrencies and their derivatives which follow their daily price changes, called [leveraged tokens](https://ftx.com/markets/lt).

> Why cryptocurrencies and their derivatives? There are three main reasons (in no particular order): 1. the REST APIs of cryptocurrency exchanges are fairly easy to use, 2. the fees of these exchanges are always relative to the value of the trades so trading can be started with small amount of capital, and 3. cryptocurrencies have historically been quite volatile (which is good for a trader like me).


## Overview

### Data gathering (`data.py`)

I collected cryptocurrency data from the website [cryptocompare.com](https://www.cryptocompare.com/). This website has real-time and historical minute price data available easily and quickly, but the same is true for FTX API as well, so there is no real reason to be using this website over FTX (see TODOs). To save time and space, I only collect data from the cryptocurrencies BTC, ETH, BCH, LTC, and XRP.

Other data that I gather includes

- FTX orderbook data an the previously mentioned cryptocurrencies adn their derivatives
- Personal order history, from which I remove non-filled orders (e.g. cancelled orders)
- Total value of my FTX wallet in USD (with a timestamp).

Due to the changing nature of the oriderbooks and the total value of my FTX wallet, and since cryptocompare minute price data is only availabe up to 7 days into the past (for my free API created 1.5 years ago; nowadays it's only 1 day for the free API), I run this data collection code once every two days or so.

### Strategy optimization (`parameter_search.py`)

Strategy: any decision making process which could be used to trade assets. Here each strategy cannot trade more frequently than once every 2 hours.

This project attempts to take a data-driven approach to strategy optimization while taking into account as many complications and restrictions that the real world brings as possible. For example, one must take into account that in the real world, for each trade:

- One must pay commissions (0.07% of the trade value in [ftx](ftx.com)).
- The execution price might not be the same as the buy price due to the distribution of existing bid/ask orders (also known as the orderbook), and this difference depends on the size of the trade.
- One must pay taxes from the winning trades (losses on cryptocurrency derivatives are not eligible for exemption in Finland).

The optimization procedure is quite straight-forward: return the values of the adjustable parameters which maximize return. However, it should be noted that during the optimization, optimal conditional order type and value are calculated automatically, and returned for the optimal strategy. Here conditional order types include take profit, stop loss, and trailing stop loss orders, and a value of, for example 3 for take profit orders means that if the price of an asset passes 3x the price of the asset at buy time, then the asset is automatically sold at market prices. These conditional orders are often crucial for the success of the strategy.

When optimizing a strategy, some options must be specified: what type of strategy it is (i.e. how to calculate when to buy or sell), is shorting enabled, is trading done on which cryptocurrency, and should the strategy use 3x or 1x leveraged tokens (see introduction above). Running the optimization procedure for multiple (but not all) different option sets yields a collection of strategies, each with a different performance and variance. Since multiple strategies can be obtained this way, portfolio optimization can be applied to obtain a strategy that is more robust to different market conditions than any single strategy by itself.

Finally, for some strategies it is possible to change the execution time from immediately after the hour change (e.g. at 3:00 PM, at 5:00 PM, ...) to x minutes after the hour change (e.g. 54 minutes after: at 3:54 PM, at 5:54 PM, ...). If real-time data can be collected such that it ends precisely when requested (i.e. at 3:54 PM, which is now from the point of view of the strategy), then it is often the case that the strategy performs better when executed e.g. 40-59 minutes after the hour change (e.g. during 2:40-2:59) rather than at exactly at the hour change (e.g. at 3:00). These values are called `displacements` in the code.


### Implementing strategies on FTX API (`trade.py`)

This file takes the saved strategy parameters and their weights – both of which were calculated in `parameter_search.py` – and starts trading on FTX according to those values. In short, the ideas that were simulated when the strategies were being optimized are implemented for the real world here, using a [fork](https://github.com/JoonaSavela/ftx) of the python sample code of the FTX REST API.




## Real-world performance

Visualization of real-world performance of my algorithm is not yet implemented; come back later.




## TODOs, in no particular order

### General

- write a README
- refactor code
    - rename files
    - split long functions into many functions
    - rename `m` and `m_bear`
- create appealing visualizations for real-world performance
- make a `requirements.txt`

### `data.py`

- update old data gathering/saving code to use pandas
- transfer into using FTX API instead of cryptocompare?


### `parameter_search.py`

- change stop loss & take profit implementation such that each strategy places one take profit order and simultaneously either a stop loss or a trailing stop loss order
    - try different optimization methods (bayesian optimization basin-hopping, ...), analyze running time & correctness -> choose the best one
        - apply multiple attribute value theory to make a decision? unless the choice is obvious
        - for each base parameter combination (e.g. (12, 41)), save the resulting optimal take profit/stop loss parameters so that they can be used as initial guesses in the future
    - check that this can be done in ftx
- automate result visualization (saving)


### `trade.py`

- refactor code
    - only use "trigger order" or "conditional order" (not both) when naming variables/functions
- make transitioning from an old strategy/parameters as smooth as possible
    - using deques is probably the simplest way
        - how to save a dict of deques?
- instead of creating 'prev_x' columns, create a copy of `buy_info`, and pass both the copy and the updated data frame into `process_trades`
- instead of asking if to wait for signal to change for each strategy, get the time when the signal changed
    - get the historical price from ftx
    - only do this for frequency='low' strategies?


### `taxes.py`

- finish function for calculating taxes


