import yfinance as yf
import pandas as pd
import numpy as np
import itertools
import multiprocessing
import time
import math
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import plotting
from pypfopt import cla
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from datetime import datetime
import matplotlib.pyplot as plt


################################
    # LIST OF SECTOR ETFS #
################################
etfs = ['XLE', 'XLF', 'XLU', 'XLI', 'GDX', 'XLK', 'XLV', 'XLP', 'XLB', 'XOP', 'IYR', 'XHB', 'ITB','VNQ','GDXJ','IYE','OIH', 'XME','XRT', 'SMH', 'IBB' , 'KBE', 'KRE', 'XTL']

############################################################
    # GETS ALL COMBINATIONS OF ETFS AND PRINT OUT HOW MANY #
############################################################
def get_combinations(n):
    comb = itertools.combinations(etfs,r=n)
    comb_list = [i for i in comb]
    return comb_list

length = len(get_combinations(5))
print(length)



###############################################################################

# THIS FUNCTION DOWNLOADS THE ADJ-CLOSE OF EACH ETF AND STORES IT AS A CSV FILE

###############################################################################

def etfs_download(etfs):
    stockStartDate = '2013-01-01' # PORTFOLIO START DATE
    today = datetime.today().strftime('%Y-%m-%d') # PORTFOLIO END DATE
    df = yf.download(etfs, start=stockStartDate, end=today)["Adj Close"]
    df.to_csv("ETFS-Adj-Close.csv")
etfs_download(etfs)



####################################################

# SEPERATES THE COMBINATION LIST INTO TWO PROCESSES 

####################################################

def two_processes(etfs, n):
    comb = itertools.combinations(etfs,r=n)
    comb_nparr = np.array([i for i in comb]) # CHANGED TO NUMPY ARRAY FOR SPEED
    for i in range(0,len(comb_nparr)):
        if comb_nparr[i][0] == 'XLI':
            first_process = comb_nparr[0:i]
            second_process = comb_nparr[i:]
            return [first_process, second_process]
    
first_process = two_processes(etfs,5)[0]
second_process = two_processes(etfs,5)[1]
# print(first_process)
# print(second_process)



###############################################################################################################

# THE START OF THE SCRIPT
# THE SCRIPT HAS A MANUAL CALCULATION AND AND A PORTFOLIO OPTIMIZATION CALCULATION FROM PYPORTFOLIOOPT LIBRARY 

###############################################################################################################

def portOpt(etfs, n, process):

    function_time = time.time() # STARTS TIMER FOR SCRIPT

    # ASSIGN WEIGHTS FOR MANUAL CALULATION
    # weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

    df = pd.read_csv('etfs-adj-close.csv',index_col='Date') # READS IN ETF DATA
    
    run_times = [] # WILL STORE TIME IT TAKES FOR ITERATIONS STARTING WITH EACH ETF. Example, time for combinations for "XLE" and so on  

    highest_return = 0 # PORTFOLIO WITH HIGHEST ANNUAL RETURN WILL BE STORED HERE

    last_stock = '' # KEEPS TRACK OF LAST ETF THE ITERATION STARTS WITH 

    ##############################################################################
    # loops thorugh each portfolio in each process
    # THIS LOOP FINDS OUT WHEN NEW ITERATION STARTS WITH A DIFFERENT ETF
    # FOR EXAMPLE WHEN IT CHANGES FROM ["XLE" ...] TO ["XLF" ...]
    ##################################################################################

    for port in process:
        print(port)
        port_df = pd.DataFrame()
        if port[0] != last_stock:
            run_times.append('{} took {} sec to run'.format(port[0], (time.time() - function_time)))
        for stock in port:
            port_df[stock] = df[stock]


############################################################
##      MANUAL CALULATION FOR PORTFOLIO OPTIMIZATION
############################################################

        # # show daily simple returns takes (current - previous) / previous 
        # returns = port_df.pct_change()
        # print(returns)
        # stock_variance = returns.var()
        # # print(stock_variance)
        # stock_volatility = returns.std()
        # print(stock_volatility)

        # ## SHOWS HOW COVARIENCE IS CALCULATED FOR GOOG AND AAPL #######

        # # covarianceDF = pd.DataFrame()
        # # covarianceDF["apple"] = returns['AAPL'] - returns['AAPL'].mean()
        # # covarianceDF["google"] = returns['GOOG'] - returns['GOOG'].mean()

        # # covarianceDF['Sum'] =  covarianceDF['apple'] * covarianceDF['google']
        # # print(covarianceDF["Sum"].sum() / 1973)

        # ################################################


        # # covariance matrix 252 annualized the covarience
        # cov_matrix_annual = returns.cov() * 252
        # # print(cov_matrix_annual)
        # # print(weights.T)
        # # print(np.dot(cov_matrix_annual, weights))


        # # calc portfolio variance / covariance matrix
        # port_variance = np.dot(weights.T, np.dot(cov_matrix_annual, weights))
        # # print(port_variance)

        # # STANDARD DEVIATION
        # port_volatility = np.sqrt(port_variance)
        # # print(port_volatility)
        # # print(returns.mean() * 252 )

        # portfolio_simple_annual_return = np.sum(returns.mean() * weights) * 252
        # # print(portfolio_simple_annual_return)

        # # # SHOW EXPECTED ANNUALL RETURN, VOLATILITY (RISK), AND VARIANCE
        # percent_var = str( round(port_variance, 2) * 100) + "%"
        # percent_volatility = str( round(port_volatility, 2) * 100) + "%"
        # percent_return = str( round(portfolio_simple_annual_return, 2) * 100) + "%"

        # print('Expected annual return: ' + percent_return)
        # print('Annual volatility / risk: ' + percent_volatility)
        # print('Annual variance: '+ percent_var)

############################################################

##               PyPortfolioOpt LIBRARY

############################################################

        # CALCULATE EXPECTED RETURNS AND SAMPLE COVARIANCE
        mu = expected_returns.mean_historical_return(port_df)
        # Gets covariance matrix
        S = risk_models.sample_cov(port_df) 
        
        if all(mean < 0 for mean in mu):
            print('bad')
            continue

        # print(mu)
        # print(S)
        # print(type(S))
        # print(df)

        # OPTIMIZE FOR SHARPE RATIO

        ef = EfficientFrontier(mu, S)
        # print(ef)
        weights = ef.max_sharpe()
        # print(weights)
        cleaned_weights = ef.clean_weights()
        # print(cleaned_weights)

        metrics = ef.portfolio_performance(verbose=False) # GETS THE METRICS OF THE PORTFOLIO metrics[0] is the annual return

        #######################################################
        # FINDS OUT IF CURRENT PORTFOLIO HAS THE HIGHEST RETURN
        #######################################################

        if metrics[0] > highest_return:
            # print(highest_return)
            highest_return = metrics[0]
            # print(metrics[0])
            best_return = ef.portfolio_performance(verbose=True)
            latest_prices = get_latest_prices(port_df)
            weights = cleaned_weights
            da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=1000)
            allocation, leftover = da.lp_portfolio()
            best_port = cleaned_weights
            stocks = port

        print(highest_return)
        last_stock = port[0] # SETS LAST ETF FOR KNOWING THE ITERATION TIMES

    print("######### BEST PORTFOLIO #########")
    print(stocks)
    print(best_port)
    print('Expected annual return: {} %'.format(round(best_return[0]*100,2)))
    print('Annual volatility: {} %'.format(round(best_return[0]*100,2)))
    print('Sharpe Ratio: {}'.format(round(best_return[0]*100,2)))
    print('Discrete Allocation: ', allocation)
    print('Funds remaing: ${:.2f}'.format(leftover))
    print("""
    
    
    """)
    print(run_times) # PRINTS THE RUN TIMES OF ITERATIONS FOR NEW ETF START 
    execution_time = (time.time() - function_time) # CALCULATES EXECUTION TIME
    print("Execution Time for Script: {} sec".format(round(execution_time, 2)))

# RUNS THE 2 PROCESSES
if __name__ == '__main__':
    # freeze_support()
    process_1 = multiprocessing.Process(target=portOpt, args=(etfs, 5, first_process))
    process_2 = multiprocessing.Process(target=portOpt, args=(etfs, 5, second_process))
    
    process_1.start()
    process_2.start()
    
    process_1.join()
    process_2.join()



##############################################################################
## This is the data about run times from first script run that took 34 minutes
##############################################################################

# ['XLE took 0.13492369651794434 sec to start', 
# 'XLF took 410.19258737564087 sec to start', 
# 'XLU took 616.2740490436554 sec to start', 
# 'XLI took 872.9125480651855 sec to start', 
# 'GDX took 1124.6930992603302 sec to start', 
# 'XLK took 1285.2598481178284 sec to start', 
# 'XLV took 1400.3425860404968 sec to start', 
# 'XLP took 1493.922125339508 sec to start', 
# 'XLB took 1563.764530658722 sec to start', 
# 'XOP took 1628.3061652183533 sec to start', 
# 'IYR took 1672.984756231308 sec to start', 
# 'XHB took 1700.365168094635 sec to start', 
# 'ITB took 1719.5674049854279 sec to start', 
# 'VNQ took 1732.1381072998047 sec to start', 
# 'GDXJ took 1740.282814025879 sec to start', 
# 'IYE took 1744.993224143982 sec to start', 
# 'OIH took 1747.5899469852448 sec to start', 
# 'XME took 1749.4680421352386 sec to start', 
# 'XRT took 1750.1885492801666 sec to start', 
# 'SMH took 1750.4778926372528 sec to start']


times = pd.Series([0.13492, 410.192, 616.274, 872.912, 1124.69, 1285.25, 1400.34, 1493.92, 1563.76, 1628.30, 1672.98, 1700.36, 1719.56, 1732.13, 1740.2, 1744.99, 1747.58, 1749.46, 1750.18, 1750.47])

print(type(times))
change = times.diff()

print(change)

plt.plot(['XLE', 'XLF','XLU','XLI','GDX','XLK','XLV','XLP','XLB','XOP','IYR','XHB','ITB','VNQ','GDXJ','IYE','OIH','XME', 'XRT', 'SMH'], change)

plt.show()