from scipy.optimize import minimize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def opti_mkv(sample, target):

    mean_returns = [sample[col].mean() for col in sample.columns.values]
    cov_matrix = sample.cov()

    def volatility(weights):
        return np.sqrt(np.dot(weights.T,np.dot(cov_matrix.values,weights)) )
        
    def min_ret(weights): 
        return np.dot(weights.T, [sample[col].mean() for col in sample.columns.values] )
    
    def sharpe_ratio(weights):
        return -min_ret(weights)/volatility(weights)

    if target == "min_var":
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        function = volatility
    elif target == "max_sharpe":
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        function = sharpe_ratio
    else:
        constraints = ({'type': 'eq', 'fun': lambda x: min_ret(x) - target}, {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        function = volatility

    bounds = [(-1,1) for i in range(0,len(mean_returns))]
    param_start = [1/len(mean_returns) for i in range(0,len(mean_returns))]
    opti_method = 'SLSQP'
    results = minimize(function, param_start, method=opti_method, bounds=bounds, constraints=constraints)

    return results.x


def plot_eff_frontier(sample):

    random_comb = pd.DataFrame(columns=["ret","vol"])
    k = np.random.rand(len(sample.columns) )
    weights_random = k / sum(k)

    for i in range(1000):
        k = np.random.rand(len(sample.columns) )
        weights_random = k / sum(k)
        vol = np.sqrt(np.dot(weights_random.T,np.dot(sample.cov().values,weights_random)) )*np.sqrt(252)
        ret = np.dot(weights_random.T, [sample[col].mean() for col in sample.columns.values] )*252
        random_comb = random_comb.append({"ret":ret,"vol":vol}, ignore_index=True)

    eff_comb = pd.DataFrame(columns=["ret","vol"])
    target_returns = np.linspace(sample.mean().min(),sample.mean().max(),50)


    for stp_ret in target_returns:
        weights = opti_mkv(sample, stp_ret)

        vol = np.sqrt(np.dot(weights.T,np.dot(sample.cov().values,weights)) )*np.sqrt(252)
        ret = np.dot(weights.T, [sample[col].mean() for col in sample.columns.values] )*252
        eff_comb = eff_comb.append({"ret":ret,"vol":vol}, ignore_index=True)

    plt.plot(eff_comb["vol"], eff_comb["ret"],'r--', color='red')
    plt.scatter(random_comb["vol"], random_comb["ret"],marker='.', color='black')

    min_y = eff_comb["ret"].min() - (eff_comb["ret"].max() - eff_comb["ret"].min())*0.1
    max_y = eff_comb["ret"].max() + (eff_comb["ret"].max() - eff_comb["ret"].min())*0.1
    min_x = eff_comb["vol"].min() - (eff_comb["vol"].max() - eff_comb["vol"].min())*0.1
    max_x = eff_comb["vol"].max() + (eff_comb["vol"].max() - eff_comb["vol"].min())*0.1
    plt.ylim(min_y,max_y)
    plt.xlim(min_x,max_x)
    plt.ylabel('mean')
    plt.xlabel('std')  
    plt.show()

