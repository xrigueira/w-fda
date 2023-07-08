# This file contains statistical test to prove that our data
# does not follow a p-dimensional stationary Gaussian distribution

def fitter_test(data):

    # distributions = ['alpha', 'anglit', 'arcsine', 'argus', 
    #     'beta', 'betaprime', 'bradford', 'burr', 'burr12', 
    #     'cauchy', 'chi', 'chi2', 'cosine', 'crystalball', 
    #     'dgamma', 'dweibull', 
    #     'erlang', 'expon', 'exponnorm', 'exponpow', 'exponweib', 
    #     'f', 'fatiguelife', 'fisk', 'foldcauchy', 'foldnorm', 
    #     'gamma', 'gausshyper', 'genexpon', 'genextreme', 'gengamma', 'genhalflogistic', 'genhyperbolic', 'geninvgauss', 'genlogistic', 'gennorm', 'genpareto', 'gilbrat', 'gompertz', 'gumbel_l', 'gumbel_r', 
    #     'halfcauchy', 'halfgennorm', 'halflogistic', 'halfnorm', 'hypsecant', 
    #     'invgamma', 'invgauss', 'invweibull', 
    #     'johnsonsb', 'johnsonsu', 
    #     'kappa3', 'kstwo', 'kstwobign', 
    #     'laplace', 'laplace_asymmetric', 'levy', 'levy_l', 'loggamma', 'logistic', 'loglaplace', 'lognorm', 'loguniform', 'lomax', 
    #     'maxwell', 'mielke', 'moyal',
    #     'nakagami', 'ncf', 'norm', 'norminvgauss', 
    #     'pareto', 'pearson3', 'powerlaw', 'powerlognorm', 'powernorm', 
    #     'rayleigh', 'reciprocal', 'rice', 
    #     'semicircular', 'skewcauchy', 'skewnorm', 
    #     't', 'truncexpon', 'truncnorm', 'tukeylambda', 
    #     'uniform', 
    #     'vonmises', 'vonmises_line', 
    #     'wald', 'wrapcauchy']

    distributions = ['alpha', 'arcsine', 'beta', 'betaprime', 'burr', 'cauchy', 'expon', 'f', 'fisk', 'foldcauchy', 'gamma', 'genexpon', 'genextreme', 'geninvgauss', 'genpareto', 'gilbrat', 'gumbel_r', 'halfcauchy', 'halfgennorm', 'halfnorm', 'invgamma', 'invgauss', 'invweibull', 'kappa3', 'ksone', 'laplace', 'lognorm', 'loguniform', 'lomax', 'ncf', 'norm', 'norminvgauss', 'pareto', 'powerlognorm', 'rayleigh', 'recipinvgauss', 'reciprocal', 'skewcauchy', 'truncexpon', 'wald']

    f = ft.Fitter(data, distributions=distributions) 
    f.fit()
    
    print(f.summary())
    
    plt.show()

def henze_zirkler_test():
    # https://online.stat.psu.edu/stat505/book/export/html/636
    # https://www.geeksforgeeks.org/how-to-perform-multivariate-normality-tests-in-python/
    
    # Read the databases into a list of dataframes (on the date and value columns)
    dbs = [pd.read_csv('Database/Caudal_pro.csv', delimiter=';').iloc[:, 0:2],
        pd.read_csv('Database/Conductividad_pro.csv', delimiter=';').iloc[:, 0:2],
        pd.read_csv('Database/Nitratos_pro.csv', delimiter=';').iloc[:, 0:2],
        pd.read_csv('Database/Oxigeno disuelto_pro.csv', delimiter=';').iloc[:, 0:2],
        pd.read_csv('Database/pH_pro.csv', delimiter=';').iloc[:, 0:2],
        pd.read_csv('Database/Pluviometria_pro.csv', delimiter=';').iloc[:, 0:2],
        pd.read_csv('Database/Temperatura_pro.csv', delimiter=';').iloc[:, 0:2],
        pd.read_csv('Database/Turbidez_pro.csv', delimiter=';').iloc[:, 0:2]
    ]
    
    # Initialize the result with the first dataframe
    result = dbs[0]
    
    # Loop over the rest of the dataframes and merge them one by one
    for i in range(1,len(dbs)):
        result = pd.merge(result, dbs[i], on='date', how='inner', suffixes=("", f"_{i}"))
    
    # Test
    hz, p, normal = pg.multivariate_normality(result.iloc[:10000, 1:], alpha=0.05)

    # If p>0.05, them it is normal
    print('hz=', hz, 'p=', p, 'normal=', normal)

def stationary_test():
    
    """Implements the Dickey-Fuller test"""
    # (Step 3) https://www.geeksforgeeks.org/how-to-check-if-time-series-data-is-stationary-with-python/
    from statsmodels.tsa.stattools import adfuller

    data = pd.read_csv('database/Turbidez_pro.csv', delimiter=';')
    data = data.value

    res = adfuller(data)

    print('Augmneted Dickey_fuller Statistic: %f' % res[0]) # If p < 0.05 is stationary
    print('p-value: %f' % res[1])
