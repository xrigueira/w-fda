import numpy as np
import fitter as ft
import pandas as pd
import random as rd
import seaborn as sns
import pingouin as pg

from scipy import stats
from kneed import KneeLocator
from matplotlib import pyplot as plt
plt.style.use('ggplot')
from statsmodels.distributions.empirical_distribution import ECDF

# This website has representation tools for bivariate distributions
# https://seaborn.pydata.org/tutorial/distributions.html#empirical-cumulative-distributions

def is_even(num):
    return num % 2 == 0

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

def empirical_cdf(data):

    ecdf = ECDF(data)

    x = np.linspace(min(data), max(data), num=len(data))
    y = ecdf(x)

    # Plot the results
    # plt.step(x, y)
    # plt.show()

    # x is the data that we need to calculate the distance between cdf (data and generated)

    return x, y

def clean_data(ranstate):
    
    """This function generates clean data from scratch with a gaussian or
    sinusoidal process"""
    import skfda as fda
    from skfda.exploratory.visualization import MagnitudeShapePlot

    # https://fda.readthedocs.io/en/latest/auto_examples/plot_magnitude_shape_synthetic.html#sphx-glr-auto-examples-plot-magnitude-shape-synthetic-py

    # Generate a synthetic dataset
    random_state = np.random.RandomState(ranstate)
    n_samples = 200

    fd = fda.datasets.make_gaussian_process(
        n_samples=n_samples,
        n_features=100,
        cov=fda.misc.covariances.Exponential(),
        mean=lambda t: 4 * t,
        random_state=random_state
    )

    # Sinusoidal data
    fd = fda.datasets.make_sinusoidal_process(
        n_samples=n_samples,
        n_features=100,
        start=0,
        stop=100,
        period=1,
        phase_mean=0,
        phase_std=1,
        amplitude_mean=0,
        amplitude_std=1,
        error_std=0,
        random_state=random_state
    )

    # Plot the data generated
    labels = [0] * n_samples
    # fd.plot(group=labels)

    # Plot MSPlot
    msplot = MagnitudeShapePlot(fd)
    # msplot.plot()

    # Extract mag shape values
    mag = msplot.points[:, 0]
    shape = msplot.points[:, 1]

    # Clean mag and shape
    mag = list(filter(lambda x: x <= (np.percentile(mag, 95)), mag))
    mag = list(filter(lambda x: x >= (np.percentile(mag, 5)), mag))
    shape = list((filter(lambda x: x <= (np.percentile(shape, 90)), shape)))

    # Save mag and shape for further processing
    varname = 'Gen'
    np.save(f'Database/{varname}_mag.npy', mag, allow_pickle=False, fix_imports=False)
    np.save(f'Database/{varname}_shape.npy', shape, allow_pickle=False, fix_imports=False)

def outDec_testing(varName, clean, use_empirical):
    
    """This function is to be used for testing and proving the well functioning of
    outDec. For that reason it implements the clean option, which allows checking
    if the methods is working correctly when there are no outliers in the data"""

    # Load the mag and shape data
    mag = np.load(f'Data_mag_shape/{varName}_mag.npy')
    shape = np.load(f'Data_mag_shape/{varName}_shape.npy')
    
    if clean == True:
        mag_upper, mag_lower = np.percentile(mag, 75), np.percentile(mag, 25)
        mag = list(filter(lambda x: x <= mag_upper, mag))
        mag = list(filter(lambda x: x >= mag_lower, mag))
        shape = list(filter(lambda x: x <= (np.percentile(shape, 50)), shape))
    
    # Find the distribution of the data
    # fitter_test(mag) -> norminvgauss
    # fitter_test(shape) -> exponential

    # Get the cumulative distribution function (cfd) of mag
    a, b, loc, scale = stats.norminvgauss.fit(mag)
    mag_cdf = stats.norminvgauss.cdf(mag, a, b, loc, scale)

    # Option to use the empirical distribution
    if use_empirical == True:
        mag_cdf, y = empirical_cdf(mag)

    # Generate trimmed norminvgauss data
    mag_gen = stats.norminvgauss.rvs(a, b, size=len(mag)*10)
    
    # Make sure the data does not have disproportionate tails
    mag_trimmed = []
    for i in mag_gen:
        item = rd.choice(mag_gen)
        if (item > np.percentile(mag_gen, 1)) and (item < np.percentile(mag_gen, 99)):
            mag_trimmed.append(item)
        
        if len(mag_trimmed) == len(mag):
            break

    # Get the cdf of mag_trimmed
    mag_gen_cdf = stats.norminvgauss.cdf(mag_trimmed, a, b)
    
    # Plot the cdf of mag and x
    fig = plt.figure(figsize=(7, 5))
    ax = plt.axes()
    if use_empirical == True:
        ax = plt.step(mag_cdf, y)
    else:
        ax = sns.lineplot(x=mag, y=mag_cdf, legend="auto")
    ax = sns.lineplot(x=mag_trimmed, y=mag_gen_cdf, legend="auto")
    ax.legend(['Process data', 'Clean data'])
    ax.set(xlabel='Magnitude', ylabel='p', title='Magnitude cumulative distribution function')
    plt.show()
    
    # Get the cdf of shape
    loc, scale = stats.expon.fit(shape)
    shape_cdf = stats.expon.cdf(shape, loc, scale)

    # Option to use the empirical distribution
    if use_empirical == True:
        shape_cdf, y = empirical_cdf(shape)
    
    # Generate exponential data
    shape_gen = stats.expon.rvs(size=len(shape)*10)
    
    # Make sure the data does not have disproportionate tails
    shape_trimmed = []
    for i in shape_gen:
        item = rd.choice(shape_gen)
        if (item < np.percentile(shape_gen, 98)):
            shape_trimmed.append(item)
        
        if len(shape_trimmed) == len(shape):
            break

    # Get the cdf of shape_trimmed
    shape_gen_cdf = stats.expon.cdf(shape_trimmed)
    
    # Plot the cdf of shape and shape_trimmed
    fig = plt.figure(figsize=(7, 5))
    ax = plt.axes()
    if use_empirical == True:
        ax = plt.step(shape_cdf, y)
    else:
        ax = sns.lineplot(x=shape, y=shape_cdf, legend="auto")
    ax = sns.lineplot(x=shape_trimmed, y=shape_gen_cdf, legend="auto")
    ax.legend(['Process data', 'Clean data'])
    ax.set(xlabel='Shape', ylabel='p', title='Shape cumulative distribution function')
    plt.show()
    
    # Get the distance (in the horizontal axis) between the two cdfs
    distance_mag = sum(abs(np.array(sorted(mag_trimmed)) - np.array(sorted(mag))))
    distance_shape = sum(abs(np.array(sorted(shape_trimmed)) - np.array(sorted(shape))))
    print('Distance between cdfs mag', distance_mag)
    print('Distance between cdfs shape', distance_shape)
    
    # Get the contribution of the tails to the total distance
    distance_tails_mag = sum(abs(np.array(sorted(mag_trimmed)) - np.array(sorted(mag)))[:int(0.05 * len(mag))]) + sum(abs(np.array(sorted(mag_trimmed)) - np.array(sorted(mag)))[-int(0.05 * len(mag)):])
    distance_tails_shape = sum(abs(np.array(sorted(shape_trimmed)) - np.array(sorted(shape)))[-int(0.1 * len(shape)):])
    print('Distance between tails mag', distance_tails_mag)
    print('Distance between tails shape', distance_tails_shape)

    contribution_tails_mag = np.round((distance_tails_mag/distance_mag) * 100, 4)
    contribution_tails_shape = np.round((distance_tails_shape/distance_shape) * 100, 4)
    print(f'Aportación mag {contribution_tails_mag}%')
    print(f'Aportación shape {contribution_tails_shape}%')
    
    # Now define if aportación mag or shape are above 90%, then there outliers in the
    # data. Otherwise there aren't outliers.
    outliers_in_data = False

    if (contribution_tails_mag >= 50) or (contribution_tails_shape >= 50):
        outliers_in_data = True
    
    return outliers_in_data

def outDec(mag, shape):
    
    """This function analyzes if there are outliers in the data"""
    
    # Get the empirial cdf of mag and shape
    # mag_cdf, y_mag = empirical_cdf(mag)
    # shape_cdf, y_shape = empirical_cdf(shape)
    
    # # Plot the empirical cdf of mag
    # fig, axes = plt.subplots(1, 1, figsize=(7, 5))
    # plt.step(mag_cdf, y_mag, label='Magnitude', axes=axes)
    # axes.legend(loc='best')
    # axes.set(xlabel='Magnitude', ylabel='p', title='Magnitude cumulative distribution function')
    # plt.show()
    
    # # Plot the empirial cdf of shape
    # fig, axes = plt.subplots(1, 1, figsize=(7, 5))
    # plt.step(shape_cdf, y_shape, label='Shape', axes=axes)
    # axes.legend(loc='best')
    # axes.set(xlabel='Shape', ylabel='p', title='Shape cumulative distribution function')
    # plt.show()

    # Convert arrays to list so I can use pop() and sort them
    mag, shape = list(sorted(mag)), list(sorted(shape))

    # Get the distance between the extreme points in mag and in shape
    distance_mag, distance_shape = abs(mag[-1] - mag[0]), abs(shape[-1] - shape[0])
    
    # Get mag crushing data
    mag_crushed = []
    for i in range(len(mag)-1):
        if is_even(i) == True:
            mag.pop(-1)
            distance_mag_updated = abs(mag[-1] - mag[0])

        elif is_even(i) == False:
            mag.pop(0)
            distance_mag_updated = abs(mag[-1] - mag[0])
        
        mag_crushed.append(np.round((distance_mag_updated/distance_mag) * 100, 3))
    
    # Get shape crushed data
    shape_crushed = []
    for i in range(len(shape)-1):
        shape.pop(-1)
        distance_shape_updated = abs(shape[-1] - shape[0])
        
        shape_crushed.append(np.round((distance_shape_updated/distance_shape) * 100, 3))
    
    # # Plot the crushed mag and shape
    # plt.plot(mag_crushed, np.arange(0, len(mag_crushed)), label = 'mag')
    # plt.plot(shape_crushed, np.arange(0, len(shape_crushed)), label = 'shape')
    # plt.legend(loc='best')
    # plt.show()

    # Get the point where the elbow/knee starts flattening
    kl_mag = KneeLocator(mag_crushed, np.arange(0, len(mag_crushed)), curve='convex', direction='decreasing')
    kl_mag_point = kl_mag.knee
    # kl_mag.plot_knee()

    kl_shape = KneeLocator(shape_crushed, np.arange(0, len(shape_crushed)), curve='convex', direction='decreasing')
    kl_shape_point = kl_shape.knee
    # kl_shape.plot_knee()
    # plt.show()

    print('Mag knee:', kl_mag_point, 'Shape knee:', kl_shape_point)

    # Define wether there are outliers or not in the data based on the value of the knee/elbow points
    if (kl_mag_point is None) and (kl_shape_point is None):
        outliers_in_data = False
    else:
        if (kl_mag_point is None):
            kl_mag_point = 100
        if (kl_shape_point is None):
            kl_shape_point = 100
        if (kl_mag_point <= 15) or (kl_shape_point <= 15):
            outliers_in_data = True
        else:
            outliers_in_data = False
    
    return outliers_in_data

def outDec2_testing(varName, clean):
    
    """This function is to be used for testing the new version of outDec
    and proving its well functioning. For that reason it implements the 
    clean option, which allows checking if the methods is working correctly 
    when there are no outliers in the data"""
    
    # Load the mag and shape data
    mag, shape = np.load(f'Data_mag_shape/{varName}_mag.npy'), np.load(f'Data_mag_shape/{varName}_shape.npy')

    if clean == True:
        mag_upper, mag_lower = np.percentile(mag, 75), np.percentile(mag, 25)
        mag = list(filter(lambda x: x <= mag_upper, mag))
        mag = list(filter(lambda x: x >= mag_lower, mag))
        shape = list(filter(lambda x: x <= (np.percentile(shape, 50)), shape))
    
    # Get the empirial cdf of mag and shape
    # mag_cdf, y_mag = empirical_cdf(mag)
    # shape_cdf, y_shape = empirical_cdf(shape)
    
    # # Plot the empirical cdf of mag
    # fig, axes = plt.subplots(1, 1, figsize=(7, 5))
    # plt.step(mag_cdf, y_mag, label='Magnitude', axes=axes)
    # axes.legend(loc='best')
    # axes.set(xlabel='Magnitude', ylabel='p', title='Magnitude cumulative distribution function')
    # plt.show()
    
    # # Plot the empirial cdf of shape
    # fig, axes = plt.subplots(1, 1, figsize=(7, 5))
    # plt.step(shape_cdf, y_shape, label='Shape', axes=axes)
    # axes.legend(loc='best')
    # axes.set(xlabel='Shape', ylabel='p', title='Shape cumulative distribution function')
    # plt.show()
    
    # Convert arrays to list so I can use pop() and sort them
    mag, shape = list(sorted(mag)), list(sorted(shape))
    
    # Get the distance between the extreme points in mag and in shape
    distance_mag, distance_shape = abs(mag[-1] - mag[0]), abs(shape[-1] - shape[0])
    
    # Get mag crushing data
    mag_crushed = []
    for i in range(len(mag)-1):
        if is_even(i) == True:
            mag.pop(-1)
            distance_mag_updated = abs(mag[-1] - mag[0])

        elif is_even(i) == False:
            mag.pop(0)
            distance_mag_updated = abs(mag[-1] - mag[0])
        
        mag_crushed.append(np.round((distance_mag_updated/distance_mag) * 100, 3))
    
    # Get shape crushed data
    shape_crushed = []
    for i in range(len(shape)-1):
        shape.pop(-1)
        distance_shape_updated = abs(shape[-1] - shape[0])
        
        shape_crushed.append(np.round((distance_shape_updated/distance_shape) * 100, 3))
    
    # Plot the crushed mag and shape
    plt.plot(mag_crushed, np.arange(0, len(mag_crushed)), label = 'mag')
    plt.plot(shape_crushed, np.arange(0, len(shape_crushed)), label = 'shape')
    plt.legend(loc='best')
    plt.show()

    # Get the point where the elbow/knee starts flattening
    kl_mag = KneeLocator(mag_crushed, np.arange(0, len(mag_crushed)), curve='convex', direction='decreasing')
    kl_mag_point = kl_mag.knee
    kl_mag.plot_knee()

    kl_shape = KneeLocator(shape_crushed, np.arange(0, len(shape_crushed)), curve='convex', direction='decreasing')
    kl_shape_point = kl_shape.knee
    kl_shape.plot_knee()
    plt.show()

    print('Mag knee:', kl_mag_point, 'Shape knee:', kl_shape_point)

    # Define wether there are outliers or not in the data based on the value of the knee/elbow points
    if (kl_mag_point is None) and (kl_shape_point is None):
        outliers_in_data = False
    else:
        if (kl_mag_point is None):
            kl_mag_point = 100
        if (kl_shape_point is None):
            kl_shape_point = 100
        if (kl_mag_point <= 15) or (kl_shape_point <= 15):
            outliers_in_data = True
        else:
            outliers_in_data = False
        
    return outliers_in_data
    
    
if __name__ == '__main__':

    for i in range(1):
        clean_data(ranstate=i)
        varName = 'pH'
        clean = False # If True it will remove the outliers
        use_empirical = False

        # outliers_in_data = outDec_testing(varName=varName, clean=clean, use_empirical=use_empirical)
        
        outliers_in_data = outDec2_testing(varName=varName, clean=clean)

        print(f'IFF {clean} != {outliers_in_data} -> the program works as expected')



