import numpy as np
import pandas as pd
import random as rd

from kneed import KneeLocator
from matplotlib import pyplot as plt
plt.style.use('ggplot')
from statsmodels.distributions.empirical_distribution import ECDF

# This website has representation tools for bivariate distributions
# https://seaborn.pydata.org/tutorial/distributions.html#empirical-cumulative-distributions

def is_even(num):
    return num % 2 == 0

def empirical_cdf(data):

    ecdf = ECDF(data)

    x = np.linspace(min(data), max(data), num=len(data))
    y = ecdf(x)

    # Plot the results
    # plt.step(x, y)
    # plt.show()

    return x, y


def outDec(magnitude, shape, amplitude):
    
    """This function analyzes if there are outliers in the data"""
    
    # Get the empirial cdf of mag and shape
    magnitude_cdf, y_magnitude = empirical_cdf(magnitude)
    shape_cdf, y_shape = empirical_cdf(shape)
    amplitude_cdf, y_amplitude = empirical_cdf(amplitude)
    
    # Plot the empirical cdf of mag
    fig, axes = plt.subplots(1, 1, figsize=(7, 5))
    plt.step(magnitude_cdf, y_magnitude, label='Magnitude', axes=axes)
    axes.legend(loc='best')
    axes.set(xlabel='Magnitude', ylabel='p', title='Magnitude cumulative distribution function')
    plt.show()
    
    # Plot the empirial cdf of shape
    fig, axes = plt.subplots(1, 1, figsize=(7, 5))
    plt.step(shape_cdf, y_shape, label='Shape', axes=axes)
    axes.legend(loc='best')
    axes.set(xlabel='Shape', ylabel='p', title='Shape cumulative distribution function')
    plt.show()
    
    # Plot the empirial cdf of shape
    fig, axes = plt.subplots(1, 1, figsize=(7, 5))
    plt.step(amplitude_cdf, y_amplitude, label='Amplitude', axes=axes)
    axes.legend(loc='best')
    axes.set(xlabel='Amplitude', ylabel='p', title='Amplitude cumulative distribution function')
    plt.show()

    # Convert arrays to list so I can use pop() and sort them
    magnitude, shape, amplitude = list(sorted(magnitude)), list(sorted(shape)), list(sorted(amplitude))

    # Get the distance between the extreme points in mag and in shape
    distance_magnitude, distance_shape, distance_amplitude = abs(magnitude[-1] - magnitude[0]), abs(shape[-1] - shape[0]), abs(amplitude[-1] - amplitude[0])
    
    # # Get mag crushing data
    # mag_crushed = []
    # for i in range(len(mag)-1):
    #     if is_even(i) == True:
    #         mag.pop(-1)
    #         distance_mag_updated = abs(mag[-1] - mag[0])

    #     elif is_even(i) == False:
    #         mag.pop(0)
    #         distance_mag_updated = abs(mag[-1] - mag[0])
        
    #     mag_crushed.append(np.round((distance_mag_updated/distance_mag) * 100, 3))
    
    # # Get shape crushed data
    # shape_crushed = []
    # for i in range(len(shape)-1):
    #     shape.pop(-1)
    #     distance_shape_updated = abs(shape[-1] - shape[0])
        
    #     shape_crushed.append(np.round((distance_shape_updated/distance_shape) * 100, 3))
    
    # # Plot the crushed mag and shape
    # plt.plot(mag_crushed, np.arange(0, len(mag_crushed)), label = 'mag')
    # plt.plot(shape_crushed, np.arange(0, len(shape_crushed)), label = 'shape')
    # plt.legend(loc='best')
    # plt.show()

    # # Get the point where the elbow/knee starts flattening
    # kl_mag = KneeLocator(mag_crushed, np.arange(0, len(mag_crushed)), curve='convex', direction='decreasing')
    # kl_mag_point = kl_mag.knee
    # # kl_mag.plot_knee()

    # kl_shape = KneeLocator(shape_crushed, np.arange(0, len(shape_crushed)), curve='convex', direction='decreasing')
    # kl_shape_point = kl_shape.knee
    # kl_shape.plot_knee()
    # plt.show()

    # print('Mag knee:', kl_mag_point, 'Shape knee:', kl_shape_point)

    # # Define wether there are outliers or not in the data based on the value of the knee/elbow points
    # if (kl_mag_point is None) and (kl_shape_point is None):
    #     outliers_in_data = False
    # else:
    #     if (kl_mag_point is None):
    #         kl_mag_point = 100
    #     if (kl_shape_point is None):
    #         kl_shape_point = 100
    #     if (kl_mag_point <= 15) or (kl_shape_point <= 15):
    #         outliers_in_data = True
    #     else:
    #         outliers_in_data = False
    
    # return outliers_in_data

def outDec2_testing(varName, clean):
    
    """This function is to be used for testing the new version of outDec
    and proving its well functioning. For that reason it implements the 
    clean option, which allows checking if the methods are working correctly 
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



