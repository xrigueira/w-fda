import random
import numpy as np
import skfda as fda
import pandas as pd
import plotly.graph_objects as go

from matplotlib.lines import Line2D
from matplotlib import pyplot as plt
plt.style.use('ggplot')
from matplotlib import colors as mcolors
from matplotlib import ticker
from pearson import pearson_correlation
from outDec import outDec

"""This file implemented the functional data analysis of the pre-processed data"""

def flatter(list):
    return [item for sublits in list for item in sublits]

def labeler(varname):

    """This function is just to label the plots correctly for the research papers."""

    if varname == 'Amonio':
        label_title = r'$NH_4$'
        label_y_axis = r'$NH_4$ ' + r'$(m*g/L)$'
    elif varname == 'Conductividad' or varname == 'Conductividad_p1' or varname == 'Conductividad_p2' or varname == 'Conductividad_p' or varname == 'Conductividad_val':
        label_title = r'Conductivity'
        label_y_axis = r'Conductivity ' r'$(\mu*S/cm)$'
    elif varname == 'Nitratos':
        label_title = r'$NO_{3^-}$'
        label_y_axis = r'$NO_{3^-}$ ' +r'$(m*g/L)$'
    elif varname == 'Oxigeno disuelto' or varname == 'Oxigeno disuelto_p' or varname == 'Oxigeno disuelto_p1' or varname == 'Oxigeno disuelto_p2':
        label_title = r'$O_2$'
        label_y_axis = r'$O_2$ ' r'$(m*g/L)$'
    elif varname == 'pH' or varname == 'pH_p1' or varname == 'pH_p2' or varname == 'pH_p3' or varname == 'pH_p':
        label_title = r'pH'
        label_y_axis = r'pH'
    elif varname == 'Temperatura' or varname == 'Temperatura_p1' or varname == 'Temperatura_p2' or varname == 'Temperatura_p':
        label_title = r'Temperature'
        label_y_axis = r'Temperature ' +u'(\N{DEGREE SIGN}C)'
    elif varname == 'Temperatura_rio_p' or varname == 'Temperatura_rio_p1' or varname == 'Temperatura_rio_p2':
        label_title = r'River water remperature '
        label_y_axis = r'Temperature ' +u'(\N{DEGREE SIGN}C)'
    elif varname == 'Caudal' or varname == 'Caudal_p' or varname == 'Caudal_p1' or varname == 'Caudal_p2':
        label_title = r'Flow'
        label_y_axis = r'Flow ' + r'($m^3/s$)'
    elif varname == "Turbidez" or varname == 'Turbidez_p' or varname == 'Turbidez_p1' or varname == 'Turbidez_p2':
        label_title = r'Turbidity'
        label_y_axis = r'Turbidity ' + r'(NTU)'
    elif varname == "Pluviometria" or varname == 'Pluviometria_p1' or varname == 'Pluviometria_p2' or varname == 'Pluviometria_p':
        label_title = r'Pluviometry'
        label_y_axis = r'Pluviometry ' + r'(mm)'
    elif varname == "Nivel" or varname == 'Nivel_p1' or varname == 'Nivel_p2' or varname == 'Nivel_p':
        label_title = r'Water level'
        label_y_axis = r'Water level ' + r'(m)'

    return label_title, label_y_axis

def dataGrid(varname, datamatrix, timestep, timeframe):
    
    # Define object FDataGrid
    if timestep == '15 min':

        if timeframe == 'a':
            gridPoints = list(range(2976))
        elif timeframe == 'b':
            gridPoints = list(range(672))
        elif timeframe == 'c':
            gridPoints = list(range(96))

    elif timestep == '1 day':

        if timeframe == 'a':
            gridPoints = list(range(31))
        elif timeframe == 'b':
            gridPoints = list(range(7))
    
    functionalData = fda.FDataGrid(data_matrix=datamatrix, grid_points=gridPoints)
    
    # Plot the data
    # label_title, label_y_axis = labeler(varname=varname)
    
    # functionaldata.plot() # Plain plot
    
    # fig, axes = plt.subplots()
    # functionalData.plot(axes=axes)
    # axes.set_title(f'Functional data ' + label_title)
    # axes.set_xlabel('Time')
    # axes.set_ylabel(label_y_axis)
    # plt.show()
    
    return gridPoints, functionalData

def smoothing(varname, datamatrix, gridpoints, functionaldata):
    
    # Calculate Fourier smoothing and the number of basis functions
    dataMatrixFlat = flatter(datamatrix)

    for nBasis in range(1, 256, 1):

        basis = fda.representation.basis.Fourier(n_basis=nBasis) # fitting the data through Fourier
        smoothedData = functionaldata.to_basis(basis) # Change class to FDataBasis

        evaluatingPoints = smoothedData.evaluate(np.array(gridpoints), derivative=0) # get the corresponding values in the resulting curve
        evaluatingPoints = evaluatingPoints.tolist() # convert from array to list        

        flat2evaluatingPoints = flatter(evaluatingPoints)
        flatevaluatingPoints = flatter(flat2evaluatingPoints)

        # rho, p = pearsonr(np.array(dataMatrixFlat), np.array(flatevaluatingPoints))
        rho = pearson_correlation(dataMatrixFlat, flatevaluatingPoints)

        if rho >= 0.99:
            break
        else:
            continue

    print('Number of basis functions: ', nBasis, 'and rho: ', rho)
    
    smoothedDataGrid = smoothedData.to_grid(grid_points=gridpoints) # Convert to FDataGrid for further needs
    
    # # Plot the smoothed data
    # label_title, label_y_axis = labeler(varname=varname)
    
    # # smoothedData.plot() # Plain plot
    
    # fig, axes = plt.subplots()
    # smoothedData.plot(axes=axes)
    # axes.set_title(f'Smoothed data ' + label_title)
    # axes.set_xlabel('Time')
    # axes.set_ylabel(label_y_axis)
    # plt.show()
    
    return smoothedData, smoothedDataGrid

# cutofffactor de 1.5 de acuedo con la definición de boxplot: https://fda.readthedocs.io/en/latest/modules/exploratory/autosummary/skfda.exploratory.outliers.BoxplotOutlierDetector.html#skfda.exploratory.outliers.BoxplotOutlierDetector
def boxplot(varname, timestamps, cutoff, contamination, smootheddata, smootheddatagrid):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Apply the outlier detection method
    funcBoxplot = fda.exploratory.visualization.Boxplot(smootheddatagrid, depth_method=fda.exploratory.depth.ModifiedBandDepth(), factor=cutoff, axes=ax1)
    funcBoxplot.show_full_outliers = True
    
    # Extract the outliers detected
    outliersBoxplot = funcBoxplot.outliers
    
    # Plot the results of the boxplot analysis
    color, outliercolor = 0.3, 0.7
    label_title, label_y_axis = labeler(varname=varname)

    ax1.set_title(f'Functional boxplot ' + label_title)
    funcBoxplot.plot()

    # Apply the results to the functional data plot
    smootheddata.plot(group=funcBoxplot.outliers.astype(int), group_colors=funcBoxplot.colormap([color, outliercolor]), group_names=['No outliers', 'Outliers'], axes=ax2)
    ax2.set_title(f'Functional data ' + label_title)
    ax2.set_xlabel('Time')
    ax2.set_ylabel(label_y_axis)
    # fig.savefig(f'images/{varname}_boxplot_{contamination}.png')
    plt.show()
    
    # Get the dates of the outliers
    outliers = [i for i,j in zip(timestamps, outliersBoxplot) if j == 1]
    
    print('outliers boxplot:', np.round(len(outliers)/len(timestamps), 3), outliers)   
    
    # Plotly implementation to display the results in a web browser
    # dataPly = []
    # for i in (smootheddatagrid.data_matrix).tolist():
    #     dataPly.append(flatter(i))
    
    # dfPlotly = pd.DataFrame.from_records(dataPly)
    # dfPlotly = dfPlotly.transpose()
    # dfPlotly.columns = timestamps
    
    # outliersBoxplotP = list(outliersBoxplot)
    
    # for i, j in enumerate(outliersBoxplotP):
    #     if j == True:
    #         outliersBoxplotP[i] = 'red'
    #     elif j == False:
    #         outliersBoxplotP[i] = 'blue'
    
    # colorDict = {}
    # for i, j in zip(timestamps, outliersBoxplotP):
    #     colorDict.update({i: j})
    
    # fig = go.Figure()
    
    # for col in dfPlotly.columns:
    #     fig.add_trace(go.Scatter(x=dfPlotly.index, y=dfPlotly[col], mode='lines', name=col, marker_color=colorDict[col]))
    
    # fig.show()
    
    plt.close(fig=fig)

    return outliers

def msplot_DaiGenton(varname, timestamps, depth, cutoff, contamination, smootheddata, smootheddatagrid):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Apply the outlier detection method
    funcMSPlot = fda.exploratory.visualization.MagnitudeShapePlot(fdata=smootheddatagrid, multivariate_depth=depth, alpha=cutoff, axes=ax1)
    
    # Extract the outliers detected
    outliersMSPlot = funcMSPlot.outliers

    # Copy of the outliers for the control charts (currently not used)
    outliersCC_DaiGenton = list(np.copy(outliersMSPlot).astype(int))

    # Plot the results on the mag and shape graph
    color, outliercolor = 0.3, 0.7
    label_title, label_y_axis = labeler(varname=varname)

    funcMSPlot.plot()
    
    # Apply the results to the functional data plot
    smootheddata.plot(group=funcMSPlot.outliers.astype(int), group_colors=funcMSPlot.colormap([color, outliercolor]), group_names=['No outliers', 'Outliers'], axes=ax2)
    ax2.set_title(f'Functional data ' + label_title)
    ax2.set_xlabel('Time')
    ax2.set_ylabel(label_y_axis)
    # fig.savefig(f'images/{varname}_DaiGenton_{contamination}.png')
    plt.show()
    
    # Get the dates of the outliers
    outliers = [i for i,j in zip(timestamps, outliersMSPlot) if j == 1]
    
    print('outliers:', np.round(len(outliers)/len(timestamps), 3), outliers)
    
    # Plotly implementation to display the results in a web browser
    # dataPly = []
    # for i in (smootheddatagrid.data_matrix).tolist():
    #     dataPly.append(flatter(i))
    
    # dfPlotly = pd.DataFrame.from_records(dataPly)
    # dfPlotly = dfPlotly.transpose()
    # dfPlotly.columns = timestamps 
    
    # # Create color dictionary
    # outliersMSPlotP = list(outliersMSPlot)
    # for i, j in enumerate(outliersMSPlotP):
    #     if j == True:
    #         outliersMSPlotP[i] = 'red'
    #     elif j == False:
    #         outliersMSPlotP[i] = 'blue'

    # colorDict = {}
    # for i, j in zip(timestamps, outliersMSPlotP):
    #     colorDict.update({i: j})
    
    # # Graph the results
    # fig = go.Figure()
    
    # for col in dfPlotly.columns:
    #     fig.add_trace(go.Scatter(x=dfPlotly.index, y=dfPlotly[col], mode='lines', name=col, marker_color=colorDict[col]))
    
    # fig.show()
    
    plt.close(fig=fig)

    return outliers

def outliergram(varname, timestamps, depth, cutoff, contamination, smootheddata, smootheddatagrid):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Apply the outlier detection method
    outliergram = fda.exploratory.visualization.Outliergram(fdata=smootheddatagrid, axes=ax1)

    # Extract the outliers detected
    outliersOG = outliergram.outlier_detector.fit_predict(smootheddatagrid)
    # Binarize outliers
    outliersOG = np.array([0 if i == 1 else 1 for i in outliersOG])
    # These are the coordinates of the points
    mbd = outliergram.outlier_detector.mbd_ # not ordered
    mei = outliergram._mei_ordered

    # Plot the results
    color, outliercolor = 'b', 'r'
    label_title, label_y_axis = labeler(varname=varname)
    
    ax1.set_title(f'Outliergram ' + label_title)
    outliergram.plot()

    # Apply the results to the functional data plot
    ax2.set_title(f'Functional data ' + label_title)
    ax2.set_xlabel('Time')
    ax2.set_ylabel(label_y_axis)
    smootheddata.plot(group=outliersOG.astype(int), group_colors=[color, outliercolor], group_names=['No outliers', 'Outliers'], axes=ax2)
    # fig.savefig(f'images/{varname}_outliergram_{contamination}.png')
    plt.show()

    # Get the dates of the outliers
    outliers = [i for i,j in zip(timestamps, outliersOG) if j == 1]
    
    print('outliers:', np.round(len(outliers)/len(timestamps), 3), outliers)

    plt.close(fig=fig)
    
    return outliers 

def iF(varname, timestamps, depth, cutoff, contamination, smootheddata, smootheddatagrid):
    
    funcMSPlot = fda.exploratory.visualization.MagnitudeShapePlot(fdata=smootheddatagrid, multivariate_depth=depth, alpha=cutoff)
    
    # Get the outliers detected
    outliersMSPlot = funcMSPlot.outliers
    
    # Extract magnitude and shape
    mag = funcMSPlot.points[:, 0]
    shape = funcMSPlot.points[:, 1]
    
    from sklearn.ensemble import IsolationForest
    
    # Apply iF
    modeliF = IsolationForest(n_estimators=100, contamination=contamination)
    modeliF.fit(funcMSPlot.points)
    pred = modeliF.predict(funcMSPlot.points)
    probs = -1*modeliF.score_samples(funcMSPlot.points)
    
    indexiF = np.where(probs >= np.quantile(probs, (1-contamination))) # 0.875
    valuesiF = funcMSPlot.points[indexiF]
    
    indexiF = [i for i in indexiF[0]] # Convert the tuple[0] to a list
    
    # Define the colors of each point and curve inlucing outliers and nonoutliers (this is for the MSPlot)
    colors_available = list((mcolors.CSS4_COLORS).keys())
    colors_available.remove('lightgray') # Remove the possibility of using lightgray as a color for the outliers
    colors = np.copy(funcMSPlot.outliers.astype(int))
    colors[:] = 0
    colors[indexiF] = 1
    colors = ['lightgray' if i == 0 else random.choice(colors_available) for i in colors]
    
    # Define the labels of the outliers (1) and nonoutliers (0)
    labels = np.copy(funcMSPlot.outliers.astype(int))
    labels[:] = 0
    labels[indexiF] = 1
    
    # Get the dates of the outleirs
    outliers = [i for i,j in zip(timestamps, labels) if j == 1]
    print('outliers iF:', np.round(len(outliers)/len(timestamps), 3), outliers)
    
    # Plot the results
    label_title, label_y_axis = labeler(varname=varname)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Anomaly score
    sp = ax1.scatter(funcMSPlot.points[:, 0], funcMSPlot.points[:, 1], c=probs, cmap='rainbow')
    fig.colorbar(sp, ax=ax1, location='right', label='Simplified Anomaly Score')
    ax1.set_title('Isolation Forest Scores')
    ax1.set_ylabel("Shape outlyingness")
    ax1.set_xlabel("Magnitude outlyingness")
    ax1.set_facecolor("#F1F0E6")
    ax1.grid(color='w', linestyle='-', linewidth=1)
    ax1.set_axisbelow(True)
    
    # Binarized results
    ax2.scatter(funcMSPlot.points[:, 0], funcMSPlot.points[:, 1], color=colors)
    ax2.set_title("Isolation Forest Binarized")
    ax2.set_ylabel("Shape outlyingness")
    ax2.set_xlabel("Magnitude outlyingness")
    ax2.set_facecolor("#F1F0E6")
    ax2.grid(color='w', linestyle='-', linewidth=1)
    ax2.set_axisbelow(True)
    filled_marker_style = dict(marker='o', linestyle='-', markersize=13,
                        color='w',
                        markerfacecolor='r',
                        markerfacecoloralt='g',
                        markeredgecolor='w')
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='No Outliers', markerfacecolor='lightgray', markersize=13),
                Line2D([0], [0], fillstyle='right', **filled_marker_style, label='Outliers')]
    ax2.legend(handles=legend_elements, loc='best')
    
    # Adjust the colors and labels of the functional results plot
    counter = 0
    for i, e in enumerate(labels):
        if e != 0:
            labels[i] = e + counter
            counter += 1
    
    # Remove the repeated 'lightgray' from the colors list
    i = 0
    item_to_remove = 'lightgray'
    while i < len(colors):
        if colors[i] == item_to_remove:
            colors.pop(i)
        else:
            i += 1
    
    # Insert 'lightgray' in the first zero
    first_zero = False
    for i, e in enumerate(labels):
        if e == 0 and first_zero == False:
            colors.insert(i, 'lightgray')
            first_zero = True
    
    # Functional results
    ax3.set_title(f'Functional data ' + label_title)
    ax3.set_xlabel('Time')
    ax3.set_ylabel(label_y_axis)
    smootheddata.plot(group=labels, group_colors=colors, axes=ax3)
    # fig.savefig(f'images/{varname}_iF_{contamination}.png')
    
    plt.tight_layout()
    plt.show()

    return outliers

def MCD(varname, timestamps, depth, cutoff, contamination, smootheddata, smootheddatagrid):
    
    funcMSPlot = fda.exploratory.visualization.MagnitudeShapePlot(fdata=smootheddatagrid, multivariate_depth=depth, alpha=cutoff)
    
    # Get the outliers detected
    outliersMSPlot = funcMSPlot.outliers
    
    # Extract magnitude and shape
    mag = funcMSPlot.points[:, 0]
    shape = funcMSPlot.points[:, 1]
    
    from sklearn.covariance import MinCovDet
    
    # Apply MCD
    modelMinCov = MinCovDet(random_state=0)
    modelMinCov.fit(funcMSPlot.points)
    mahaDistance = modelMinCov.mahalanobis(funcMSPlot.points)
    
    indexMinCov = np.where(mahaDistance >= np.quantile(mahaDistance, (1-contamination))) # 0.875
    valuesMinCov = funcMSPlot.points[indexMinCov]
    
    indexMinCov = [i for i in indexMinCov[0]]
    
    # Define the colors of each point and curve inlucing outliers and nonoutliers (this is for the MSPlot)
    colors_available = list((mcolors.CSS4_COLORS).keys())
    colors_available.remove('lightgray') # Remove the possibility of using lightgray as a color for the outliers
    colors = np.copy(funcMSPlot.outliers.astype(int))
    colors[:] = 0
    colors[indexMinCov] = 1
    colors = ['lightgray' if i == 0 else random.choice(colors_available) for i in colors]
    
    # Define the labels of the outliers (1) and nonoutliers (0)
    labels = np.copy(funcMSPlot.outliers.astype(int))
    labels[:] = 0
    labels[indexMinCov] = 1

    # Get the dates of the outleirs
    outliers = [i for i,j in zip(timestamps, labels) if j == 1]
    print('outliers MCD:', np.round(len(outliers)/len(timestamps), 3), outliers) 
    
    # Plot the results
    label_title, label_y_axis = labeler(varname=varname)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Mahalanobis distance
    sp = ax1.scatter(funcMSPlot.points[:, 0], funcMSPlot.points[:, 1], c=mahaDistance, s=50, cmap='bwr')
    fig.colorbar(sp, ax=ax1, location='right', label='Mahalanobis Distance')
    ax1.set_ylabel("Shape outlyingness")
    ax1.set_xlabel("Magnitude outlyingness")
    ax1.set_title("Minimum Covariance Determinant Score")
    ax1.set_facecolor("#F1F0E6")
    ax1.grid(color='w', linestyle='-', linewidth=1)
    ax1.set_axisbelow(True)
    
    # Binarized results
    ax2.scatter(funcMSPlot.points[:, 0], funcMSPlot.points[:, 1], color=colors)
    ax2.set_title("Minimum Covariance Determinant Binarized")
    ax2.set_ylabel("Shape outlyingness")
    ax2.set_xlabel("Magnitude outlyingness")
    ax2.set_facecolor("#F1F0E6")
    ax2.grid(color='w', linestyle='-', linewidth=1)
    ax2.set_axisbelow(True)
    filled_marker_style = dict(marker='o', linestyle='-', markersize=13,
                        color='w',
                        markerfacecolor='r',
                        markerfacecoloralt='g',
                        markeredgecolor='w')
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='No Outliers', markerfacecolor='lightgray', markersize=13),
                Line2D([0], [0], fillstyle='right', **filled_marker_style, label='Outliers')]
    ax2.legend(handles=legend_elements, loc='best')
    
    # Adjust the colors and labels of the functional results plot
    counter = 0
    for i, e in enumerate(labels):
        if e != 0:
            labels[i] = e + counter
            counter += 1
    
    # Remove the repeated 'lightgray' from the colors list
    i = 0
    item_to_remove = 'lightgray'
    while i < len(colors):
        if colors[i] == item_to_remove:
            colors.pop(i)
        else:
            i += 1
    
    # Insert 'lightgray' in the first zero
    first_zero = False
    for i, e in enumerate(labels):
        if e == 0 and first_zero == False:
            colors.insert(i, 'lightgray')
            first_zero = True
    
    # Functional results
    ax3.set_title(f'Functional data ' + label_title)
    ax3.set_xlabel('Time')
    ax3.set_ylabel(label_y_axis)
    smootheddata.plot(group=labels, group_colors=colors, axes=ax3)
    # fig.savefig(f'images/{varname}_MCD_{contamination}.png')
    
    plt.tight_layout()
    plt.show()

    return outliers

def kNN(varname, timestamps, depth, cutoff, contamination, smootheddata, smootheddatagrid):
    
    funcMSPlot = fda.exploratory.visualization.MagnitudeShapePlot(fdata=smootheddatagrid, multivariate_depth=depth, alpha=cutoff)
    
    # Get the outliers detected
    outliersMSPlot = funcMSPlot.outliers
    
    # Extract magnitude and shape
    mag = funcMSPlot.points[:, 0]
    shape = funcMSPlot.points[:, 1]

    from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
    from sklearn.model_selection import GridSearchCV

    # Apply unsupervised kNN
    modelkNN = NearestNeighbors(n_neighbors=20, algorithm='ball_tree')
    modelkNN.fit(funcMSPlot.points)
    distances, indexes = modelkNN.kneighbors(funcMSPlot.points)
    
    indexkNN = np.where(distances.mean(axis=1) >= np.quantile(distances.mean(axis=1), (1-contamination)))
    valueskNN = funcMSPlot.points[indexkNN]

    indexkNN = [i for i in indexkNN[0]] # Convert the tuple[0] to a list

    # Define the colors of each point and curve inlucing outliers and nonoutliers (this is for the MSPlot)
    colors_available = list((mcolors.CSS4_COLORS).keys())
    colors_available.remove('lightgray') # Remove the possibility of using lightgray as a color for the outliers
    colors = np.copy(funcMSPlot.outliers.astype(int))
    colors[:] = 0
    colors[indexkNN] = 1
    colors = ['lightgray' if i == 0 else random.choice(colors_available) for i in colors]
    
    labels = np.copy(funcMSPlot.outliers.astype(int))
    labels[:] = 0
    labels[indexkNN] = 1

    # Get the dates of the outleirs
    outliers = [i for i,j in zip(timestamps, labels) if j == 1]

    # Plot the results
    label_title, label_y_axis = labeler(varname=varname)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Anomaly score
    sp = ax1.scatter(funcMSPlot.points[:, 0], funcMSPlot.points[:, 1], c=distances.mean(axis=1), cmap='rainbow')
    fig.colorbar(sp, ax=ax1, location='right', label='k Distances')
    ax1.set_title('k-Nearest Neighbors Scores')
    ax1.set_ylabel("Shape outlyingness")
    ax1.set_xlabel("Magnitude outlyingness")
    ax1.set_facecolor("#F1F0E6")
    ax1.grid(color='w', linestyle='-', linewidth=1)
    ax1.set_axisbelow(True)

    # Binarized results
    ax2.scatter(funcMSPlot.points[:, 0], funcMSPlot.points[:, 1], color=colors)
    ax2.set_title("k-Nearest Neighbors Binarized")
    ax2.set_ylabel("Shape outlyingness")
    ax2.set_xlabel("Magnitude outlyingness")
    ax2.set_facecolor("#F1F0E6")
    ax2.grid(color='w', linestyle='-', linewidth=1)
    ax2.set_axisbelow(True)
    filled_marker_style = dict(marker='o', linestyle='-', markersize=13,
                        color='w',
                        markerfacecolor='r',
                        markerfacecoloralt='g',
                        markeredgecolor='w')
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='No Outliers', markerfacecolor='lightgray', markersize=13),
                Line2D([0], [0], fillstyle='right', **filled_marker_style, label='Outliers')]
    ax2.legend(handles=legend_elements, loc='best')

    # Adjust the colors and labels of the functional results plot
    counter = 0
    for i, e in enumerate(labels):
        if e != 0:
            labels[i] = e + counter
            counter += 1
    
    # Remove the repeated 'lightgray' from the colors list
    i = 0
    item_to_remove = 'lightgray'
    while i < len(colors):
        if colors[i] == item_to_remove:
            colors.pop(i)
        else:
            i += 1
    
    # Insert 'lightgray' in the first zero
    first_zero = False
    for i, e in enumerate(labels):
        if e == 0 and first_zero == False:
            colors.insert(i, 'lightgray')
            first_zero = True
    
    # Functional results
    ax3.set_title(f'Functional data ' + label_title)
    ax3.set_xlabel('Time')
    ax3.set_ylabel(label_y_axis)
    colormap = plt.cm.get_cmap('seismic')
    smootheddata.plot(group=labels, group_colors=colors, axes=ax3)
    # fig.savefig(f'images/{varname}_kNN_{contamination}.png')
    
    plt.tight_layout()
    plt.show()

    return outliers

def CBLOF(varname, timestamps, depth, cutoff, contamination, smootheddata, smootheddatagrid):

    funcMSPlot = fda.exploratory.visualization.MagnitudeShapePlot(fdata=smootheddatagrid, multivariate_depth=depth, alpha=cutoff)
    
    # Get the outliers detected
    outliersMSPlot = funcMSPlot.outliers
    
    # Extract magnitude and shape
    mag = funcMSPlot.points[:, 0]
    shape = funcMSPlot.points[:, 1]
    
    from pyod.models import cblof
    
    # Apply CBLOF
    modelCBLOF = cblof.CBLOF(n_clusters=2, contamination=contamination)
    modelCBLOF.fit(funcMSPlot.points)
    probs = modelCBLOF.decision_function(funcMSPlot.points)

    indexCBLOF = np.where(probs >= np.quantile(probs, (1-contamination)))
    valuesCBLOF = funcMSPlot.points[indexCBLOF]
    
    indexCBLOF = [i for i in indexCBLOF[0]] # Convert the tuple[0] to a list
    
    # Define the colors of each point and curve inlucing outliers and nonoutliers (this is for the MSPlot)
    colors_available = list((mcolors.CSS4_COLORS).keys())
    colors_available.remove('lightgray') # Remove the possibility of using lightgray as a color for the outliers
    colors = np.copy(funcMSPlot.outliers.astype(int))
    colors[:] = 0
    colors[indexCBLOF] = 1
    colors = ['lightgray' if i == 0 else random.choice(colors_available) for i in colors]
    
    labels = np.copy(funcMSPlot.outliers.astype(int))
    labels[:] = 0
    labels[indexCBLOF] = 1

    # Get the dates of the outleirs
    outliers = [i for i,j in zip(timestamps, labels) if j == 1]
    
    # Plot the results
    label_title, label_y_axis = labeler(varname=varname)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Anomaly score
    sp = ax1.scatter(funcMSPlot.points[:, 0], funcMSPlot.points[:, 1], c=probs, cmap='rainbow')
    fig.colorbar(sp, ax=ax1, location='right', label='Anomaly Score')
    ax1.set_title('Cluster-Based LOF Scores')
    ax1.set_ylabel("Shape outlyingness")
    ax1.set_xlabel("Magnitude outlyingness")
    ax1.set_facecolor("#F1F0E6")
    ax1.grid(color='w', linestyle='-', linewidth=1)
    ax1.set_axisbelow(True)
    
    # Binarized results
    ax2.scatter(funcMSPlot.points[:, 0], funcMSPlot.points[:, 1], color=colors)
    ax2.set_title("CBLOF Binarized")
    ax2.set_ylabel("Shape outlyingness")
    ax2.set_xlabel("Magnitude outlyingness")
    ax2.set_facecolor("#F1F0E6")
    ax2.grid(color='w', linestyle='-', linewidth=1)
    ax2.set_axisbelow(True)
    filled_marker_style = dict(marker='o', linestyle='-', markersize=13,
                        color='w',
                        markerfacecolor='r',
                        markerfacecoloralt='g',
                        markeredgecolor='w')
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='No Outliers', markerfacecolor='lightgray', markersize=13),
                Line2D([0], [0], fillstyle='right', **filled_marker_style, label='Outliers')]
    ax2.legend(handles=legend_elements, loc='best')
    
    # Adjust the colors and labels of the functional results plot
    counter = 0
    for i, e in enumerate(labels):
        if e != 0:
            labels[i] = e + counter
            counter += 1
    
    # Remove the repeated 'lightgray' from the colors list
    i = 0
    item_to_remove = 'lightgray'
    while i < len(colors):
        if colors[i] == item_to_remove:
            colors.pop(i)
        else:
            i += 1
    
    # Insert 'lightgray' in the first zero
    first_zero = False
    for i, e in enumerate(labels):
        if e == 0 and first_zero == False:
            colors.insert(i, 'lightgray')
            first_zero = True
    
    # Functional results
    ax3.set_title(f'Functional data ' + label_title)
    ax3.set_xlabel('Time')
    ax3.set_ylabel(label_y_axis)
    smootheddata.plot(group=labels, group_colors=colors, axes=ax3)
    # fig.savefig(f'images/{varname}_CBLOF_{contamination}.png')
    
    plt.tight_layout()
    plt.show()
    
    return outliers

def OCSVM(varname, timestamps, depth, cutoff, contamination, smootheddata, smootheddatagrid):
    
    funcMSPlot = fda.exploratory.visualization.MagnitudeShapePlot(fdata=smootheddatagrid, multivariate_depth=depth, alpha=cutoff)
    
    # Get the outliers detected
    outliersMSPlot = funcMSPlot.outliers
    
    # Extract magnitude and shape
    mag = funcMSPlot.points[:, 0]
    shape = funcMSPlot.points[:, 1]
    
    from sklearn.svm import OneClassSVM
    
    # Apply OCSVM
    modelOCSVM = OneClassSVM(nu=contamination, kernel='poly', gamma='auto')
    modelOCSVM.fit(funcMSPlot.points)
    probs = modelOCSVM.score_samples(funcMSPlot.points)
    
    indexOCSVM = np.where(probs >= np.quantile(probs, (1-contamination)))
    valuesOCSVM = funcMSPlot.points[indexOCSVM]
    
    indexOCSVM = [i for i in indexOCSVM[0]] # Convert the tuple[0] to a list
    
    # Define the colors of each point and curve inlucing outliers and nonoutliers (this is for the MSPlot)
    colors_available = list((mcolors.CSS4_COLORS).keys())
    colors_available.remove('lightgray') # Remove the possibility of using lightgray as a color for the outliers
    colors = np.copy(funcMSPlot.outliers.astype(int))
    colors[:] = 0
    colors[indexOCSVM] = 1
    colors = ['lightgray' if i == 0 else random.choice(colors_available) for i in colors]

    # Define the labels of the outliers (1) and nonoutliers (0)
    labels = np.copy(funcMSPlot.outliers.astype(int))
    labels[:] = 0
    labels[indexOCSVM] = 1
    
    # Get the dates of the outleirs
    outliers = [i for i,j in zip(timestamps, labels) if j == 1]
    
    # Plot the results
    label_title, label_y_axis = labeler(varname=varname)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Anomaly score
    sp = ax1.scatter(funcMSPlot.points[:, 0], funcMSPlot.points[:, 1], c=probs, cmap='rainbow')
    fig.colorbar(sp, ax=ax1, location='right', label='Anomaly Score')
    ax1.set_title('One class SVM Scores')
    ax1.set_ylabel("Shape outlyingness")
    ax1.set_xlabel("Magnitude outlyingness")
    ax1.set_facecolor("#F1F0E6")
    ax1.grid(color='w', linestyle='-', linewidth=1)
    ax1.set_axisbelow(True)
    
    # Binarized results
    ax2.scatter(funcMSPlot.points[:, 0], funcMSPlot.points[:, 1], color=colors)
    ax2.set_title("OCSVM Binarized")
    ax2.set_ylabel("Shape outlyingness")
    ax2.set_xlabel("Magnitude outlyingness")
    ax2.set_facecolor("#F1F0E6")
    ax2.grid(color='w', linestyle='-', linewidth=1)
    ax2.set_axisbelow(True)
    filled_marker_style = dict(marker='o', linestyle='-', markersize=13,
                        color='w',
                        markerfacecolor='r',
                        markerfacecoloralt='g',
                        markeredgecolor='w')
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='No Outliers', markerfacecolor='lightgray', markersize=13),
                Line2D([0], [0], fillstyle='right', **filled_marker_style, label='Outliers')]
    ax2.legend(handles=legend_elements, loc='best')
    
    # Adjust the colors and labels of the functional results plot
    counter = 0
    for i, e in enumerate(labels):
        if e != 0:
            labels[i] = e + counter
            counter += 1
    
    # Remove the repeated 'lightgray' from the colors list
    i = 0
    item_to_remove = 'lightgray'
    while i < len(colors):
        if colors[i] == item_to_remove:
            colors.pop(i)
        else:
            i += 1
    
    # Insert 'lightgray' in the first zero
    first_zero = False
    for i, e in enumerate(labels):
        if e == 0 and first_zero == False:
            colors.insert(i, 'lightgray')
            first_zero = True
    
    # Functional results
    ax3.set_title(f'Functional data ' + label_title)
    ax3.set_xlabel('Time')
    ax3.set_ylabel(label_y_axis)
    smootheddata.plot(group=labels, group_colors=colors, axes=ax3)
    # fig.savefig(f'images/{varname}_iF_{contamination}.png')
    
    plt.tight_layout()
    plt.show()
    
    return outliers

def iFMCD(varname, timestamps, depth, cutoff, contamination, smootheddata, smootheddatagrid):

    label_title, label_y_axis = labeler(varname=varname)

    funcMSPlot = fda.exploratory.visualization.MagnitudeShapePlot(fdata=smootheddatagrid, multivariate_depth=depth, alpha=cutoff)
    
    # Get the outliers detected
    outliersMSPlot = funcMSPlot.outliers
    
    # Extract magnitude and shape
    mag = funcMSPlot.points[:, 0]
    shape = funcMSPlot.points[:, 1]

    # Save mag and shape for further processing
    np.save(f'Database/{varname}_mag.npy', mag, allow_pickle=False, fix_imports=False)
    np.save(f'Database/{varname}_shape.npy', shape, allow_pickle=False, fix_imports=False)

    # Check if there are outliers in the data
    outliers_in_data = outDec(mag=mag, shape=shape)
    
    # Double filter
    if outliers_in_data == True:

        from sklearn.ensemble import IsolationForest
        from sklearn.covariance import MinCovDet

        # Apply IF
        modeliF = IsolationForest(n_estimators=100, contamination=contamination)
        modeliF.fit(funcMSPlot.points)
        pred = modeliF.predict(funcMSPlot.points)
        probs = -1*modeliF.score_samples(funcMSPlot.points)

        indexiF = np.where(probs >= np.quantile(probs, (1-contamination))) # 0.875
        valuesiF = funcMSPlot.points[indexiF]

        indexiF = [i for i in indexiF[0]]

        # Apply MCD
        modelMinCov = MinCovDet(random_state=0)
        modelMinCov.fit(funcMSPlot.points)
        mahaDistance = modelMinCov.mahalanobis(funcMSPlot.points)

        indexMinCov = np.where(mahaDistance >= np.quantile(mahaDistance, (1-contamination))) # 0.875
        valuesMinCov = funcMSPlot.points[indexMinCov]

        indexMinCov = [i for i in indexMinCov[0]]

        # OR option
        indexFinal = list(dict.fromkeys(indexiF + indexMinCov))

        # AND option
        # indexFinal = [i for i in indexiF if (i in indexiF) and (i in indexMinCov)]

        # Define the colors of each point and curve inlucing outliers and nonoutliers (this is for the MSPlot)
        colors_available = list((mcolors.CSS4_COLORS).keys())
        colors_available.remove('lightgray') # Remove the possibility of using lightgray as a color for the outliers
        colors = np.copy(funcMSPlot.outliers.astype(int))
        colors[:] = 0
        colors[indexFinal] = 1
        colors = ['lightgray' if i == 0 else random.choice(colors_available) for i in colors]

        # Define the labels of the outliers (1) and nonoutliers (0)
        labels = np.copy(funcMSPlot.outliers.astype(int))
        labels[:] = 0
        labels[indexFinal] = 1

        # Copy of the labels list for the control charts
        outliersCCiFMCD = list(labels.copy())

        # Plot the results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.scatter(funcMSPlot.points[:, 0], funcMSPlot.points[:, 1], color=colors)
        ax1.set_title("MS-Plot")
        ax1.set_xlabel("Magnitude outlyingness")
        ax1.set_ylabel("Shape outlyingness")
        ax1.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        filled_marker_style = dict(marker='o', linestyle='-', markersize=13,
                        color='w',
                        markerfacecolor='r',
                        markerfacecoloralt='g',
                        markeredgecolor='w')
        legend_elements = [Line2D([0], [0], marker='o', color='w', label='No Outliers', markerfacecolor='lightgray', markersize=13),
                    Line2D([0], [0], fillstyle='right', **filled_marker_style, label='Outliers')]
        ax1.legend(handles=legend_elements, loc='best')
        
        # Adjust the colors and labels of the functional results plot
        counter = 0
        for i, e in enumerate(labels):
            if e != 0:
                labels[i] = e + counter
                counter += 1

        # Remove the repeated 'lightgray' from the colors list
        i = 0
        item_to_remove = 'lightgray'
        while i < len(colors):
            if colors[i] == item_to_remove:
                colors.pop(i)
            else:
                i += 1
        
        # Insert 'lightgray' in the first zero
        first_zero = False
        for i, e in enumerate(labels):
            if e == 0 and first_zero == False:
                colors.insert(i, 'lightgray')
                first_zero = True

        # Functional results
        ax2.set_title(f'Functional data ' + label_title)
        ax2.set_xlabel('Time')
        ax2.set_ylabel(label_y_axis)
        smootheddata.plot(group=labels, group_colors=colors, axes=ax2)
        # fig.savefig(f'images/{varname}_iFMCD_{contamination}.png')
        
        # Plotly implementation to display the results on the browser
        dataPly = []
        for i in (smootheddatagrid.data_matrix).tolist():
            dataPly.append(flatter(i))
        
        dfPlotly = pd.DataFrame.from_records(dataPly)
        dfPlotly = dfPlotly.transpose()
        dfPlotly.columns = timestamps
        
        # Create color dictionary
        labelsPlotly = list(labels)
        for i, j in enumerate(labels):
            if j == 1:
                labelsPlotly[i] = 'red'
            elif j == 0:
                labelsPlotly[i] = 'blue'

        colorDict = {}
        for i, j in zip(timestamps, labelsPlotly):
            colorDict.update({i: j})
        
        # Graph the results
        fig = go.Figure()
        
        for col in dfPlotly.columns:
            fig.add_trace(go.Scatter(x=dfPlotly.index, y=dfPlotly[col], mode='lines', name=col, marker_color=colorDict[col]))
        
        # fig.show()
    
    else:

        labels = []
        outliersCCiFMCD = [0] * len(outliersMSPlot)
    
    # Get the dates of the outliers
    if outliers_in_data == True:

        outliers = [i for i,j in zip(timestamps, labels) if j == 1]
        outliersMag = [i for i,j in zip(mag, labels) if j == 1]
        outlierShape = [i for i,j in zip(shape, labels) if j == 1]

        dfOutliers = pd.DataFrame(list(zip(outliersMag, outlierShape)), index=outliers, columns=['magnitud', 'shape'])

        # Save outliers dataframe
        dfOutliers.index.name = 'timeStamps'
        dfOutliers.to_csv(f'Database/{varname}_out.csv', sep=';', encoding='utf-8', index=True, header=['magnitude', 'shape'])

        # Print results
        # print('outliers boosted:', np.round(len(outliersBoosted)/len(timestamps), 3), outliersBoosted)
        # print(dfOutliers)
        # print('Average magnitude: {} | Average shape: {}'.format(np.average(mag), np.average(shape)))

    else:

        outliers = [0] * len(outliersMSPlot)

        # Print the results
        # print('No outliers')
    
    return outliers, outliersCCiFMCD

# This is the original msplot function which works correctly. DO NOT DELETE: contains red/blue implementations of the plots
def msplot_oldversion(varname, depthname, timestamps, depth, cutoff, smootheddata, smootheddatagrid):
    
    """This function contains the old implementation os msplot, which first applies
    MSPlot Dai and Genton and then iFMCD"""

    color, outliercolor = 0.3, 0.7
    depthName = depthname

    label_title, label_y_axis = labeler(varname=varname)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    funcMSPlot = fda.exploratory.visualization.MagnitudeShapePlot(fdata=smootheddatagrid, multivariate_depth=depth, alpha=cutoff, axes=ax1)

    outliersMSPlot = funcMSPlot.outliers

    # Copy of the outliers for the control charts
    outliersCC = list(np.copy(outliersMSPlot).astype(int))
    
    funcMSPlot.plot()
    smootheddata.plot(group=funcMSPlot.outliers.astype(int), group_colors=funcMSPlot.colormap([color, outliercolor]), group_names=['No outliers', 'Outliers'], axes=ax2)
    
    # ax2.set_title(f'Outliers {depthName} depth ' + label_title)
    ax2.set_title(f'Functional weekly data ' + label_title)
    ax2.set_xlabel('Time (1 day intervals)')
    ax2.set_ylabel(label_y_axis)
    # fig.savefig(f'outliers_MSPlot_{varName}.png')
    # plt.show()
    
    # Plotly implementation to display the results in a web browser
    dataPly = []
    for i in (smootheddatagrid.data_matrix).tolist():
        dataPly.append(flatter(i))
    
    dfPlotly = pd.DataFrame.from_records(dataPly)
    dfPlotly = dfPlotly.transpose()
    dfPlotly.columns = timestamps 
    
    # Create color dictionary
    outliersMSPlotP = list(outliersMSPlot)
    for i, j in enumerate(outliersMSPlotP):
        if j == True:
            outliersMSPlotP[i] = 'red'
        elif j == False:
            outliersMSPlotP[i] = 'blue'

    colorDict = {}
    for i, j in zip(timestamps, outliersMSPlotP):
        colorDict.update({i: j})
    
    # Graph the results
    fig = go.Figure()
    
    for col in dfPlotly.columns:
        fig.add_trace(go.Scatter(x=dfPlotly.index, y=dfPlotly[col], mode='lines', name=col, marker_color=colorDict[col]))
    
    # fig.show()
    
    # Double filter
    if all(outliersMSPlot == 0) == False: # Aquí es donde tendría que cambiar la condición para que entre el resultado del nuevo método
        
        # Take the two dimensions separate
        mag = funcMSPlot.points[:, 0] # Estas líneas [252, 258] tiene que subir hasta nates del if
        shape = funcMSPlot.points[:, 1]

        # Save mag and shape for further processing
        np.save(f'Database/{varname}_mag.npy', mag, allow_pickle=False, fix_imports=False)
        np.save(f'Database/{varname}_shape.npy', shape, allow_pickle=False, fix_imports=False)

        # Implement algos
        from sklearn.cluster import KMeans
        from sklearn.ensemble import IsolationForest
        from sklearn.covariance import MinCovDet
        
        # Isolation Forest
        modeliF = IsolationForest(n_estimators=100, contamination=0.10)
        modeliF.fit(funcMSPlot.points)
        pred = modeliF.predict(funcMSPlot.points)
        probs = -1*modeliF.score_samples(funcMSPlot.points)
        
        fig, axes = plt.subplots(1, figsize=(7, 5))
        sp = axes.scatter(funcMSPlot.points[:, 0], funcMSPlot.points[:, 1], c=probs, cmap='rainbow')
        fig.colorbar(sp, label='Simplified Anomaly Score')
        axes.set_title('Isolation Forest Scores ' + label_title)
        axes.set_ylabel("Shape outlyingness")
        axes.set_xlabel("Magnitude outlyingness")
        axes.set_facecolor("#F1F0E6")
        axes.grid(color='w', linestyle='-', linewidth=1)
        axes.set_axisbelow(True)

        indexiF = np.where(probs >= np.quantile(probs, 0.875))
        valuesiF = funcMSPlot.points[indexiF]

        fig, axes = plt.subplots(1, figsize=(6, 5))
        axes.scatter(funcMSPlot.points[:, 0], funcMSPlot.points[:, 1])
        axes.scatter(valuesiF[:, 0], valuesiF[:, 1], color='r')
        axes.set_title("Isolation Forest Binarized " + label_title)
        axes.set_ylabel("Shape outlyingness")
        axes.set_xlabel("Magnitude outlyingness")
        axes.set_facecolor("#F1F0E6")
        axes.grid(color='w', linestyle='-', linewidth=1)
        axes.set_axisbelow(True)
        legend_elements = [Line2D([0], [0], marker='o', color='w', label='No Outliers', markerfacecolor='b', markersize=13),
                    Line2D([0], [0], marker='o', color='w', label='Outliers', markerfacecolor='r', markersize=13)]
        axes.legend(handles=legend_elements, loc='best')

        # Minimum Covariance Determinant
        modelMinCov = MinCovDet(random_state=0)
        modelMinCov.fit(funcMSPlot.points)
        mahaDistance = modelMinCov.mahalanobis(funcMSPlot.points)

        fig, axes = plt.subplots(1, figsize=(7, 5))
        sp = axes.scatter(funcMSPlot.points[:, 0], funcMSPlot.points[:, 1], c=mahaDistance, s=50, cmap='bwr')
        fig.colorbar(sp, label='Mahalanobis Distance')
        axes.set_ylabel("Shape outlyingness")
        axes.set_xlabel("Magnitude outlyingness")
        axes.set_title("Minimum Covariance Determinant Score " + label_title)
        axes.set_facecolor("#F1F0E6")
        axes.grid(color='w', linestyle='-', linewidth=1)
        axes.set_axisbelow(True)

        indexMinCov = np.where(mahaDistance >= np.quantile(mahaDistance, 0.875))
        valuesMinCov = funcMSPlot.points[indexMinCov]

        fig, axes = plt.subplots(1, figsize=(6, 5))
        axes.scatter(funcMSPlot.points[:, 0], funcMSPlot.points[:, 1])
        axes.scatter(valuesMinCov[:, 0], valuesMinCov[:, 1], color='r')
        axes.set_title("Minimum Covariance Determinant Binarized " + label_title)
        axes.set_ylabel("Shape outlyingness")
        axes.set_xlabel("Magnitude outlyingness")
        axes.set_facecolor("#F1F0E6")
        axes.grid(color='w', linestyle='-', linewidth=1)
        axes.set_axisbelow(True)
        legend_elements = [Line2D([0], [0], marker='o', color='w', label='No Outliers', markerfacecolor='b', markersize=13),
                            Line2D([0], [0], marker='o', color='w', label='Outliers', markerfacecolor='r', markersize=13)]
        axes.legend(handles=legend_elements, loc='best')

        indexiF = [i for i in indexiF[0]]
        indexMinCov = [i for i in indexMinCov[0]]

        # OR option
        indexFinal = indexiF + indexMinCov
        indexFinal = list(dict.fromkeys(indexFinal))

        # AND option
        # indexFinal = [i for i in indexiF if (i in indexiF) and (i in indexMinCov)]

        colors = np.copy(outliersMSPlot).astype(float)
        colors[:] = color
        colors[indexFinal] = outliercolor

        labels = np.copy(funcMSPlot.outliers.astype(int))
        labels[:] = 0
        labels[indexFinal] = 1
        
        # Copy of the labels list for the control charts
        outliersCCBoosted = list(labels.copy())
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        # ax1 = fig.add_subplot(1, 1, 1)

        ax1.scatter(funcMSPlot.points[:, 0], funcMSPlot.points[:, 1], c=funcMSPlot.colormap(colors))
        ax1.set_title("MS-Plot")
        ax1.set_xlabel("Magnitude outlyingness")
        ax1.set_ylabel("Shape outlyingness")
        ax1.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        # ax1.add_artist(ellipse)
        
        ax2.set_title(f'Functional weekly data ' + label_title)
        ax2.set_xlabel('Time (1 day intervals)')
        ax2.set_ylabel(label_y_axis)
        
        colormap = plt.cm.get_cmap('seismic')
        smootheddata.plot(group=labels, group_colors=colormap([color, outliercolor]), group_names=['No outliers', 'Outliers'], axes=ax2)
        
        # Plotly implementation to display the results on the browser
        dataPly = []
        for i in (smootheddatagrid.data_matrix).tolist():
            dataPly.append(flatter(i))
        
        dfPlotly = pd.DataFrame.from_records(dataPly)
        dfPlotly = dfPlotly.transpose()
        dfPlotly.columns = timestamps
        
        # Create color dictionary
        labelsPlotly = list(labels)
        for i, j in enumerate(labels):
            if j == 1:
                labelsPlotly[i] = 'red'
            elif j == 0:
                labelsPlotly[i] = 'blue'

        colorDict = {}
        for i, j in zip(timestamps, labelsPlotly):
            colorDict.update({i: j})
        
        # Graph the results
        fig = go.Figure()
        
        for col in dfPlotly.columns:
            fig.add_trace(go.Scatter(x=dfPlotly.index, y=dfPlotly[col], mode='lines', name=col, marker_color=colorDict[col]))
        
        fig.show()
    
    else:

        labels = []
        outliersCCBoosted = [0] * len(outliersCC)
    
    # Get the dates of the outleirs
    outliers = [i for i,j in zip(timestamps, outliersMSPlot) if j == 1]

    # print(outIntDepth)
    # print('time stamps:', timeStamps)
    # print('outliers:', np.round(len(outliers)/len(timestamps), 3), outliers)
    
    if all(outliersMSPlot == 0) == False:

        outliersBoosted = [i for i,j in zip(timestamps, labels) if j == 1]
        outliersMag = [i for i,j in zip(mag, labels) if j == 1]
        outlierShape = [i for i,j in zip(shape, labels) if j == 1]

        dfOutliers = pd.DataFrame(list(zip(outliersMag, outlierShape)), index=outliersBoosted, columns=['magnitud', 'shape'])

        dfOutliers.index.name = 'timeStamps'
        dfOutliers.to_csv(f'Database/{varname}_out.csv', sep=';', encoding='utf-8', index=True, header=['magnitude', 'shape'])
        
        # print('outliers boosted:', np.round(len(outliersBoosted)/len(timestamps), 3), outliersBoosted)
        print(dfOutliers)
        print('Average magnitude: {} | Average shape: {}'.format(np.average(mag), np.average(shape)))
    
    else:

        outliersBoosted = [0] * len(outliersMSPlot)
    
    return outliers, outliersBoosted, outliersCC, outliersCCBoosted

def functionalAnalysis(varname, depthname, datamatrix, timestamps, timestep, timeframe, depth, cutoff, contamination):
    
    gridPoints, functionalData = dataGrid(varname, datamatrix, timestep, timeframe)

    smoothedData, smoothedDataGrid = smoothing(varname, datamatrix, gridpoints=gridPoints, functionaldata=functionalData)
    
    # outliers = msplot_DaiGenton(varname, timestamps, depth, cutoff, contamination, smootheddata=smoothedData, smootheddatagrid=smoothedDataGrid)

    # outliers = outliergram(varname, timestamps, depth, cutoff, contamination, smootheddata=smoothedData, smootheddatagrid=smoothedDataGrid)
    
    # outliers = kNN(varname, timestamps, depth, cutoff, contamination, smootheddata=smoothedData, smootheddatagrid=smoothedDataGrid)

    # outliers = OCSVM(varname, timestamps, depth, cutoff, contamination, smootheddata=smoothedData, smootheddatagrid=smoothedDataGrid)
    
    # outliers = iF(varname, timestamps, depth, cutoff, contamination, smootheddata=smoothedData, smootheddatagrid=smoothedDataGrid)

    outliers, outliersCCiFMCD = iFMCD(varname, timestamps, depth, cutoff, contamination=contamination, smootheddata=smoothedData, smootheddatagrid=smoothedDataGrid)    
    
    # Old implementation
    # outliers, outliersBoosted, outliersCC, outliersCCBoosted = msplot_oldversion(varname, depthname, timestamps, depth, cutoff, smootheddata=smoothedData, smootheddatagrid=smoothedDataGrid)
    
    plt.show()
    
    return outliers