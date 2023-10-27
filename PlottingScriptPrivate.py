# Import other files
from UserInput import *
from DerivedInput import *
import DataHandling as DH
import Network as NW
import PlottingScript as PS

# Import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

import subprocess

#import sys
#import random as rand
#import math
#import numpy as np
#import glob
#import os
#import matplotlib
#import matplotlib.cm as cm
#import matplotlib.patches as mpatches
#from matplotlib.ticker import AutoMinorLocator
#from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
#import matplotlib.colors as colors

#import ast

#import matplotlib.patches as mpatches
#import matplotlib.gridspec as gridspec

#from sklearn.preprocessing import StandardScaler, FunctionTransformer
#from scipy import stats
#from scipy.stats import gaussian_kde

#import glob

#import matplotlib.ticker as ticker

#minorLocator=AutoMinorLocator()
#plt.rcParams['font.family'] = 'serif'
#plt.rcParams['axes.linewidth'] = 1.5
#colorsbasis=['#00A0B0','#6A4A3C','#CC333F','#EB6841','#EDC951','#A3A948','#B3E099','#5165E9','#B1B59D']
#-------------------------------------------------#

mpl.rcParams['text.usetex']=True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{bm}'
plt.rcParams.update({'font.size': 20})
mpl.rcParams["legend.framealpha"] = 1.0
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams["figure.figsize"] = [8.5, 5.5]



def ReadData(data_type1, data_type2):   # Change name
    """
    Add info
    """

    DH.ReadFiles(data_type1, data_type2)
    return


def PlotData(data_type1, data_type2):   # Change name
    """
    #Add info
    """

    DH.ReadFiles(data_type1, data_type2)
    if data_type1==1:
        PS.PlotTData(data_type2, plot_seperate_constr=False, fig_name="TrainingDataPlot.png")
    elif data_type1==3:
        PS.PlotFData(data_type2, fig_name="FinalDataPlot.png")

    return


def PlotFDataSTESM(data_type2, fig_name):
    """
    #Add info
    """
    
    # Read data and store into pandas data frame without labels
    free_param_lists = DH.ReadFreeParams(data_type1=3)
    fixed_param_lists = DH.ReadFixedParams(data_type1=3)
    df_plot = pd.DataFrame(np.hstack((free_param_lists, fixed_param_lists)), columns = series_free_param.tolist() + series_fixed_param.tolist())

    fig, axes = plt.subplots(1,2, figsize=(13,5))
    fig.tight_layout()
    color_map = "terrain"
    

    s = axes[0].scatter(np.float64(df_plot['lam5']), np.float64(df_plot['lam6']), c=np.float64(df_plot["lam7"]), cmap=color_map, s=10, edgecolors='none', rasterized=True,label='_Hidden label', marker="o")
    axes[0].set_xlabel(r'$\lambda_1$', fontweight='bold', fontsize=25)
    axes[0].set_ylabel(r'$\lambda_2$', fontweight='bold', fontsize=25)
    axes[0].set_xlim(-2.5, 2.5)
    axes[0].set_ylim(-7.5, 7.5)

    colorb1 = plt.colorbar(s, ax=axes[0], pad=0.025, fraction=0.1)
    colorb1.ax.locator_params(nbins=5)
    colorb1.set_label(r'$\lambda_3$', labelpad=8, fontsize=20)
    colorb1.ax.tick_params(labelsize=20)

    
    s = axes[1].scatter(np.float64(df_plot['mN1']), np.float64(df_plot['mN2']), c=np.float64(df_plot["mC"]), cmap=color_map, s=10, edgecolors='none', rasterized=True,label='_Hidden label', marker="o")
    axes[1].set_xlabel(r'$m_{\mathrm{N_1}} \,\, \mathrm{[GeV]}$', fontweight='bold', fontsize=25)
    axes[1].set_ylabel(r'$m_{\mathrm{N_2}} \,\, \mathrm{[GeV]}$', fontweight='bold', fontsize=25)
    axes[1].set_xlim(200, 1000)
    axes[1].set_ylim(200, 1000)

    colorb2 = plt.colorbar(s, ax=axes[1], pad=0.025, fraction=0.1)
    colorb2.ax.locator_params(nbins=5)
    colorb2.set_label(r'$m_{\mathrm{H^{\pm}}} \,\, \mathrm{[GeV]}$', labelpad=8, fontsize=20)
    colorb2.ax.tick_params(labelsize=20)
    

    for i in range(2):
        axes[i].grid(which='minor', color='lightgrey', linestyle='--')
        axes[i].grid(which='major', color='grey', linestyle='-')
        axes[i].tick_params(labelsize=25)
        axes[i].set_axisbelow(True)


    plt.tight_layout()
    plt.savefig("Figures/{}".format(fig_name), bbox_inches="tight")#, dpi=800)
    print("Plot {}".format(fig_name), "See Figures directory")


def PlotFDataTHDM(data_type2, fig_name):
    """
    #Add info
    """
    
    # Read data and store into pandas data frame without labels
    free_param_lists = DH.ReadFreeParams(data_type1=3)
    fixed_param_lists = DH.ReadFixedParams(data_type1=3)
    df_plot = pd.DataFrame(np.hstack((free_param_lists, fixed_param_lists)), columns = series_free_param.tolist() + series_fixed_param.tolist())

    fig, axes = plt.subplots(1,2, figsize=(13,5))
    fig.tight_layout()
    color_map = "terrain"
    

    s = axes[0].scatter(np.float64(df_plot['mA']), np.float64(df_plot['mC']), c=np.float64(df_plot["mH"]), cmap=color_map, s=10, edgecolors='none', rasterized=True,label='_Hidden label', marker="o")
    axes[0].set_xlabel(r'$m_{\mathrm{A}} \,\, \mathrm{[GeV]}$', fontweight='bold', fontsize=25)
    axes[0].set_ylabel(r'$m_{\mathrm{H^{\pm}}} \,\, \mathrm{[GeV]}$', fontweight='bold', fontsize=25)
    #axes[0].set_xlim(-2.5, 2.5)
    #axes[0].set_ylim(-7.5, 7.5)

    colorb1 = plt.colorbar(s, ax=axes[0], pad=0.025, fraction=0.1)
    colorb1.ax.locator_params(nbins=5)
    colorb1.set_label(r'$m_{\mathrm{H}} \,\, \mathrm{[GeV]}$', labelpad=8, fontsize=20)
    colorb1.ax.tick_params(labelsize=20)

    s = axes[1].scatter(np.float64(df_plot['lam1']), np.float64(df_plot['lam2']), c=np.float64(df_plot["mH"]), cmap=color_map, s=10, edgecolors='none', rasterized=True,label='_Hidden label', marker="o")
    axes[1].set_xlabel(r'$\lambda_1$', fontweight='bold', fontsize=25)
    axes[1].set_ylabel(r'$\lambda_2$', fontweight='bold', fontsize=25)
    axes[1].set_xlim(0, 4.5)
    axes[1].set_ylim(0, 1.3)

    colorb1 = plt.colorbar(s, ax=axes[1], pad=0.025, fraction=0.1)
    colorb1.ax.locator_params(nbins=5)
    colorb1.set_label(r'$m_{\mathrm{H}} \,\, \mathrm{[GeV]}$', labelpad=8, fontsize=20)
    colorb1.ax.tick_params(labelsize=20)


    #s = axes[1].scatter(np.float64(df_plot['lam3']), np.float64(df_plot['lam4']), c=np.float64(df_plot["lam5"]), cmap=color_map, s=10, edgecolors='none', rasterized=True,label='_Hidden label', marker="o")
    #axes[1].set_xlabel(r'$\lambda_1$', fontweight='bold', fontsize=25)
    #axes[1].set_ylabel(r'$\lambda_2$', fontweight='bold', fontsize=25)
    ##axes[1].set_xlim(0, 4.5)
    ##axes[1].set_ylim(0, 1.3)

    #colorb1 = plt.colorbar(s, ax=axes[1], pad=0.025, fraction=0.1)
    #colorb1.ax.locator_params(nbins=5)
    #colorb1.set_label(r'$m_{\mathrm{H}} \,\, \mathrm{[GeV]}$', labelpad=8, fontsize=20)
    #colorb1.ax.tick_params(labelsize=20)

    
    for i in range(2):
        axes[i].grid(which='minor', color='lightgrey', linestyle='--')
        axes[i].grid(which='major', color='grey', linestyle='-')
        axes[i].tick_params(labelsize=25)
        axes[i].set_axisbelow(True)


    plt.tight_layout()
    plt.savefig("Figures/{}".format(fig_name), bbox_inches="tight")#, dpi=800)
    print("Plot {}".format(fig_name), "See Figures directory")



#PlotFDataTHDM(data_type2="collider", fig_name="THDM.png")
#PlotFDataSTESM(data_type2="both", fig_name="STESM.png")
#PlotData(data_type1=3, data_type2='collider')
#ReadData(data_type1=3, data_type2='both')
