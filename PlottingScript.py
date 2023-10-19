# Import other files
from UserInput import *
from DerivedInput import *
import DataHandling as DH
import Network as NW

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




def PlotTData(data_type2, plot_seperate_constr, fig_name):
    """
    Add info
    """
    
    print("Creating Plot: {}".format(fig_name), "See Figures directory")

    data = DH.ReadFiles(data_type1=1, data_type2=data_type2, seperate_labels=plot_seperate_constr, print_summary=False)
    
    if plot_seperate_constr:
        dct = {0.0 : "BG", 1.0 : "U", 2.0 : "H", 3.0 : "STU", 4.0 : "FOPT", 5.0 : "S-FOPT", 6.0 : "D-FOPT"}
        if data_type2 == "collider":
            # 0.0="BG", 1.0="U", 2.0="H", 3.0="STU"
            palette = {"BG" : "black", "U" : "green", "H" : "blue", "STU" : "red"}
            us = {0.0 : 350, 1.0 : 100, 2.0 : 300, 3.0 : 70}
            os = None
            data  = NW.Boosting(data, under_sample=us, over_sample=os)
        elif data_type2 == "cosmic":
            # 0.0="BG", 1.0="FOPT", 2.0="S-FOPT", 3.0="D-FOPT"
            palette = {"BG" : "black", "FOPT" : "orange", "S-FOPT" : "dodgerblue", "D-FOPT" : "darkgreen"}
            data  = NW.Boosting(data, under_sample={0.0 : 10, 4.0 : 10, 5.0 : 10, 6.0 : 10}, over_sample=None)
        elif data_type2 == "both":
            palette = {"BG" : "black", "U" : "green", "H" : "blue", "STU" : "red", "BG" : "black", "FOPT" : "orange", "S-FOPT" : "dodgerblue", "D-FOPT" : "darkgreen"}
            data  = NW.Boosting(data, under_sample={0.0 : 10, 1.0 : 10, 2.0 : 10, 3.0 : 10, 4.0 : 10, 5.0 : 10, 6.0 : 10}, over_sample=None)

    else:
        
        dct = {0.0 : "Neg", 1.0 : "Pos"}
        palette={"Neg" : "red", "Pos" : "blue"}
        data  = NW.Boosting(data, under_sample={0.0 : 170, 1.0 : 170}, over_sample=None)

    df_plot = pd.DataFrame(data, columns = series_free_param.tolist() + ["Constraints"])
    df_plot["Constraints"] = df_plot["Constraints"].map(dct)

    sns.set_style("whitegrid");
    sns.pairplot(df_plot, hue="Constraints", palette=palette, plot_kws={"s": 2})

    subprocess.run(["mkdir", "-p", "Figures"])
    plt.savefig('Figures/{}'.format(fig_name))


def PlotFData(data_type2, fig_name):
    """
    Add info
    """
    
    print("Creating Plot: {}".format(fig_name), "See Figures directory")

    data = DH.ReadFiles(data_type1=3, data_type2=data_type2, seperate_labels=False, print_summary=False)
        
    dct = {0.0 : "Neg", 1.0 : "Pos"}
    palette={"Neg" : "red", "Pos" : "blue"}

    df_plot = pd.DataFrame(data, columns = series_free_param.tolist() + ["Constraints"])
    df_plot["Constraints"] = df_plot["Constraints"].map(dct)

    sns.set_style("whitegrid");
    sns.pairplot(df_plot, hue="Constraints", palette=palette, plot_kws={"s": 2})

    subprocess.run(["mkdir", "-p", "Figures"])
    plt.savefig('Figures/{}'.format(fig_name))




def f():
    data = np.c_[X,y]
    #df = pd.DataFrame(data, columns=["$\\Lambda_1$", '\Lambda_2', '\lambda_1', '\lambda_2', '\lambda_3','\lambda_6', '\lambda_7', 'm_{\mathrm{N}_1}', 'm_{\mathrm{N}_2}', 'm_{\mathrm{C}}', 'Constraints'])
    df = pd.DataFrame(data, columns=["$\\Lambda_1$", '$\\Lambda_2$', '$\\lambda_1$', '$\\lambda_2$', '$\\lambda_3$','$\\lambda_6$', '$\\lambda_7$', '$m_{\\mathrm{N}_1}$', '$m_{\\mathrm{N}_2}$', '$m_{\\mathrm{C}}$', 'Constraints'])
    name_list=["\Lambda_1", '\Lambda_2', '\lambda_1', '\lambda_2', '\lambda_3','\lambda_6', '\lambda_7', 'm_{\mathrm{N}_1}', 'm_{\mathrm{N}_2}', 'm_{\mathrm{C}}']

    if plot_dist:
        if read_data=='collider':
            param_names = df.columns.values[:-1]
            for i in range(len(param_names)):
                name = param_names[i]
                name2=name_list[i]
                df[name] = pd.to_numeric(df[name])
                if i==len(param_names)-1:
                    g = sns.displot(df, hue='Constraints', kind="kde", fill=True, bw_adjust=1, palette={"U" : 'green', "H" : 'blue', "STU" : 'red', "BG" : "black"}, hue_order=["U", "H", "STU", "BG"], x=name, legend=True, aspect=1)
                    # Increase legend size and remove title
                    legend = g._legend
                    legend.set_title("")
                    plt.setp(g._legend.get_texts(), fontsize=30)
                    #plt.legend(["U","H","STU", "BG"], frameon=True, fontsize=20)
                else:
                    g = sns.displot(df, hue='Constraints', kind="kde", fill=True, bw_adjust=1, palette={"U" : 'green', "H" : 'blue', "STU" : 'red', "BG" : "black"}, hue_order=["U", "H", "STU", "BG"], x=name, legend=False, aspect=1)

                #g.set_axis_labels(name,"", fontsize=20)
                if i<=1:
                    g.set_axis_labels(fr"${name2} \ [\mathrm{{GeV^2}}]$", "PDF", fontsize=20)
                elif i <=6:
                    g.set_axis_labels(fr"${name2}$", "PDF", fontsize=20)
                else:
                    g.set_axis_labels(fr"${name2} \ [\mathrm{{GeV}}]$", "PDF", fontsize=20)

                #g.set(yticks=[])
                plt.yticks(fontsize=11)
                plt.xticks(fontsize=11)
                #plt.title("{} Distribution".format(name), fontsize=25)
                plt.title(fr"${name2} \ \mathrm{{Distribution}}$", fontsize=25)
                #plt.show()
                plt.savefig('TempPlots/HistDist_col{}.pdf'.format(i), bbox_inches="tight")


        if read_data=='cosmic':
            param_names = df.columns.values[:-1]
            for i in range(len(param_names)):
                name = param_names[i]
                name2=name_list[i]
                df[name] = pd.to_numeric(df[name])
                if i==len(param_names)-1:
                    g = sns.displot(df, hue='Constraints', kind="kde", fill=True, bw_adjust=0.8, palette={"S-FOPT" : 'dodgerblue', "D-FOPT" : "red", "FOPT" : 'orange', "BG" : "black"}, hue_order=['FOPT', "S-FOPT", 'D-FOPT', 'BG'], x=name, legend=True, aspect=1)
                    # Increase legend size and remove title
                    legend = g._legend
                    legend.set_title("")
                    plt.setp(g._legend.get_texts(), fontsize=30)
                    #plt.legend(["FOPT","S-FOPT", "D-FOPT", "BG"], frameon=True, fontsize=20, loc='upper left')
                else:
                    g = sns.displot(df, hue='Constraints', kind="kde", fill=True, bw_adjust=0.8, palette={"S-FOPT" : 'dodgerblue', "D-FOPT" : "red", "FOPT" : 'orange', "BG" : "black"}, hue_order=['FOPT', "S-FOPT", 'D-FOPT', 'BG'], x=name, legend=False, aspect=1)

                g.set_axis_labels(fr"${name}$","", fontsize=20)
                if i<=1:
                    g.set_axis_labels(fr"${name2} \ [\mathrm{{GeV^2}}]$", "PDF", fontsize=20)
                elif i <=6:
                    g.set_axis_labels(fr"${name2}$", "PDF", fontsize=20)
                else:
                    g.set_axis_labels(fr"${name2} \ [\mathrm{{GeV}}]$", "PDF", fontsize=20)

                #g.set(yticks=[])
                plt.xticks(fontsize=11)
                plt.yticks(fontsize=11)
                #plt.title("{} Distribution".format(name), fontsize=25)
                plt.title(fr"${name2} \ \mathrm{{Distribution}}$", fontsize=25)
                #plt.show()
                plt.savefig('TempPlots/HistDist_cos{}.pdf'.format(i), bbox_inches="tight")


    else:
        sns.set_style("whitegrid");
        sns.pairplot(df, hue='Constraints', plot_kws={"s": 2})
        print("Saving plot")
        plt.savefig('{}'.format(fig_name))
        #plt.show()

    return None

def SpecialPlot(X, y, fig_name):
    name_list=["\Lambda_1", '\Lambda_2', '\lambda_1', '\lambda_2', '\lambda_3','\lambda_6', '\lambda_7', 'm_{\mathrm{N}_1}', 'm_{\mathrm{N}_2}', 'm_{\mathrm{C}}']
    data = np.c_[X,y]
    
    index = np.where(y=='DDFOPT')[0]
    print(index)
    print(X[index], y[index])
    data1 = np.c_[X[index],y[index]]
    print(data1)
    df = pd.DataFrame(data, columns=["$\\Lambda_1$", '$\\Lambda_2$', '$\\lambda_1$', '$\\lambda_2$', '$\\lambda_3$','$\\lambda_6$', '$\\lambda_7$', '$m_{\\mathrm{N}_1}$', '$m_{\\mathrm{N}_2}$', '$m_{\\mathrm{C}}$', 'Constraints'])
    df1 = pd.DataFrame(data1, columns=["$\\Lambda_1$", '$\\Lambda_2$', '$\\lambda_1$', '$\\lambda_2$', '$\\lambda_3$','$\\lambda_6$', '$\\lambda_7$', '$m_{\\mathrm{N}_1}$', '$m_{\\mathrm{N}_2}$', '$m_{\\mathrm{C}}$', 'Constraints'])
    param_names = df.columns.values[:-1]
    fig, axes = plt.subplots(9,5, figsize=(38,50))
    pltNr = 0
    #plt.rcParams['font.size'] = 28
    for i in range(10):
        nameX = param_names[i]
        nameX2 = name_list[i]
        df[nameX] = pd.to_numeric(df[nameX])
        for j in range(10):
            if j<=i:
                continue
            else:
                nameY = param_names[j]
                nameY2 = name_list[j]
                df[nameY] = pd.to_numeric(df[nameY])
                #sns.set_style("whitegrid");
                Xaxis = (pltNr // 5) 
                Yaxis = (pltNr % 5)
                if pltNr == 44:
                    leg=True
                else:
                    leg=False
                #sns.scatterplot(df1, ax=axes[Xaxis,Yaxis], x=nameX, y=nameY, hue="Constraints", legend=False, size="Constraints", sizes={"DDFOPT" : 120}, palette={"DDFOPT" : "red"})
                sns.scatterplot(df, ax=axes[Xaxis,Yaxis], x=nameX, y=nameY, hue="Constraints", legend=leg, size="Constraints", palette={"DDFOPT" : "red", "DFOPT" : "orange", "BG" : "grey"}, sizes={"DDFOPT" : 120, "DFOPT" : 15, "BG" : 15}, hue_order=['DFOPT', "BG", 'DDFOPT'])
                if pltNr == 44:
                    plt.legend(loc="upper right", fontsize="20")

                #if i<=1:
                #    g.set_axis_xlabels(fr"${nameX2} \ [\mathrm{{GeV^2}}]$", fontsize=20)
                #elif i <=6:
                #    g.set_axis_xlabels(fr"${nameX2}$", fontsize=20)
                #else:
                #    g.set_axis_xlabels(fr"${nameX2} \ [\mathrm{{GeV}}]$", fontsize=20)


                pltNr += 1
                #plt.show()

    plt.savefig('{}.png'.format(fig_name))


#PlotGrid(data_type1=1, data_type2="collider", plot_seperate_constr=False, fig_name="TestPlot.png")
