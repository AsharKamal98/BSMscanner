# Import other files
from UserInput import *
from DerivedInput import *
import DataHandling as DH

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




def PlotGrid(data_type1, data_type2, plot_seperate_constr, fig_name):
    """
    Add info
    """

    data = DH.ReadFiles(data_type1, data_type2, plot_seperate_constr, print_summary=False)
    
    if plot_seperate_constr:
        dct = {1.0 : "U", 2.0 : "HBS", 3.0 : "STU"}
        palette={1.0 : "green", 2.0 : "blue", 3.0 : "red"}
    else:
        dct = {0.0 : "Neg", 1.0 : "Pos"}
        palette={0.0 : "red", 1.0 : "blue"}

    #new_labels = DH.ConvertLabels(dct, data[:,-1]) 
    #new_data = np.concatenate((data[:,:-1], new_labels.reshape(-1,1)), axis=1)[:10]

    df_plot = pd.DataFrame(data, columns = series_free_param.tolist() + ["Constraints"])

    sns.set_style("whitegrid");
    sns.pairplot(df_plot, hue="Constraints", palette=palette, plot_kws={"s": 2})

    subprocess.run(["mkdir", "-p", "Figures"])
    print("Creating Plot: {}".format(fig_name), "See Figures directory")
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
