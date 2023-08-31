# IMPORT ANN libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# MATH AND OTHER
import numpy as np
import math
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd
#from operator import itemgetter


# OTHER SCRIPTS IN FRAMEWORK 
from UserInput import *
from UserInputPaths import *
import DataHandling 


"""
def PlotData(X, y, fig_name, plot_dist=False, read_data='both'):
    '''
    Plots data.
    input
    -----
    X : Array?
        (free) Input parameters
    y : array?
        Labels. 0/1 for negative/positive points. If plots_dist==True, elements are then
        Strings describing which constraints they satisfy
    fig_name : String
        Name of figure. Currently only relevant for plot_dist==False
    read_data : string
        'both', 'cosmic' or 'collider. Currently only relevant for plot_dist==True
    '''
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

"""

def Boosting(X, y, under_sample=None, over_sample=None):
    counter = Counter(y)
    print("Classes distribution before boosting", counter)

    if under_sample != None:
        undersample = RandomUnderSampler(sampling_strategy=under_sample)
        X, y = undersample.fit_resample(X,y)
        counter = Counter(y)
        print("Classes dstribution after under sampling", counter)

    #oversample = SMOTE(sampling_strategy=0.1)
    #X, y = oversample.fit_resample(X,y)
    
    if over_sample != None:
        oversample = SMOTE(sampling_strategy=over_sample) #RandomOverSampler(sampling_strategy=over_sample)
        X, y = oversample.fit_resample(X,y)
        counter = Counter(y)
        print("Class distribution after over sampling", counter)

    return X,y


def ConstructModel():
    model = keras.Sequential(
            [
                layers.Dense(60, activation="relu", name="layer1", input_shape=(10,),  kernel_regularizer=keras.regularizers.L1L2(0.001)),
                layers.Dense(60, activation="relu", name="layer2", kernel_regularizer=keras.regularizers.L1L2(0.001)),
                layers.Dense(60, activation="relu", name="layer3",  kernel_regularizer=keras.regularizers.L1L2(0.001)),
                layers.Dense(60, activation="relu", name="layer4",  kernel_regularizer=keras.regularizers.L1L2(0.001)),
                layers.Dense(60, activation="relu", name="layer5",  kernel_regularizer=keras.regularizers.L1L2(0.001)),
                layers.Dense(60, activation="relu", name="layer6",  kernel_regularizer=keras.regularizers.L1L2(0.001)),
                #layers.Dense(60, activation="relu", name="layer7",  kernel_regularizer=keras.regularizers.L1L2(0.001)),
                #layers.Dense(60, activation="relu", name="layer8",  kernel_regularizer=keras.regularizers.L1L2(0.001)),
                layers.Dense(1, activation="sigmoid", name="output_layer")
            ]
        )

    return model



def Train(model, X_trn, y_trn, X_val, y_val):
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss=keras.losses.BinaryCrossentropy(),
                metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.TruePositives(),
                        tf.keras.metrics.FalsePositives()])

    history = model.fit(
        X_trn,
        y_trn,
        batch_size=3000,
        epochs=network_epochs,
        validation_data = (X_val, y_val),
        class_weight = {0: 1.0, 1: 2.0},
        verbose=network_verbose,
        )
    return history.history, model


def PlotMetric(history,y_trn,y_val):
    keys = [item for item in history.keys()]
    #print(keys, "\n")

    num_epochs= len((history)['loss'])

    TP = np.array(history[keys[2]])
    FP = np.array(history[keys[3]])
    val_TP = np.array(history[keys[6]])
    val_FP = np.array(history[keys[7]])
    P = np.sum(y_trn)
    val_P = np.sum(y_val)


    def average(Mlist, epoch_Mlist=False):
        split=round(network_epochs/25)
        mul_Mlist = np.array_split(Mlist, split)
        avg_Mlist = np.zeros(split)
        for i in range(split):
            avg_Mlist[i] = np.sum(mul_Mlist[i])/len(mul_Mlist[i])
            if math.isnan(avg_Mlist[i]):
                avg_Mlist[i] = 0
        if epoch_Mlist:
            Elist = np.linspace(0, network_epochs, split)
            return avg_Mlist, Elist
        else:
            return avg_Mlist


    with np.errstate(divide='ignore',invalid='ignore'):
        Efficiency = np.divide(TP, TP+FP)
        val_Efficiency = np.divide(val_TP, val_TP+val_FP)
        Exhaustiveness = TP/P
        val_Exhaustiveness = val_TP/val_P

        avg_Exhaustiveness = average(Exhaustiveness)
        avg_val_Exhaustiveness = average(val_Exhaustiveness)
        avg_Efficiency = average(Efficiency)
        avg_val_Efficiency = average(val_Efficiency)

        avg_TP = average(TP)
        avg_val_TP = average(val_TP)
        avg_FP = average(FP)
        avg_val_FP, Elist = average(val_FP, epoch_Mlist=True)


    print("\nLoss:", round(history['loss'][-1], 5), "      Val Loss:", round(history['val_loss'][-1], 5))
    print("Accuracy", round(history[keys[1]][-1], 4), "       Val Accuracy", round(history[keys[5]][-1], 4))
    print("Efficiency", round(avg_Efficiency[-1], 4), "       Val Efficiency", round(avg_val_Efficiency[-1], 4))
    print("Exhaustiveness", round(avg_Exhaustiveness[-1], 4), "       Val Exhaustiveness", round(avg_val_Exhaustiveness[-1], 4))


    print("\nMaking plots to visualize ANN training") 
    plt.plot(np.arange(num_epochs-5), (history)['loss'][5:], 'b', label="loss")
    plt.plot(np.arange(num_epochs-5), (history)['val_loss'][5:], 'r', label="val loss")
    plt.legend()
    #plt.show()
    plt.savefig('Trained_Model/LossPlot.pdf')
    plt.figure()

    plt.plot(np.arange(num_epochs), history[keys[1]], 'b', label="accuracy")
    plt.plot(np.arange(num_epochs), history[keys[5]], 'r', label="val accuracy")
    plt.ylim(0,1.1)
    plt.legend()
    #plt.show()
    plt.savefig('Trained_Model/AccuracyPlot.pdf')
    plt.figure()

    plt.plot(Elist, avg_Efficiency, linestyle=':', color='blue', label="Efficiency")
    plt.plot(Elist, avg_val_Efficiency, linestyle=':', color='red', label="Val Efficiency")
    plt.ylim(-0.001,0.03)
    plt.legend()
    #plt.show()
    plt.savefig('Trained_Model/EfficiencyPlot.pdf')
    plt.figure()

    plt.plot(Elist, avg_Exhaustiveness, linestyle=':', color='blue', label="Exhaustiveness")
    plt.plot(Elist, avg_val_Exhaustiveness, linestyle=':', color='red', label="Val Exhaustiveness")
    plt.ylim(-0.02,1.1)
    plt.legend()
    #plt.show()
    plt.savefig('Trained_Model/ExhaustivenessPlot.pdf')
    plt.figure()

    plt.plot(Elist, avg_TP, linestyle=':', color='blue', label="True Positives")
    plt.plot(Elist, avg_val_TP, linestyle=':', color='red', label="Val True Positives")
    plt.axhline(y = np.sum(y_trn), color = 'blue')
    plt.axhline(y = np.sum(y_val), color = 'red')
    plt.legend()
    #plt.show()
    plt.savefig('Trained_Model/TruePosPlot.pdf')
    plt.figure()

    plt.plot(Elist, avg_FP, linestyle=':', color='blue', label="False Positives")
    plt.plot(Elist, avg_val_FP, linestyle=':', color='red', label="Val False Positives")
    plt.legend()
    #plt.show()
    plt.savefig('Trained_Model/FalsePosPlot.pdf')

    
    return None

def NormalizeInput(X, new_scheme=True, mean=None, std=None):
    if new_scheme:
        mean = np.mean(X, axis = 0)
        std = np.std(X, axis = 0)
        X_norm = np.array([(x - mean)/std for x in X])
        return X_norm, np.array([mean,std])
    else:
        X_norm = np.array([(x - mean)/std for x in X])
        return X_norm



def TrainANN(data_type2, under_sample=None, over_sample=None, load_network=False, train_network=True, save_network=True):

    if load_network:
        model = tf.keras.models.load_model("Trained_Model/Model")
        with open("Trained_Model/Model_History.pkl", "rb") as f:
            history = pickle.load(f)
        X_boosted_trn = np.load("Trained_Model/x_train.npy")
        y_boosted_trn = np.load("Trained_Model/y_train.npy")
        X_val = np.load("Trained_Model/x_val.npy")
        y_val = np.load("Trained_Model/y_val.npy")
        print("Loaded saved ANN model")
        print("\nNetwork architecture")
        print(model.summary())

    else:
        model = ConstructModel()
        print("Network architecture")
        print(model.summary())
    
        data = ReadFiles(data_type=1, data_type2=data_type2, plot_dist=False)
        X = data[:,:10]
        X_norm, norm_var = NormalizeInput(X, new_scheme=True, mean=None, std=None)
        y = data[:,10]

        # Defining training and validation sets
        X_trn, X_val, y_trn, y_val = train_test_split(X_norm,y)

        X_boosted_trn, y_boosted_trn = Boosting(X_trn, y_trn, under_sample=0.01, over_sample=None)

        print("Size of batch is", len(X_boosted_trn))


    if train_network:
        print("Training network...")

        history, model = Train(model, X_boosted_trn, y_boosted_trn, X_val, y_val)
        print("Network trained!")
        if save_network:
            model.save("Trained_Model/Model")
            with open("Trained_Model/Model_History.pkl", "wb") as f:
                pickle.dump(history, f)
            np.save("Trained_Model/x_train.npy", X_boosted_trn)
            np.save("Trained_Model/y_train.npy", y_boosted_trn)
            np.save("Trained_Model/x_val.npy", X_val)
            np.save("Trained_Model/y_val.npy", y_val)
            np.save("Trained_Model/NormVariables.npy", norm_var)

            print("Saved network, its history and training/validation sets in the directory Trained_Network")

        #y = model.predict(X_val)
        #y = [1 if item > 0.5 else 0 for item in y]
        #print("pos points", np.sum(y), "out of", len(y))

    print("The baseline accuracy is", np.sum(y_val)/y_val.shape[0])
    print("Number of points in validation set is", y_val.shape[0])

    print("\nConstructing training plots")
    PlotMetric(history, y_boosted_trn, y_val)
    #if not load_network:
        #PlotData(X_boosted_trn, y_boosted_trn, "Trained_Model/TrnDataPlot", plot_dist=False, read_data=read_data)
        #PlotData(X_val, y_val, "Trained_Model/ValDataPlot", plot_dist=False, read_data=read_data)
    
    return None

def Predict():
    model = tf.keras.models.load_model("Trained_Model/Model")
    norm_var = np.load("Trained_Model/NormVariables.npy")
    mean, std = norm_var[0], norm_var[1]

    X = DataHandling.ReadFreeParams(data_type1=2)
    X_norm = NormalizeInput(X, new_scheme=False, mean=mean, std=std)

    print("\nNetwork is making predictions")
    y = model.predict(X_norm)
    y = [1 if item > 0.5 else 0 for item in y]

    return y


#RunANN()





