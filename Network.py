# Import other files
from UserInput import *
from DerivedInput import *
import DataHandling as DH 

# Import libraries
import numpy as np
import math
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import pandas as pd
import sys

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Silence tensorflow warnings
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from collections import Counter
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler




def Boosting(data, under_sample, over_sample):
    X, y = data[:,:-1], data[:,-1]
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

    data_boosted = np.c_[X,y]
    return data_boosted


def ConstructModel():
    model = keras.Sequential(
            [
                layers.Dense(60, activation="relu", name="layer1", input_shape=(num_free_param,),  kernel_regularizer=keras.regularizers.L1L2(0.001)),
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
        batch_size=batch_size,
        epochs=network_epochs,
        validation_data = (X_val, y_val),
        class_weight = {0: 1.0, 1: class_weight},
        verbose=network_verbose,
        )
    return history.history, model


def PlotMetric(history,y_trn,y_val):
    
    ANN_path = "SavedANNs/{}-ANN".format(BSM_model)
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


    print("\nNeural network training summary")
    print("Loss:", round(history['loss'][-1], 5), "      Val Loss:", round(history['val_loss'][-1], 5))
    print("Accuracy", round(history[keys[1]][-1], 4), "       Val Accuracy", round(history[keys[5]][-1], 4))
    print("Efficiency", round(avg_Efficiency[-1], 4), "       Val Efficiency", round(avg_val_Efficiency[-1], 4))
    print("Exhaustiveness", round(avg_Exhaustiveness[-1], 4), "       Val Exhaustiveness", round(avg_val_Exhaustiveness[-1], 4))


    print("\nMaking plots to visualize ANN training ...") 
    plt.plot(np.arange(num_epochs-5), (history)['loss'][5:], 'b', label="loss")
    plt.plot(np.arange(num_epochs-5), (history)['val_loss'][5:], 'r', label="val loss")
    plt.legend()
    #plt.show()
    plt.savefig('{}/LossPlot.pdf'.format(ANN_path))
    plt.figure()

    plt.plot(np.arange(num_epochs), history[keys[1]], 'b', label="accuracy")
    plt.plot(np.arange(num_epochs), history[keys[5]], 'r', label="val accuracy")
    plt.ylim(0,1.1)
    plt.legend()
    #plt.show()
    plt.savefig('{}/AccuracyPlot.pdf'.format(ANN_path))
    plt.figure()

    plt.plot(Elist, avg_Efficiency, linestyle=':', color='blue', label="Efficiency")
    plt.plot(Elist, avg_val_Efficiency, linestyle=':', color='red', label="Val Efficiency")
    plt.ylim(-0.001,0.08)
    plt.legend()
    #plt.show()
    plt.savefig('{}/EfficiencyPlot.pdf'.format(ANN_path))
    plt.figure()

    plt.plot(Elist, avg_Exhaustiveness, linestyle=':', color='blue', label="Exhaustiveness")
    plt.plot(Elist, avg_val_Exhaustiveness, linestyle=':', color='red', label="Val Exhaustiveness")
    plt.ylim(-0.02,1.1)
    plt.legend()
    #plt.show()
    plt.savefig('{}/ExhaustivenessPlot.pdf'.format(ANN_path))
    plt.figure()

    plt.plot(Elist, avg_TP, linestyle=':', color='blue', label="True Positives")
    plt.plot(Elist, avg_val_TP, linestyle=':', color='red', label="Val True Positives")
    plt.axhline(y = np.sum(y_trn), color = 'blue')
    plt.axhline(y = np.sum(y_val), color = 'red')
    plt.legend()
    #plt.show()
    plt.savefig('{}/TruePosPlot.pdf'.format(ANN_path))
    plt.figure()

    plt.plot(Elist, avg_FP, linestyle=':', color='blue', label="False Positives")
    plt.plot(Elist, avg_val_FP, linestyle=':', color='red', label="Val False Positives")
    plt.legend()
    #plt.show()
    plt.savefig('{}/FalsePosPlot.pdf'.format(ANN_path))

    print("Done. See {} directory".format(ANN_path))
    
    
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



def TrainANN(data_type2, under_sample, over_sample, load_network, train_network, save_network):
    ANN_path = "SavedANNs/{}-ANN".format(BSM_model)
    if load_network: 
        print("Loading trained neural network")
        model = tf.keras.models.load_model("{}/Model".format(ANN_path))
        with open("{}/Model_History.pkl".format(ANN_path), "rb") as f:
            history = pickle.load(f)
        X_boosted_trn = np.load("{}/x_train.npy".format(ANN_path))
        y_boosted_trn = np.load("{}/y_train.npy".format(ANN_path))
        X_val = np.load("{}/x_val.npy".format(ANN_path))
        y_val = np.load("{}/y_val.npy".format(ANN_path))
        norm_var = np.load("{}/NormVariables.npy".format(ANN_path))
        print(model.summary())

    else:
        # Construct neural network
        model = ConstructModel()
        #print("\nNetwork architecture")
        print(model.summary())
   
        # Load data
        data = DH.ReadFiles(data_type1=1, data_type2=data_type2)
        X, y = data[:,:-1], data[:,-1]
        X_norm, norm_var = NormalizeInput(X)    # Switch orders of boosting and normalizing!

        # Defining training and validation sets
        X_trn, X_val, y_trn, y_val = train_test_split(X_norm,y)
        data_boosted_trn = Boosting(np.c_[X_trn, y_trn], under_sample, over_sample)
        X_boosted_trn, y_boosted_trn = data_boosted_trn[:,:-1], data_boosted_trn[:,-1]

    #print("\nSize of batch is", len(X_boosted_trn))
    if train_network:
        print("Training network...")
        history, model = Train(model, X_boosted_trn, y_boosted_trn, X_val, y_val)
        print("Network trained!")

        if save_network:
            model.save("{}/Model".format(ANN_path))
            with open("{}/Model_History.pkl".format(ANN_path), "wb") as f:
                pickle.dump(history, f)
            np.save("{}/x_train.npy".format(ANN_path), X_boosted_trn)
            np.save("{}/y_train.npy".format(ANN_path), y_boosted_trn)
            np.save("{}/x_val.npy".format(ANN_path), X_val)
            np.save("{}/y_val.npy".format(ANN_path), y_val)
            np.save("{}/NormVariables.npy".format(ANN_path), norm_var)
            print("Saved network, its history and training/validation sets in the directory {}".format(ANN_path))

        #y = model.predict(X_val)
        #y = [1 if item > 0.5 else 0 for item in y]
        #print("pos points", np.sum(y), "out of", len(y))

    print("The baseline accuracy is", np.sum(y_val)/y_val.shape[0])
    #print("Number of points in validation set is", y_val.shape[0])

    PlotMetric(history, y_boosted_trn, y_val)
    #if not load_network:
        #PlotData(X_boosted_trn, y_boosted_trn, "Trained_Model/TrnDataPlot", plot_dist=False, read_data=read_data)
        #PlotData(X_val, y_val, "Trained_Model/ValDataPlot", plot_dist=False, read_data=read_data)
    
    return model, norm_var

def Predict(model, norm_var, X):
    #model = tf.keras.models.load_model("TrainedANN/Model")
    #norm_var = np.load("TrainedANN/NormVariables.npy")
    mean, std = norm_var[0], norm_var[1]

    #X = DH.ReadFreeParams(data_type1=2)
    X_norm = NormalizeInput(X, new_scheme=False, mean=mean, std=std)

    y = model.predict(X_norm.tolist())
    y = [1 if item > 0.5 else 0 for item in y]

    return y







