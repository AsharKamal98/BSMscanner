import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd

import sys
import os
import tensorflow as tf
import optuna
import multiprocessing as mp
from tabulate import tabulate
from collections import Counter
from sklearn.model_selection import train_test_split
import imblearn 

from UserInput import *
from DerivedInput import *
import DataHandling as DH 

# Silence tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def Boosting(data, under_sample, over_sample):
    X, y = data[:,:-1], data[:,-1]
    counter = Counter(y)
    print("Classes distribution before boosting", counter)

    if under_sample != None:
        undersample = imblearn.under_sampling.RandomUnderSampler(sampling_strategy=under_sample)
        X, y = undersample.fit_resample(X,y)
        counter = Counter(y)
        print("Classes dstribution after under sampling", counter)

    #oversample = imblearn.over_sampling.SMOTE(sampling_strategy=0.1)
    #X, y = oversample.fit_resample(X,y)
    
    if over_sample != None:
        oversample = imblearn.over_sampling.SMOTE(sampling_strategy=over_sample)
        X, y = oversample.fit_resample(X,y)
        counter = Counter(y)
        print("Class distribution after over sampling", counter)

    data_boosted = np.c_[X,y]
    return data_boosted


def NormalizeInput(X, new_scheme=True, mean=None, std=None):
    if new_scheme:
        mean = np.mean(X, axis = 0)
        std = np.std(X, axis = 0)
        X_norm = np.array([(x - mean)/std for x in X])
        return X_norm, np.array([mean,std])
    else:
        X_norm = np.array([(x - mean)/std for x in X])
        return X_norm


def PrintBaselineMetrics(y_train, y_val):
    """Baseline metrics if random search was performed"""
    precision_train = round(np.sum(y_train)/y_train.shape[0], 10)
    precision_val = round(np.sum(y_val)/y_val.shape[0], 10)

    recall = 1.0
    
    f_beta_score_train = round(calcFBetaScore(precision_train, recall), 10)
    f_beta_score_val = round(calcFBetaScore(precision_val, recall), 10)

    print("\nBaseline metrics using random search:")
    print(tabulate(
        [
            ["Precision",precision_train, precision_val],
            ["Recall",recall, recall],
            ["FBetaScore", f_beta_score_train, f_beta_score_val]
         ],
        headers=["", "Training", "Validation"],
        tablefmt="grid"
    ))

    return 


def PrintANNMetrics(history, y_trn, y_val, plot_metric_history):
    ANN_path = "SavedANNs/{}-ANN".format(BSM_model)

    # training metrics
    precision = history["precision"][-1]
    recall = history["recall"][-1]
    f_beta_score = calcFBetaScore(precision, recall)
    # validation metrics
    val_precision = history["val_precision"][-1]
    val_recall = history["val_recall"][-1]
    val_f_beta_score = calcFBetaScore(val_precision, val_recall)

    print("\nNeural network training summary")
    print(tabulate(
        [
            ["Precision", round(precision, 10), round(val_precision, 10)],
            ["Recall", round(recall, 10), round(val_recall, 10)],
            ["FBetaScore", round(f_beta_score, 10), round(val_f_beta_score, 10)],
            ["Loss", round(history["loss"][-1], 10), round(history["val_loss"][-1], 10)],
        ],
        headers=["", "Training", "Validation"],
        tablefmt="grid"
    ))

    # FIXME: temporary
    plt.rcParams["text.usetex"] = False
    def make_plot(metric):
        num_epochs= len((history)['loss'])
        epoch_array = np.arange(num_epochs-5)
        plt.plot(epoch_array, history[f"{metric}"][5:], 'b', label=f"train {metric}")
        plt.plot(epoch_array, history[f"val_{metric}"][5:], 'r', label=f"val {metric}")
        plt.legend()
        plt.savefig(f"{ANN_path}/{metric}Plot.pdf")
        plt.figure()
    
    if plot_metric_history:
        print("\nPlotting ANN metric history ...")
        make_plot(metric = "precision")
        make_plot(metric = "recall")
        # FIXME: f_beta_score not included in history currently
        # make_plot(metric = "f_beta_score")
        make_plot(metric = "loss")
        print(f"Done. See {ANN_path} directory")
    
    return


def ConstructModel(num_hidden_layers, num_hidden_nodes, regularization_strength):
    model = tf.keras.Sequential(
        [tf.keras.layers.Dense(num_hidden_nodes, activation="relu", name="hidden_layer1", input_shape=(num_free_param,),  kernel_regularizer=tf.keras.regularizers.L1L2(regularization_strength))] +
        [tf.keras.layers.Dense(num_hidden_nodes, activation="relu", name=f"hidden_layer{i+2}", kernel_regularizer=tf.keras.regularizers.L1L2(regularization_strength)) for i in range(num_hidden_layers)] +
        [tf.keras.layers.Dense(1, activation="sigmoid", name="output_layer")]
    )

    return model


def calcFBetaScore(precision, recall):
    if precision == 0 and recall == 0:
        return 0

    return (1+fscore_beta_sq) * (precision*recall)/((fscore_beta_sq*precision)+recall)


def objective(trial, X_boosted_trn, y_boosted_trn, X_val, y_val, shared_namespace, lock):
    
    def hyperparameter_sampling(name, type, hyperparameter):
        if isinstance(hyperparameter, list):
            if type=="float":
                return trial.suggest_float(name, hyperparameter[0], hyperparameter[1])
            else:
                return trial.suggest_int(name, hyperparameter[0], hyperparameter[1])
        else:
            return hyperparameter

    # hyperparameter space sampling
    num_hidden_layers = hyperparameter_sampling("num_hidden_layers", "int", num_hidden_layers_sampling)
    num_hidden_nodes = hyperparameter_sampling("num_hidden_nodes", "int", num_hidden_nodes_sampling)
    regularization_strength = hyperparameter_sampling("regularization_strength", "float", regularization_strength_sampling)
    adam_learning_rate = hyperparameter_sampling("adam_learning_rate", "float", adam_learning_rate_sampling)
    steps_per_epoch = hyperparameter_sampling("steps_per_epoch", "int", steps_per_epoch_sampling)
    class_weight = hyperparameter_sampling("class_weight", "float", class_weight_sampling)

    # Construct neural network
    model = ConstructModel(num_hidden_layers, num_hidden_nodes, regularization_strength) 

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=adam_learning_rate),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[
                        tf.keras.metrics.Precision(name="precision"),
                        tf.keras.metrics.Recall(name="recall"),                        
                    ]
    )

    # Early stopping: stops ANN training if metric of choice stops improving
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=early_stop_min_delta,
        patience=early_stop_patience,         
        restore_best_weights=early_stop_restore_best_weights,
        verbose=True,
    )

    history = model.fit(
        X_boosted_trn,
        y_boosted_trn.reshape(-1, 1),
        #batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        epochs=network_epochs,
        validation_data = (X_val, y_val.reshape(-1, 1)),
        class_weight = {0: 1.0, 1: class_weight},
        callbacks= [early_stopping] if early_stop else None,
        verbose=network_verbose,
        )
        
    # pick metric to maximize
    val_precision = history.history['val_precision'][-1]
    val_recall = history.history['val_recall'][-1]
    val_f_beta_score = calcFBetaScore(val_precision, val_recall)

    # Save the best model if this trial is better
    with lock:
        if val_f_beta_score >= shared_namespace.best_val_f_beta_score:  # Maximize val_f_beta_score
            shared_namespace.best_val_f_beta_score = val_f_beta_score
            shared_namespace.best_model = model
            shared_namespace.best_history = history  

    return val_f_beta_score
    

def run_study(args):
    X_boosted_trn, y_boosted_trn, X_val, y_val, shared_namespace, lock = args

    # Each process loads the same study by name from the shared database
    study = optuna.load_study(
        study_name=optuna_study_name, storage=f"sqlite:///{optuna_study_name}.db"
    )
    study.optimize(lambda trial: objective(trial, X_boosted_trn, y_boosted_trn, X_val, y_val, shared_namespace, lock), n_trials=int(num_ann_models/num_processes)+1)


def TrainANN(data_type2, under_sample, over_sample, save_network):
    # Load data
    data = DH.ReadFiles(data_type1=1, data_type2=data_type2)
    X, y = data[:,:-1], data[:,-1]
    X_norm, norm_var = NormalizeInput(X)    # Switch orders of boosting and normalizing!

    # Defining training and validation sets
    X_trn, X_val, y_trn, y_val = train_test_split(X_norm,y)
    data_boosted_trn = Boosting(np.c_[X_trn, y_trn], under_sample, over_sample)
    X_boosted_trn, y_boosted_trn = data_boosted_trn[:,:-1], data_boosted_trn[:,-1]
    # FIXME: temporary
    # ANN_path = "SavedANNs/{}-ANN".format(BSM_model)
    # X_boosted_trn = np.load("{}/x_train.npy".format(ANN_path))
    # y_boosted_trn = np.load("{}/y_train.npy".format(ANN_path))
    # X_val = np.load("{}/x_val.npy".format(ANN_path))
    # y_val = np.load("{}/y_val.npy".format(ANN_path))
    # norm_var = np.load("{}/NormVariables.npy".format(ANN_path))

    try:
        # Create study with RDBStorage to enable sharing between processes
        storage = optuna.storages.RDBStorage(f"sqlite:///{optuna_study_name}.db")
        study = optuna.create_study(
            study_name=optuna_study_name, storage=storage, direction="maximize", load_if_exists=False
        )
    except optuna.exceptions.DuplicatedStudyError:
        while True:
            user_input = input(
                f"WARNING: an optuna study with the name {optuna_study_name} already exists. "
                "The code does not currently support re-using old studies. "
                "Delete old study and create new one with the same name? (y/n): ").strip().lower()
            if user_input=="y":
                # Remove old study
                os.remove(f"{optuna_study_name}.db")
                # Re-create study with RDBStorage to enable sharing between processes
                storage = optuna.storages.RDBStorage(f"sqlite:///{optuna_study_name}.db")
                study = optuna.create_study(
                    study_name=optuna_study_name, storage=storage, direction="maximize", load_if_exists=False
                )
                break
            elif user_input=="n":
                sys.exit(f"\nEXITING: save the desired model from {optuna_study_name} and re-start")
            else:
                print("\nInvalid input. Please enter 'y' or 'n'.")

    manager = mp.Manager()
    # shared variables to track the best model and its performance
    shared_namespace = manager.Namespace()
    shared_namespace.best_val_f_beta_score = 0
    shared_namespace.best_model = None
    shared_namespace.best_history = None
    # lock for shared variables
    lock = manager.Lock()

    print("Training network...")
    with mp.Pool(num_processes) as pool:
        pool.map(run_study, [(X_boosted_trn, y_boosted_trn, X_val, y_val, shared_namespace, lock)] * num_processes)
    print("Network trained!")

    if save_network:
        ANN_path = "SavedANNs/{}-ANN".format(BSM_model)
        os.makedirs(f"{ANN_path}", exist_ok=True)
        shared_namespace.best_model.save("{}/Model".format(ANN_path))
        with open("{}/Model_History.pkl".format(ANN_path), "wb") as f:
            pickle.dump(shared_namespace.best_history.his, f)
        np.save("{}/x_train.npy".format(ANN_path), X_boosted_trn)
        np.save("{}/y_train.npy".format(ANN_path), y_boosted_trn)
        np.save("{}/x_val.npy".format(ANN_path), X_val)
        np.save("{}/y_val.npy".format(ANN_path), y_val)
        np.save("{}/NormVariables.npy".format(ANN_path), norm_var)
        print("Saved network, its history and training/validation sets in the directory {}".format(ANN_path))

    # ANN metrics
    PrintANNMetrics(shared_namespace.best_history.history, y_boosted_trn, y_val, plot_metric_history=True)
    # baseline metrics
    PrintBaselineMetrics(y_boosted_trn, y_val)
    
    return shared_namespace.best_model, norm_var


def LoadANN():
    ANN_path = "SavedANNs/{}-ANN".format(BSM_model)

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

    # ANN metrics
    PrintANNMetrics(history, y_boosted_trn, y_val, plot_metric_history=False)
    # baseline metrics
    PrintBaselineMetrics(y_boosted_trn, y_val)

    return model, norm_var


def Predict(model, norm_var, X):
    mean, std = norm_var[0], norm_var[1]
    X_norm = NormalizeInput(X, new_scheme=False, mean=mean, std=std)
    y = model.predict(X_norm.tolist())
    y = [1 if item > 0.5 else 0 for item in y]

    return y