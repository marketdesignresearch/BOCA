"""
FILE DESCRIPTION:

This file implements the mlca MVNN class that has the following functionalities:
    0.CONSTRUCTOR:  __init__(self, X_train, Y_train, scaler)
        *(X_train,Y_train) is the training set of bundle-value pairs*.
        X_train = The bundles of items
        Y_train = The corresponding values for the bundles from a specific bidder. If alreday scaled to a vertai range than you also have to set the scaler variable in teh folowing.
        scaler =  A scaler instance from sklearn.preprocessing.*, e.g., sklearn.preprocessing.MinMaxScaler(), which was used to scale the Y_train variables before creating a NN instance.
                  This instance ins used in the class NN for rescaling errors to the original scale, i.e., it is used as scaler.inverse_transform().
    1. METHOD: initialize_model(self, model_parameters)
        model_parameters = the parameters specifying the neural network:
        This method initializes the attribute model in the class NN by defining the architecture and the parameters of the neural network.
    2. METHOD: fit(self, epochs, batch_size, X_valid=None, Y_valid=None, sample_weight=None)
        epochs = Number of epochs the neural network is trained
        batch_size = Batch size used in training
        X_valid = Test set of bundles
        Y_valid = Values for X_valid.
        sample_weight = weights vector for datapoints of bundle-value pairs.
        This method fits a neural network to data and returns loss numbers.
    3. METHOD: loss_info(self, batch_size, plot=True, scale=None)
        batch_size = Batch size used in training
        plot = boolean parameter if a plots for the goodness of fit should be executed.
        scale = either None or 'log' defining the scaling of the y-axis for the plots
        This method calculates losses on the training set and the test set (if specified) and plots a goodness of fit plot.
"""

import logging
import random

import numpy as np
import torch
from mvnns.eval_mvnnUB import eval_model


# %% NN Class for each bidder


class MLCA_MVNN:

    def __init__(self,
                 X_train,
                 Y_train,
                 scaler,
                 local_scaling_factor):

        self.M = X_train.shape[1]  # number of items
        self.X_train = X_train  # training set of bundles
        self.Y_train = Y_train  # bidder's values for the bundels in X_train
        self.X_valid = None  # test/validation set of bundles
        self.Y_valid = None  # bidder's values for the bundels in X_valid
        self.model_parameters = None  # neural network parameters
        self.uUB_model = None  # pytorch uUB model
        self.mean_model = None  # pytorch mean model
        self.exp_100_uUB_model = None  # pytorch uUB model
        self.scaler = scaler  # the scaler used for initially scaling the Y_train values
        self.history = None  # return value of the model.fit() method from keras
        self.loss = None  # return value of the model.fit() method from keras
        self.local_scaling_factor = local_scaling_factor
        self.device = torch.device("cpu")

    def initialize_model(self, model_parameters):
        self.model_parameters = model_parameters

    def fit(self,
            epochs,
            batch_size,
            seed,
            bidder_id,
            X_valid=None,
            Y_valid=None):

        # set test set if desired
        self.X_valid = X_valid
        self.Y_valid = Y_valid
        target_max = self.Y_train.reshape(-1, 1).max() / self.local_scaling_factor \
            if self.local_scaling_factor is not None else 1.0

        self.Y_train = self.Y_train / target_max

        # fit model and validate on test set
        train_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(self.X_train.astype(np.float32)),
            torch.from_numpy(self.Y_train.reshape(-1).astype(np.float32)))

        # SEEDING ------------------
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        # ---------------------------
        self.model_parameters['num_train_data'] = len(self.Y_train.reshape(-1))
        self.model_parameters['num_val_data'] = 0
        self.model_parameters['num_test_data'] = 0
        self.model_parameters['data_gen_method'] = None
        uUB_model, logs, mean_model, exp_100_uUB_model = eval_model(
            seed=None, input_dim=self.X_train.shape[1],
            train_dataset=train_dataset, val_dataset=None, test_dataset=None, eval_test=False,
            save_datasets=False, send_to=None, new_test_plot=False, plot_history=False, log_full_train_history=False,
            log_path=None, bidder_id=bidder_id, target_max=target_max, **self.model_parameters)

        self.uUB_model = uUB_model
        self.mean_model = mean_model
        self.exp_100_uUB_model = exp_100_uUB_model

        best_epoch = logs['best_epoch']
        best_attempt = logs['best_attempt']

        # self.history = logs
        logging.debug('loss: {:.7f}, kt: {:.4f}, r2: {:.4f}'.format(
            logs['metrics']['train'][best_attempt][best_epoch]['loss'],
            logs['metrics']['train'][best_attempt][best_epoch]['kendall_tau'],
            logs['metrics']['train'][best_attempt][best_epoch]['r2']))

        # only log train metrics forbest epoch and best_attempt
        logs['train metrics'] = logs['metrics']['train'][best_attempt][best_epoch]
        del logs['metrics']

        return logs

    def evaluate(self, X, y):
        raise NotImplementedError()

    def predict(self, X):
        self.model.eval()
        return self.model(torch.from_numpy(X)) * self.dataset_info['target_max']
