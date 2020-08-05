import heapq

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import hw1.datasets as hw1datasets
import torch.utils.data.sampler as sampler

#import sys

from helpers import dataloader_utils

#sys.path.append('/Users/YuvalM/Documents/Convolutional/HW2/assignment/helpers/dataloader_utils.py')

import helpers.dataloader_utils as dataloader_utils
from . import dataloaders


class KNNClassifier(object):
    def __init__(self, k):
        self.k = k
        self.x_train = None
        self.y_train = None
        self.n_classes = None

    def train(self, dl_train: DataLoader):
        """
        Trains the KNN model. KNN training is memorizing the training data.
        Or, equivalently, the model parameters are the training data itself.
        :param dl_train: A DataLoader with labeled training sample (should
            return tuples).
        :return: self
        """

        x_train, y_train = dataloader_utils.flatten(dl_train)
        self.x_train = x_train
        self.y_train = y_train
        self.n_classes = len(set(y_train.numpy()))
        return self

    def predict(self, x_test: Tensor):
        """
        Predict the most likely class for each sample in a given tensor.
        :param x_test: Tensor of shape (N,D) where N is the number of samples.
        :return: A tensor of shape (N,) containing the predicted classes.
        """

        # Calculate distances between training and test samples
        dist_matrix = self.calc_distances(x_test)

        # TODO: Implement k-NN class prediction based on distance matrix.
        # For each training sample we'll look for it's k-nearest neighbors.
        # Then we'll predict the label of that sample to be the majority
        # label of it's nearest neighbors.

        n_test = x_test.shape[0]
        y_pred = torch.zeros(n_test, dtype=torch.int64)

        n_train = self.x_train.shape[0]
        y_list = []

        for i in range(n_test):
            # TODO:
            # - Find indices of k-nearest neighbors of test sample i
            # - Set y_pred[i] to the most common class among them

            # ====== YOUR CODE: ======
            curr_dist = dist_matrix[:, i]
            _, k_neighbors = torch.topk(curr_dist, self.k, largest=False)
            curr_y = self.y_train[k_neighbors]
            y_pred[i] = torch.bincount(curr_y).argmax()

        return y_pred

    def calc_distances(self, x_test: Tensor):
        """
        Calculates the L2 distance between each point in the given test
        samples to each point in the training samples.
        :param x_test: Test samples. Should be a tensor of shape (Ntest,D).
        :return: A distance matrix of shape (Ntrain,Ntest) where Ntrain is the
            number of training samples. The entry i, j represents the distance
            between training sample i and test sample j.
        """

        # TODO: Implement L2-distance calculation as efficiently as possible.
        # Notes:
        # - Use only basic pytorch tensor operations, no external code.
        # - No credit will be given for an implementation with two explicit
        #   loops.
        # - Partial credit will be given for an implementation with only one
        #   explicit loop.
        # - Full credit will be given for a fully vectorized implementation
        #   (zero explicit loops). Hint: Open the expression (a-b)^2.

        dists = torch.tensor([])
        # ====== YOUR CODE: ======
        # every row of x_test should be a flattened test vector
        # every row of x_train should be a flattened train vector
        # (a-b)^2 = a^2 + b^2 -2ab
        # the 2st and 2nd exp are l2 norm of the rows
        # the 3rd is a dot product of a and b^T
        dot_test_train = - 2 * torch.mm(self.x_train, x_test.transpose(0, 1))
        sum_train = torch.sum(self.x_train ** 2, axis=1)
        sum_test = torch.sum((x_test ** 2), axis=1)
        dists = dot_test_train + sum_test + sum_train.view(len(sum_train), 1).expand_as(dot_test_train)
        # ========================
        return dists


def accuracy(y: Tensor, y_pred: Tensor):
    """
    Calculate prediction accuracy: the fraction of predictions in that are
    equal to the ground truth.
    :param y: Ground truth tensor of shape (N,)
    :param y_pred: Predictions vector of shape (N,)
    :return: The prediction accuracy as a fraction.
    """
    assert y.shape == y_pred.shape
    assert y.dim() == 1

    # TODO: Calculate prediction accuracy. Don't use an explicit loop.

    wrong_preds = torch.nonzero(y-y_pred).shape[0]  #count how many non-zero vals
    accuracy = 1 - wrong_preds/y.shape[0]
    return accuracy


def find_best_k(ds_train: Dataset, k_choices, num_folds):
    """
    Use cross validation to find the best K for the kNN model.

    :param ds_train: Training dataset.
    :param k_choices: A sequence of possible value of k for the kNN model.
    :param num_folds: Number of folds for cross-validation.
    :return: tuple (best_k, accuracies) where:
        best_k: the value of k with the highest mean accuracy across folds
        accuracies: The accuracies per fold for each k (list of lists).
    """
    print("best k")

    accuracies = []
    fold_size = int(np.floor(len(ds_train) / num_folds))
    indices = list(range(len(ds_train)))

    for i, k in enumerate(k_choices):
        model = KNNClassifier(k=k)

        # TODO: Train model num_folds times with different train/val data.
        # Don't use any third-party libraries.
        # You can use your train/validation splitter from part 1 (even if
        # that means that it's not really k-fold CV since it will be a
        # different split each iteration), or implement something else.

        cur_accuracies = []

        for j_fold in range(num_folds):
            # set the j-th part as the validation set
            # set the remaining parts as the training set
            split1 = fold_size * j_fold
            split2 = fold_size * (j_fold + 1)

            train, validation = indices[:split1] + indices[split2:], indices[split1:split2]

            train_smp = sampler.SubsetRandomSampler(train)
            validation_smp = sampler.SubsetRandomSampler(validation)

            dl_train = torch.utils.data.DataLoader(ds_train, shuffle=False, sampler=train_smp)
            dl_validation = torch.utils.data.DataLoader(ds_train, shuffle=False, sampler=validation_smp)

            x_validation, y_validation = dataloader_utils.flatten(dl_validation)

            # train on the current training set
            model.train(dl_train)
            # evaluate current accuracy
            y_pred = model.predict(x_validation)
            cur_accuracies.append(accuracy(y_validation, y_pred))

        accuracies.append(cur_accuracies)

    best_k_idx = np.argmax([np.mean(acc) for acc in accuracies])
    best_k = k_choices[best_k_idx]

    return best_k, accuracies
