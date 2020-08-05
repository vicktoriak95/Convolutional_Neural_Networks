import torch
from torch import Tensor
from torch.utils.data import DataLoader
from collections import namedtuple
import helpers.dataloader_utils as dl_utils

from .losses import ClassifierLoss


class LinearClassifier(object):

    def __init__(self, n_features, n_classes, weight_std=0.001):
        """
        Initializes the linear classifier.
        :param n_features: Number or features in each sample.
        :param n_classes: Number of classes samples can belong to.
        :param weight_std: Standard deviation of initial weights.
        """
        self.n_features = n_features
        self.n_classes = n_classes

        # TODO: Create weights tensor of appropriate dimensions
        # Initialize it from a normal dist with zero mean and the given std.

        self.weights = None
        # ====== YOUR CODE: ======
        self.weights = torch.empty(n_features, n_classes).normal_(mean=0, std=weight_std)

        # ========================

    def predict(self, x: Tensor):
        """
        Predict the class of a batch of samples based on the current weights.
        :param x: A tensor of shape (N,n_features) where N is the batch size.
        :return:
            y_pred: Tensor of shape (N,) where each entry is the predicted
                class of the corresponding sample. Predictions are integers in
                range [0, n_classes-1].
            class_scores: Tensor of shape (N,n_classes) with the class score
                per sample.
        """

        # TODO: Implement linear prediction.
        # Calculate the score for each class using the weights and
        # return the class y_pred with the highest score.

        y_pred, class_scores = None, None

        # ====== YOUR CODE: ======
        class_scores = torch.mm(x, self.weights)
        y_pred = torch.zeros(x.size()[0], dtype=torch.int64)

        _, y_pred = torch.max(class_scores, 1)
        # ========================

        return y_pred, class_scores

    @staticmethod
    def evaluate_accuracy(y: Tensor, y_pred: Tensor):
        """
        Calculates the prediction accuracy based on predicted and ground-truth
        labels.
        :param y: A tensor of shape (N,) containing ground truth class labels.
        :param y_pred: A tensor of shape (N,) containing predicted labels.
        :return: The accuracy in percent.
        """

        # TODO: calculate accuracy of prediction.
        # Use the predict function above and compare the predicted class
        # labels to the ground truth labels to obtain the accuracy (in %).
        # Do not use an explicit loop.

        acc = None
        # ====== YOUR CODE: ======
        acc = torch.sum(torch.eq(y, y_pred)).item() / y.size()[0]  # not sure if and how works
        # ========================

        return acc * 100

    def train(self,
              dl_train: DataLoader,
              dl_valid: DataLoader,
              loss_fn: ClassifierLoss,
              learn_rate=0.1, weight_decay=0.001, max_epochs=100):

        Result = namedtuple('Result', 'accuracy loss')
        train_res = Result(accuracy=[], loss=[])
        valid_res = Result(accuracy=[], loss=[])

        print('Training', end='')
        for epoch_idx in range(max_epochs):
            # TODO: Implement model training loop.
            # At each epoch, evaluate the model on the entire training set
            # (batch by batch) and update the weights.
            # Each epoch, also evaluate on the validation set.
            # Accumulate average loss and total accuracy for both sets.
            # The train/valid_res variables should hold the average loss and
            # accuracy per epoch.
            #
            # Don't forget to add a regularization term to the loss, using the
            # weight_decay parameter.

            total_correct = 0
            average_loss = 0

            # ====== YOUR CODE: ======
            #train
            batch_loss = 0
            batch_acc = 0
            batch_size = 0
            norm_weights = torch.norm(self.weights)
            batch_grad = torch.zeros(self.weights.size())
            for x_train, y_train in dl_train:
                y_predict, x_scores = self.predict(x_train)
                batch_loss += loss_fn.loss(x_train, y_train, x_scores, y_predict)+0.5*weight_decay*norm_weights
                batch_grad += loss_fn.grad()
                batch_acc += self.evaluate_accuracy(y_train, y_predict)
                batch_size += 1

            grad = batch_grad / batch_size
            self.weights = self.weights - learn_rate * (grad + weight_decay * self.weights)

            train_res.accuracy.append(batch_acc / batch_size)
            train_res.loss.append(batch_loss.item() / batch_size)

            #valid:
            batch_loss = 0
            batch_acc = 0
            batch_size = 0
            batch_grad = torch.zeros(self.weights.size())
            norm_weights = torch.norm(self.weights)
            for x_valid, y_valid in dl_train:
                yv_predict, xv_scores = self.predict(x_valid)
                batch_loss += loss_fn.loss(x_valid, y_valid, xv_scores, yv_predict) + 0.5 * weight_decay * norm_weights
                batch_grad += loss_fn.grad()
                batch_acc += self.evaluate_accuracy(y_valid, yv_predict)
                batch_size += 1

            valid_res.accuracy.append(batch_acc / batch_size)
            valid_res.loss.append(batch_loss.item() / batch_size)

            # ========================
            print('.', end='')

        print('')
        return train_res, valid_res

    def weights_as_images(self, img_shape, has_bias=True):
        """
        Create tensor images from the weights, for visualization.
        :param img_shape: Shape of each tensor image to create, i.e. (C,H,W).
        :param has_bias: Whether the weights include a bias component
            (assumed to be at the end).
        :return: Tensor of shape (n_classes, C, H, W).
        """

        # TODO: Convert the weights matrix into a tensor of images.
        # The output shape should be (n_classes, C, H, W).

        # ====== YOUR CODE: ======
        w_images = self.weights
        features, classes = w_images.size()
        if has_bias:
            w_images = w_images[:features-1]    #slicing
        w_images = torch.reshape(w_images, (classes, *img_shape))
        # ========================

        return w_images

