import numpy as np
import sklearn
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
from pandas import DataFrame
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, check_X_y


class LinearRegressor(BaseEstimator, RegressorMixin):
    """
    Implements Linear Regression prediction and closed-form parameter fitting.
    """

    def __init__(self, reg_lambda=0.1):
        self.reg_lambda = reg_lambda

    def predict(self, X):
        """
        Predict the class of a batch of samples based on the current weights.
        :param X: A tensor of shape (N,n_features_) where N is the batch size.
        :return:
            y_pred: np.ndarray of shape (N,) where each entry is the predicted
                value of the corresponding sample.
        """
        X = check_array(X)
        check_is_fitted(self, 'weights_')

        # DONE: Calculate the model prediction, y_pred

        y_pred = None
        # ====== YOUR CODE: ======
        y_pred = np.matmul(X, self.weights_)
        # ========================
        return y_pred

    def fit(self, X, y):
        """
        Fit optimal weights to data using closed form solution.
        :param X: A tensor of shape (N,n_features_) where N is the batch size.
        :param y: A tensor of shape (N,) where N is the batch size.
        """
        X, y = check_X_y(X, y)

        # DONE: Calculate the optimal weights using the closed-form solution
        # Use only numpy functions.

        w_opt = None
        # ====== YOUR CODE: ====== #

        #N = X.shape[0]
        # lambdaI
        lambda_eye = self.reg_lambda * np.eye(X.shape[1])
        #lambda_eye[0, 0] = 0
        # (XT*X + lambdaI)^-1
        inv_mat = np.linalg.inv(X.transpose() @ X + lambda_eye)
        # w = (XT*X + lambdaI)^-1 * X^T*y
        w_opt = inv_mat @ X.transpose() @ y
        # ========================

        self.weights_ = w_opt
        return self

    def fit_predict(self, X, y):
        return self.fit(X, y).predict(X)


class BiasTrickTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        :param X: A tensor of shape (N,D) where N is the batch size or of shape
            (D,) (which assumes N=1).
        :returns: A tensor xb of shape (N,D+1) where xb[:, 0] == 1
        """
        X = check_array(X)
        # Add bias term to X as the first feature.

        # ====== YOUR CODE: ======
        ones = np.ones((np.shape(X)[0], 1))
        xb = np.concatenate((ones, X), axis=1)
        # ========================

        return xb


class BostonFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Generates custom features for the Boston dataset.
    """
    def __init__(self, degree=2):
        self.degree = degree

        # DONE: Your custom initialization, if needed
        # Add any hyperparameters you need and save them as above
        # ====== YOUR CODE: ======
        # ========================

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Transform features to new features matrix.
        :param X: Matrix of shape (n_samples, n_features_).
        :returns: Matrix of shape (n_samples, n_output_features_).
        """
        X = check_array(X)
        # check_is_fitted(self, ['n_features_', 'n_output_features_'])

        # DONE: Transform the features of X into new features in X_transformed
        # Note: You can count on the order of features in the Boston dataset
        # (this class is "Boston-specific"). For example X[:,1] is the second
        # feature ('ZN').

        X_transformed = None
        # ====== YOUR CODE: ======
        x_poly = PolynomialFeatures(degree=self.degree)
        X_transformed = x_poly.fit_transform(X)
        # ========================

        return X_transformed


def top_correlated_features(df: DataFrame, target_feature, n=5):
    """
    Returns the names of features most strongly correlated (correlation is
    close to 1 or -1) with a target feature. Correlation is Pearson's-r sense.

    :param df: A pandas dataframe.
    :param target_feature: The name of the target feature.
    :param n: Number of top features to return.
    :return: A tuple of
        - top_n_features: Sequence of the top feature names
        - top_n_corr: Sequence of correlation coefficients of above features
        Both the returned sequences should be sorted so that the best (most
        correlated) feature is first.
    """

    # Calculates correlations with target and sort features by it

    # ====== YOUR CODE: ======

    # Initiating cor_array
    cor_array = []
    # Getting the target vector
    target_feature_vector = df.iloc[:][target_feature]

    # Looping over df and calculating corr
    for column_name in df.columns:
        if column_name != target_feature:
            feature_vector = df[:][column_name]
            cor = np.corrcoef(target_feature_vector, feature_vector)[0, 1]
            cor_array.append((column_name, cor))

    # Sorting the features by corr
    cor_array = sorted(cor_array, key=lambda x: abs(x[1]), reverse=True)
    result_array = list(zip(*cor_array[:n]))
    top_n_features = [i for i in result_array[0]]
    top_n_corr = [i for i in result_array[1]]

    # ========================

    return top_n_features, top_n_corr


def cv_best_hyperparams(model: BaseEstimator, X, y, k_folds,
                        degree_range, lambda_range):
    """
    Cross-validate to find best hyperparameters with k-fold CV.
    :param X: Training data.
    :param y: Training targets.
    :param model: sklearn model.
    :param lambda_range: Range of values for the regularization hyperparam.
    :param degree_range: Range of values for the degree hyperparam.
    :param k_folds: Number of folds for splitting the training data into.
    :return: A dict containing the best model parameters,
        with some of the keys as returned by model.get_params()
    """

    # TODO: Do K-fold cross validation to find the best hyperparameters
    #
    # Notes:
    # - You can implement it yourself or use the built in sklearn utilities
    #   (recommended). See the docs for the sklearn.model_selection package
    #   http://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
    # - If your model has more hyperparameters (not just lambda and degree)
    #   you should add them to the search.
    # - Use get_params() on your model to see what hyperparameters is has
    #   and their names. The parameters dict you return should use the same
    #   names as keys.
    # - You can use MSE or R^2 as a score.
    best_params = None

    # ====== YOUR CODE: ======
    best_accr = np.inf

    # Splitting the data k-fold
    k_folder = sklearn.model_selection.KFold(k_folds)
    # Iterating over all parameters
    for degree_param in degree_range:
        for lambda_param in lambda_range:

            # Defying current params and setting the model
            params = {'bostonfeaturestransformer__degree': degree_param,
                     'linearregressor__reg_lambda': lambda_param}
            model.set_params(**params)

            avg_accur = 0

            # Checking params on all k folds
            for train_indices, val_indices in k_folder.split(X):
                train_X, train_y = X[train_indices], y[train_indices]
                val_X, val_y = X[val_indices], y[val_indices]

                # Training model on training set
                model.fit(train_X, train_y)

                # Evaluate accuracy on validation set
                y_pred = model.predict(val_X)
                mse = np.mean((val_y - y_pred) ** 2)
                avg_accur += mse

            # Calculating avg of all k_folds
            avg_accur = avg_accur / k_folds

            # Updating Best params
            if avg_accur < best_accr:
                best_accr = avg_accur
                best_params = params


    # ========================
    return best_params
