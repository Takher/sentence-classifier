from __future__ import print_function
import numpy as np
import operator

from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score
from sklearn.linear_model import SGDClassifier, LogisticRegression

from tf_logistic_regression import TFLogisticRegression
from comm_fn import minibatch


def gen_test_vals(n_values = 10, starting_factor = -5):
    """
    Generate list of values starting from 1e-5 (by default).
    :param n_values: int
        number of values in new list.
    :param starting_factor: int
        The order of the starting value. This is the smallest value in the
        list.
    :return: list
    """
    test_values = []
    for i in range(n_values):
        try_value_1 = (10)**(i+starting_factor)
        try_value_2 = try_value_1*3
        test_values.extend((try_value_1, try_value_2))
    return test_values


def tune_param(X, y, type, max_iter=1000, batch_size=500, learning_rate=0.01,
               seed=0, test_values=None):
    """
    Tunes the regularisation hyperparameter for each classifier using K-fold
    cross-validation.

    :param X: array-like, shape (n_samples, n_features)
        Training samples.
    :param X: array-like, shape (n_samples, n_features)
        Training samples.
    :param type: string
        Type of supervised task to which the hyperparameter is tuned.
    :param max_iter: int
        Maximum number of iterations used in minibatch gradient descent.
    :param batch_size: int
        Size of batch used in minibatch gradient descent.
    :param learning_rate: float
    :param seed: int
    :param test_values: list
        Values to try for the hyperparameter we are tuning. If not supplied,
        test values will be generated using the gen_test_vals() function.
    :return: float
        Optimal value for hyperparameter
    """
    rand = np.random.RandomState(seed)

    if test_values == None:
        test_values = gen_test_vals()

    # Used to store the mean ave precision from each fold
    ave_param_score = {}

    # Used to construct a unique name for each time we use a the tf classifier
    i = 0

    for param in test_values:
        ap_results = []

        # Tuning using KFold Cross validation
        kf = KFold(n_splits=3, shuffle=True, random_state=seed)
        for train_index, test_index in kf.split(X, y):
            i += 1
            X_train = X[train_index]
            y_train = y[train_index]
            X_test = X[test_index]
            y_test = y[test_index]

            if type == 'tf':
                # Give each classifier a name. The name is used to ensure that
                # weights for each tf classifier are not shared.
                name = 'para_' + str(param) + '_fold_' + str(i)

                clf = TFLogisticRegression(max_iter, random_state=seed,
                                           name=name)
                clf.fit(X_train, y_train, batch_size, learning_rate, reg=param)
            elif type == 'lr':
                clf = LogisticRegression(random_state=seed, C=(1.0/param))
                clf.fit(X_train, y_train)
            elif type == 'sgd':
                clf = SGDClassifier(random_state=0, loss='log', penalty='l2',
                                    alpha=(param/float(batch_size)))
                for _ in range(max_iter):
                    # Select random minibatch.
                    X_batch, y_batch = minibatch(rand, X_train, y_train,
                                                 batch_size)
                    clf.partial_fit(X_batch, y_batch,
                                    classes=np.array(([0, 1])))
            ap = average_precision_score(y_test, clf.decision_function(X_test))
            ap_results.append(ap)
        # ave_param_score is a dictionary, {param: ap}
        ave_param_score[str(param)] = np.mean(ap_results)

    # The key that corresponds to the largest Average Precision score
    best_param = max(ave_param_score.iteritems(),
                     key=operator.itemgetter(1))[0]
    return float(best_param)
