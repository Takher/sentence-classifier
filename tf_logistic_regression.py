from __future__ import division

import tensorflow as tf
import numpy as np
from sklearn.base import BaseEstimator

from comm_fn import minibatch


class TFLogisticRegression(BaseEstimator):
    """ Logistic Regression classifier.
    Parameters
    ----------
    C : float, default: 100.0
        Inverse of regularization strength; must be a positive float.
        Like in support vector machines, smaller values specify stronger
        regularization.
    max_iter : int, default: 1000
        Maximum number of iterations taken for the solvers to converge.
    random_state : int seed, RandomState instance, default: 0
        The seed of the pseudo random number generator to use when
        shuffling the data
    tol : float, default: 1e-8
        Tolerance for stopping criteria.
        # Will use this once the issue with the cost function is solved
    Attributes
    ----------
    weight_ : array, shape (n_features + 1,)
        Coefficient of features, where n_features is the number of features.
        The intercept (a.ka. bias) term as the first element.
    """
    def __init__(self, max_iter=1000, random_state=0, tol=1e-8, name='ttt'):
        """Initialize model attributes."""
        self.max_iter = max_iter
        self.random_state = random_state
        self.tol = tol
        self.name = name

    def decision_function(self, X_data):
        """Returns the probability that the samples are from the positive class
        Parameters
        ----------
        X_data : array-like, shape (n_samples, n_features)
            Test samples.
        Returns
        -------
        y_pred : array-like, shape (n_samples, 1)
            Predicted labels for X.
        """
        n_samples, n_features = X_data.shape

        # By definition we have X[:, 0] set to one.
        X_data = np.c_[np.ones([n_samples, 1]), X_data]

        X = tf.placeholder(tf.float32, [n_samples, n_features+1])
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # Activation function, shape (n_samples,1)
            a = tf.sigmoid(tf.matmul(X, self.weight_))
            y_pred = sess.run(a, {X: X_data})
        return y_pred

    def predict(self, X_data):
        """Uses the values from predict_proba to estimates the class that a
        sample belongs to

        Parameters
        ----------
        X_data : array-like, shape (n_samples, n_features)
            Test samples.
        Returns
        -------
        y_pred : array-like, shape (n_samples, 1)
            Predicted labels for X.
        """
        return np.round(self.decision_function(X_data))

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Test samples.
        y : array-like, shape (n_samples,1)
            True labels for X.
        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        correct_pred = (y == self.predict(X)).sum()
        score = correct_pred/y.shape[0]
        return score

    def fit(self, X_data, y_data, batch_size=500, learning_rate=0.01, reg=10):
        """Fit the model according to the given training data using
        mini-batch Gradient Descent.
        Parameters
        ----------
        X_data : array-like, shape (n_samples, n_features)
            Feature matrix, where n_samples is the number of samples and
            n_features is the number of features.
        y_data : array-like, shape (n_samples,)
            Target vector relative to X.
        batch_size : int, default: 500
            Size of batch used for computing mini-batch gradient descent
        learning_rate: float, default: 0.01
            Used to control rate of convergence for gradient descent.
        Returns
        -------
        self : object
            Returns self.
        """
        # Ensure results are reproducible.
        tf.set_random_seed(self.random_state)
        rand = np.random.RandomState(self.random_state)  # Is it called seed?

        n_samples, n_features = X_data.shape

        # y_data should match y's dimensions.
        y_data = np.reshape(y_data, (y_data.shape[0], 1))

        # Define placeholders for input. Add column to X for bias term.
        X = tf.placeholder(tf.float32, shape=(batch_size, n_features + 1))
        y = tf.placeholder(tf.float32, shape=(batch_size, 1))

        # To ensure multiple objects from this class have different weights
        # (what is the standard way of dealing with this?)
        with tf.variable_scope(self.name):
            W = tf.get_variable("weight", (n_features + 1, 1),
                                initializer=tf.random_normal_initializer())


        # Activation function shape (n_samples, 1)
        a = tf.sigmoid(tf.matmul(X, W))

        # Used so we can ignore the bias term in regularisation calculations.
        # Is there a better way to do this?
        selecter = tf.concat([tf.zeros([1, 1]), tf.ones([n_features, 1])], 0)

        # Objective function.
        cost_reg_term = (reg/2)*(tf.matmul(tf.transpose(W*selecter),
                                                    W*selecter))/batch_size
        # J_vec contains the raw cost for each sample, shape (batch_size, 1)
        J_vec = (-y*tf.log(a+1e-7)-(1-y)*tf.log(1-a+1e-7))/batch_size
        J = tf.reduce_sum(J_vec) + cost_reg_term

        # AdamOptimizer uses a momentum approach to minimise cost.
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(J)

        with tf.Session() as sess:
            # Initialize Variables in graph.
            sess.run(tf.global_variables_initializer())

            for _ in range(self.max_iter):
                # Select random minibatch.
                X_batch, y_batch = minibatch(rand, X_data, y_data, batch_size)
                # By definition we have X[:, 0] set to one.
                X_batch = np.c_[np.ones(([batch_size, 1])), X_batch]

                # Update weights directly using optimizer
                sess.run(optimizer, {X: X_batch, y: y_batch})

            self.weight_ = sess.run(W, {X: X_batch, y: y_batch})
        return self
