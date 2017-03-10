from __future__ import division

import tensorflow as tf
import numpy as np

from comm_fn import minibatch


class TFLogisticRegression:
    """ Logistic Regression classifier.

    Parameters
    ----------
    C : float, default: 5.0
        Inverse of regularization strength; must be a positive float.
        Like in support vector machines, smaller values specify stronger
        regularization.

    max_iter : int, default: 100
        Maximum number of iterations taken for the solvers to converge.

    random_state : int seed, RandomState instance, default: 0
        The seed of the pseudo random number generator to use when
        shuffling the data

    tol : float, default: 1e-4
        Tolerance for stopping criteria.
        # Will use this once the issue with the cost function is solved

    Attributes
    ----------
    weight_ : array, shape (n_features + 1,)
        Coefficient of features, where n_features is the number of features.
        The intercept (a.ka. bias) term as the first element.
    """
    def __init__(self, C=5.0, max_iter=100, random_state=0, tol=1e-4):
        """Initialize model attributes."""
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.tol = tol

    def predict(self, X_data):
        """Returns the mean accuracy on the given test data and labels.

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

            y_pred = sess.run(tf.round(a), {X: X_data})
        return y_pred

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

    def fit(self, X_data, y_data, batch_size=100, learning_rate=0.1, seed=0):
        """Fit the model according to the given training data using
        mini-batch Gradient Descent.

        Parameters
        ----------
        X_data : array-like, shape (n_samples, n_features)
            Feature matrix, where n_samples is the number of samples and
            n_features is the number of features.

        y_data : array-like, shape (n_samples,)
            Target vector relative to X.

        batch_size : int, default: 100
            Size of batch used for computing mini-batch gradient descent

        learning_rate: float, default: 0.01
            Used to control rate of convergence for gradient descent.
        Returns
        -------
        self : object
            Returns self.
        """
        # Ensure results are reproducible.
        tf.set_random_seed(seed)
        rand = np.random.RandomState(seed)  # Is it called seed?

        n_samples, n_features = X_data.shape

        # y_data should match y's dimensions.
        y_data = np.reshape(y_data, (y_data.shape[0], 1))

        # Learning_rate must be a tensor for use in gradient descent.
        alpha = tf.constant([learning_rate])  # Is this ok?

        # Define placeholders for input. Add column to X for bias term.
        X = tf.placeholder(tf.float32, shape=(batch_size, n_features + 1))
        y = tf.placeholder(tf.float32, shape=(batch_size, 1))

        W = tf.get_variable("weight", (n_features + 1, 1),
                            initializer=tf.random_normal_initializer())

        # Activation function shape (n_samples, 1)
        a = tf.sigmoid(tf.matmul(X, W))

        # Used so we can ignore the bias term in regularisation calculations.
        # Is there a better way to do this?
        selecter = tf.concat([tf.zeros([1, 1]), tf.ones([n_features, 1])], 0)

        reg_update = (W / (self.C * batch_size)) * selecter
        gradient = (tf.reshape(tf.reduce_mean(X * (a - y), 0),
                               [n_features+1, 1]) + reg_update)

        # Objective function.
        cost_reg_term = (1.0/(self.C*2))*(tf.reduce_sum(tf.square(W*selecter)))
        J = tf.reduce_mean((-y*tf.log(a+1e-7)+(y-1)*tf.log(1 - a + 1e-7)) +
                           cost_reg_term)

        with tf.Session() as sess:
            # Initialize Variables in graph.
            sess.run(tf.global_variables_initializer())

            for _ in range(self.max_iter):
                # Select random minibatch.
                X_batch, y_batch = minibatch(rand, X_data, y_data,
                                             n_samples, batch_size)
                # By definition we have X[:, 0] set to one.
                X_batch = np.c_[np.ones(([batch_size, 1])), X_batch]

                # Perform update.
                W -= alpha*gradient
            self.weight_ = sess.run(W, {X: X_batch, y: y_batch})
        return self
