from __future__ import print_function
import os.path
import operator
import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split, GridSearchCV

from comm_fn import sentences2vec, load_glove_model, load_data, PC_comparision
from comm_fn import plot_curve, minibatch, save_misclf_data, parse_options, standard
from tf_logistic_regression import TFLogisticRegression
# from dl_model import DAN
from tuning import set_param

# Set constants for classification algorithms
BATCH_SIZE = 500
MAX_ITER = 1000
SEED = 0
LEARNING_RATE = 0.006

# Set optional parameters
args = parse_options()

# Prepare data
X_train, X_test, y_train, y_test, X_ind = standard(args)

reg_tuned_lr, reg_tuned_tf, reg_tuned_sgd = set_param(args, X_train, y_train)

# Run training and testing with LogisticRegression.
print('Evaluating...using LogisticRegression')
clf_lr = LogisticRegression(random_state=SEED, C=(1.0/reg_tuned_lr))
clf_lr.fit(X_train, y_train)
ap_lr = average_precision_score(y_test, clf_lr.decision_function(X_test))
print('Test accuracy: %.4f'%ap_lr)
y_pred = clf_lr.predict(X_test)
print('Test samples: %d  Misclassified samples: %d'
      %(y_test.shape[0],(y_test != y_pred).sum()))

# Run training and testing with SGDClassifier.
print('Evaluating...using SGDClassifier')
rand = np.random.RandomState(SEED)

# alpha is defined as lambda/n_samples
clf_sgd = SGDClassifier(random_state=0, loss='log', penalty='l2',
                        alpha=(reg_tuned_sgd/float(BATCH_SIZE)))
for _ in range(MAX_ITER):
    # Select random minibatch.
    X_batch, y_batch = minibatch(rand, X_train, y_train, BATCH_SIZE)
    clf_sgd.partial_fit(X_batch, y_batch, classes=np.array(([0,1])))
ap_sgd = average_precision_score(y_test, clf_sgd.decision_function(X_test))
print('Test accuracy: %.4f'%ap_sgd)
y_pred = clf_sgd.predict(X_test)
print('Test samples: %d  Misclassified samples: %d'
      %(y_test.shape[0], (y_test != y_pred).sum()))

# Reshape to match y_pred (output from tensorflow implementation).
y_test = y_test.reshape((y_test.shape[0], 1))

# Run training and testing with tensorflow.
print('Evaluating...using tensorflow')
tf_clf = TFLogisticRegression(max_iter=MAX_ITER, random_state=SEED)
tf_clf.fit(X_train, y_train, BATCH_SIZE, LEARNING_RATE, reg=reg_tuned_tf)
y_pred = tf_clf.predict(X_test)
print('Test samples: %d  Misclassified samples: %d'
      %(y_test.shape[0], (y_test != y_pred).sum()))
ap_tf = average_precision_score(y_test, tf_clf.decision_function(X_test))
print('Test accuracy: %.4f'%ap_tf)
print('Regular test accuracy for tf classifier: %.4f' % tf_clf.score(X_test, y_test))

##################### Evaluate optimal number of PCs ##########################
# # Example of evaluatinf PCs with SGD
# X_train_PCs, X_test_PCs = PC_comparision(X_train, X_test)
# print('Evaluating...PCs for SGD')
# pc_results = {}
# type = 'sgd'
#
# for key in X_train_PCs:
#     if type == 'lr':
#         clf = LogisticRegression(random_state=SEED, C=(1.0/reg_tuned_lr))
#         clf.fit(X_train_PCs[key], y_train)
#     elif type == 'tf':
#         clf = TFLogisticRegression(max_iter=MAX_ITER, random_state=SEED,
#                                    name=str(key))
#         clf.fit(X_train_PCs[key], y_train, BATCH_SIZE, LEARNING_RATE,
#                                    reg=reg_tuned_tf)
#     elif type == 'sgd':
#         rand = np.random.RandomState(SEED)
#         clf = SGDClassifier(random_state=0, loss='log', penalty='l2',
#                             alpha=(reg_tuned_sgd/float(BATCH_SIZE)))
#         for _ in range(MAX_ITER):
#             X_batch, y_batch = minibatch(rand, X_train_PCs[key], y_train,
#                                           BATCH_SIZE)
#             clf.partial_fit(X_batch, y_batch, classes=np.array(([0,1])))
#     ap = average_precision_score(y_test,
#                                  clf.decision_function(X_test_PCs[key]))
#     print('Test accuracy for PC ', key, ' is %.4f'%ap)
#     pc_results[key] = ap
# best_pc = max(pc_results.iteritems(), key=operator.itemgetter(1))[0]
# print('highest precision is for::', best_pc)
# print()
# print('%.4f'%pc_results[2])
# print('%.4f'%pc_results[5])
# print('%.4f'%pc_results[10])
# print('%.4f'%pc_results[20])
# print('%.4f'%pc_results[50])
# print('%.4f'%pc_results[100])
# print('%.4f'%pc_results[150])
# print('%.4f'%pc_results[200])
# print('%.4f'%pc_results[250])
# print('%.4f'%pc_results[300])
# print()
##################### Evaluate optimal number of PCs ##########################

############# Plot learning curves for particular hyperparameter ##############
# for param, color in zip([0.003, 0.01, 0.03, 0.1, 0.3],
#                         ['r', 'g', 'b', 'y', 'k']):
#     clf = LogisticRegression(C=param)
#     plot_curve(clf, X_train, y_train, title="Logistic Regression (Varying C)",
#                train_sizes=args.data, label='C = ' + str(param), color=color)
# plt.show()
############# Plot learning curves for particular hyperparameter ##############

################## ANALYSE MISCLASSIFIED DATA POINTS ##########################
# # misclf_ind contains a list of indices (corresponding to the original
# feature vector, X) of our missclassified samples
# a = np.where(y_test != y_pred)[0]
# misclf_ind = X_ind[a]
#
# # Split back into pos & neg, since sentences are stored in pos & neg
# pos_ind = [int(i) for i in misclf_ind if i < pos_vectors.shape[0]]
# neg_ind = [int(i) for i in misclf_ind if i >= pos_vectors.shape[0]]
#
# # Saves misclassified results to the results folder with the true value (pos
# # or neg) in the filename
# save_misclf_data(pos_ind, neg_ind, pos, neg, args.input)
################## ANALYSE MISCLASSIFIED DATA POINTS ##########################
