from __future__ import print_function
import argparse

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

from comm_fn import sentences2vec, load_glove_model, load_data, plot_curve
from tf_logistic_regression import TFLogisticRegression

# Handles the command-line argument, which specifies the dataset to be used.
parser = argparse.ArgumentParser(description='Choose data to load into the '
                                             'Logistic Regression classifier.')
parser.add_argument('input', help='choose from: "MR", "SO", "CR", "MPQA"',
                    choices=["MR", "SO", "CR", "MPQA"])
args = parser.parse_args()

# Model now contains a dictionary {word:vector}. Where each word is a key
# corresponding to a 300-d row vector, shape (300,).
model = load_glove_model('glove.840B.300d.txt')

# Load data in to lists of positive and negative examples.
pos, neg = load_data(args.input)

# Shape (n_pos_samples, 300)
pos_vectors = np.asarray(sentences2vec(pos, model))
# Shape (n_neg_samples, 300)
neg_vectors = np.asarray(sentences2vec(neg, model))

# Prepare a matrix containing both positive and negative samples
X = np.r_[pos_vectors, neg_vectors]
y = np.zeros(X.shape[0])
y[:pos_vectors.shape[0]] = 1.0

# Randomly split the X and y arrays into 30/70 test/train split.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=0)

# Preprocess data. When is this useful?
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

# Run training and testing with sklearn.
print('Evaluating...using sklearn')
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('Test samples: %d  Misclassified samples: %d'
      %(y_test.shape[0], (y_test != y_pred).sum()))
print('Test accuracy: %.4f'%clf.score(X_test, y_test))

# Reshape to match y_pred (output from tensorflow implementation).
y_test = y_test.reshape((y_test.shape[0], 1))

# Run training and testing with tensorflow.
print('Evaluating...using tensorflow')
tf_clf = TFLogisticRegression()
tf_clf.fit(X_train, y_train)
y_pred = tf_clf.predict(X_test)
# Where is the best place to break the below statement in to a new line -- or
# is this just personal preference?
print('Test samples: %d  Misclassified samples: %d'
      %(y_test.shape[0], (y_test != y_pred).sum()))
print('Test accuracy: %.4f' % tf_clf.score(X_test, y_test))

# Plot learning curves.
for param, color in zip([0.003, 0.01, 0.03, 0.1, 0.3],
                        ['r', 'g', 'b', 'y', 'k']):
    clf = LogisticRegression(C=param)
    plot_curve(clf, X_train, y_train, title="Logistic Regression (Varying C)",
               label='C = ' + str(param), color=color)
plt.show()
