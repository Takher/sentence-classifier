from __future__ import print_function
import os.path

import numpy as np
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, train_test_split


def load_glove_model(gloveFile, word_count=100000):
    """
    Loads a selection of the most common glove vectors into a dictionary.

    :param gloveFile: textfile
        Contains glove vectors. Each line contains a word string followed by a
        'n_features'-dimensional vector to describe the word. Where n_features
        is the number of features.

    :param word_count: int, default: 100000
        Number of words to load from the gloveFile

    :return: dictionary
        {'word': vector}
        word = string of the word we wish to load
        vector = 'n_features'-d vectors to describe the word
    """
    print("Loading Glove Vectors")
    path = './data/gloveFile_done_%d.npy' % (word_count)

    # Saves time by loading existing file, if available.
    if os.path.exists(path):
        glove = np.load(path).item()
        print(len(glove), " words loaded.")
    else:
        f = open(gloveFile,'r')
        glove = {}
        count = 0
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = [float(val) for val in split_line[1:]]
            embedding = np.asanyarray(embedding)
            glove[word] = embedding
            count += 1
            if count >= word_count:
                break
        # Saves the vectors, so we can load it faster next time.
        np.save(path, glove)
        print("Done.",len(glove)," words loaded!")
    return glove


def sentences2vec(sentences, glove):
    """ Uses the glove word vectors to convert whole sentences to single
    vectors by averaging the input vectors.

    :param sentences: list
        List if sentences, where each element is a list of words as strings.
        Example: [['first', 'sentence'], ['second', 'sentence'],...]

    :param glove: dictionary
        {'word': vector}, where vector is 'n_features'-dimensional.

    :return: list
        List of sentences as a single vector.
    """
    sentence_vectors = []
    for sentence in sentences:
        matched_words = len(sentence)

        # All vectors have the same dimensionality. Using the most common
        # 'word' to set the size of our new sentence vector.
        sum_of_words = np.zeros(len(glove[',']))

        # Add a neutral word if we don't have its glove vector. This works
        # faster than checking to see if we have it and then adjusting
        # matched_words to take the correct average.
        for word in sentence:
            sum_of_words += glove.get(word, 0)
        # In theory, we should have sentences that have at least 1 element.
        # However for some datasets, such as CR, the if statement is necessary,
        # to ensure we don't divide by zero.
        if matched_words != 0:
            sentence_vector = sum_of_words/matched_words
            sentence_vectors.append(sentence_vector)
    return sentence_vectors


def load_data(type, loc='./data/'):
    """Loads data from the directory, loc.

    :param type: string  # Can I do this to speed things up?
        MR = movie review sentiment
        SO = subjective/objective classification
        CR = product reviews
        MPQA = opinion polarity

    :param loc: string
        Locates the directory in which the data is saved.

    :return: pos, list
        Contains a list of all the positive sentence examples as embedded lists
        of word strings. Example:
        [['first', 'pos', 'sentence'], ['second', 'pos', 'sentence'],...]

    :return: neg, list
        Contains a list of all the negative sentence examples as embedded lists
        of word strings. Example:
        [['first', 'neg', 'sentence'], ['second', 'neg', 'sentence'],...]

    """
    if type == 'MR':
        pos, neg = load_mr(loc)
    if type == 'SO':
        pos, neg = load_so(loc)
    if type == 'CR':
        pos, neg = load_cr(loc)
    if type == 'MPQA':
        pos, neg = load_mpqa(loc)
    return pos, neg


def load_so(loc):
    """Loads the objective vs subjective data, setting positive examples as
    objective examples

    :param loc: string
        Locates the directory in which the data is saved.

    :return: pos, list
        Contains a list of all the positive sentence examples as embedded lists
        of word strings. Example:
        [['first', 'pos', 'sentence'], ['second', 'pos', 'sentence'],...]

    :return: neg, list
        Contains a list of all the negative sentence examples as embedded lists
        of word strings. Example:
        [['first', 'neg', 'sentence'], ['second', 'neg', 'sentence'],...]

    """
    pos, neg = [], []
    with open(loc + 'plot.tok.gt9.5000.txt', 'rb') as f:
        for line in f:
            pos.append(word_tokenize(line.decode('latin-1')))
    with open(loc + 'quote.tok.gt9.5000.txt', 'rb') as f:
        for line in f:
            neg.append(word_tokenize(line.decode('latin-1')))
    return pos, neg


def load_mr(loc):
    """Loads the positive vs negative movie reviews, setting positive
    examples as the positive movie reviews.

    :param loc: string
        Locates the directory in which the data is saved.

    :return: pos, list
        Contains a list of all the positive sentence examples as embedded lists
        of word strings. Example:
        [['first', 'pos', 'sentence'], ['second', 'pos', 'sentence'],...]

    :return: neg, list
        Contains a list of all the negative sentence examples as embedded lists
        of word strings. Example:
        [['first', 'neg', 'sentence'], ['second', 'neg', 'sentence'],...]

    """
    pos, neg = [], []
    # why are we using 'rb'?
    with open(loc + 'rt-polarity.neg.txt', 'rb') as f:
        for line in f:
            # adds each sentence as a list, which is itself a list of words
            # Why did I have to use 'latin-1'? why not 'utf-8'...read about this!
            neg.append(word_tokenize(line.decode('latin-1')))
    with open(loc + 'rt-polarity.pos.txt', 'rb') as f:
        for line in f:
            pos.append(word_tokenize(line.decode('latin-1')))
    return pos, neg


def load_cr(loc):
    """Loads the positive vs negative customer reviews, setting positive
    examples as the positive customer reviews.

    :param loc: string
        Locates the directory in which the data is saved.

    :return: pos, list
        Contains a list of all the positive sentence examples as embedded lists
        of word strings. Example:
        [['first', 'pos', 'sentence'], ['second', 'pos', 'sentence'],...]

    :return: neg, list
        Contains a list of all the negative sentence examples as embedded lists
        of word strings. Example:
        [['first', 'neg', 'sentence'], ['second', 'neg', 'sentence'],...]

    """
    pos, neg = [], []
    with open(loc + 'custrev.neg.txt', 'rb') as f:
        for line in f:
            neg.append(word_tokenize(line))
    with open(loc + 'custrev.pos.txt', 'rb') as f:
        for line in f:
            pos.append(word_tokenize(line))
    return pos, neg


def load_mpqa(loc):
    """Loads the opinion polarity data in to positive and negative examples.

    :param loc: string
        Locates the directory in which the data is saved.

    :return: pos, list
        Contains a list of all the positive sentence examples as embedded lists
        of word strings. Example:
        [['first', 'pos', 'sentence'], ['second', 'pos', 'sentence'],...]

    :return: neg, list
        Contains a list of all the negative sentence examples as embedded lists
        of word strings. Example:
        [['first', 'neg', 'sentence'], ['second', 'neg', 'sentence'],...]

    """
    pos, neg = [], []
    with open(loc + 'mpqa.pos.txt', 'rb') as f:
        for line in f:
            pos.append(word_tokenize(line))
    with open(loc + 'mpqa.neg.txt', 'rb') as f:
        for line in f:
            neg.append(word_tokenize(line))
    return pos, neg


def minibatch(rand, X_data, y_data, n_samples, batch_size):
    """ Takes a random batch of rows (examples) from X_data and y_data.

    Parameters
    ----------
    rand : RandomState numpy object

    X_data : array-like, shape (n_samples, n_features)
        Training samples.

    y_data : array-like, shape (n_samples, 1)
        Training labels.

    n_samples : int
        Number of samples in X_data (or, equivalently, in y_data).

    batch_size : int
        The size of the batch to be returned. i.e the number of examples (rows)
        to be returned

    Returns
    -------
    X_batch : array-like, shape (batch_size, n_features)
        Randomly selected set of training samples from X_data.

    y_batch : array-like, shape (batch_size, 1)
        Corresponding labels for the randomly selected examples from X_data.
    """
    indices = rand.choice(n_samples, batch_size)
    X_batch, y_batch = X_data[indices], y_data[indices]
    return X_batch, y_batch


def plot_curve(clf, X_train, y_train, title, label='label', color='b', cv=3,
               figure=1):
    """ Plot curves using cross-validation set for hyperparmeter evaluation.

    This function uses k-fold cross-validation (default is 3-fold) to plot
    learning curves.

    Parameters
    ----------
    clf : Classifier

    X_train : array-like, shape (n_samples, n_features)
        Training samples.

    y_train : array-like, shape (n_samples, 1)
        Training labels.

    title : string
        Title of the chart.

    label : string, default: 'label'
        Label corresponding to the curve which is to be plotted.

    color : string, default: 'b'

    cv : integer, default: 3
        Number of 'folds' in our cross-validation training split.

    figure : integer, default:1
        Sets the current figure.

    Returns
    -------
    plt : matplotlib object
    """
    plt.figure(figure)
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid()

    # Must be a faster way of doing this (we only need the cv_scores).
    train_sizes, train_scores, cv_scores = learning_curve(clf, X_train,
                                                          y_train, cv=cv)
    cv_scores_mean = np.mean(cv_scores, axis=1)
    plt.plot(train_sizes, cv_scores_mean, 'o-', color=color, label=label)
    plt.legend(loc="best")
    return plt
