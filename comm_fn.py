from __future__ import print_function
import os.path
import argparse

import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.preprocessing import StandardScaler


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
    path = './data/gloveFile_done_%d.npy' % (word_count)

    # Saves time by loading existing file, if available.
    if os.path.exists(path):
        glove = np.load(path).item()
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


def load_data(type, loc='./data/', remove_stop=False):
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
    # Remove stop words
    if remove_stop == True:
        pos = remove_stop_words(pos)
        neg = remove_stop_words(neg)
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


def minibatch(rand, X_data, y_data, batch_size):
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
    indices = rand.choice(X_data.shape[0], batch_size)
    X_batch, y_batch = X_data[indices], y_data[indices]
    return X_batch, y_batch


def plot_curve(clf, X_train, y_train, title, train_sizes, label='label', color='b', cv=3,
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

    train_sizes, train_scores, cv_scores = learning_curve(
        clf, X_train, y_train, train_sizes=train_sizes, cv=cv)
    cv_scores_mean = np.mean(cv_scores, axis=1)
    plt.plot(train_sizes, cv_scores_mean, 'o-', color=color, label=label)
    plt.legend(loc="best")
    return plt

def remove_stop_words(full_sentences):
    """ Removes stopwords from full_sentences.
    Parameters
    ----------
    full_sentences: list
        Each item in the list represents a sentence. Each sentence contains
        tokenized words.
    Returns
    -------
    filtered_sentences : list
        New list of sentences with stopwords removed.
    """
    stop_words = stopwords.words('english')
    stop_words.extend((',', '.'))  # Removing punctuation increases accuracy.
    filtered_sentences = []
    for sentence in full_sentences:
        filtered_sentence = [w for w in sentence
                             if w.lower() not in stop_words]
        filtered_sentences.append(filtered_sentence)
    return filtered_sentences


def save_misclf_data(pos_ind, neg_ind, pos_sentences, neg_sentences, type):
    """
    Saves misclassified samples in two seperate text documents.
    :param pos_ind: list
        Indices of all positive samples that were classified as negative.
    :param neg_ind:  list
        Indices of all negative samples that were classified as positive.
    :param pos_sentences: list
        List of all positive samples.
    :param neg_sentences: list
            List of all positive samples.
    :param type: string
        Type of supervised learning problem e.g. 'SO'
    :return:
    """
    pos_new = []
    for sentence in pos_sentences:
        # Make sentences easier to read
        string_sent = [x.encode('UTF8') for x in sentence]
        string_sent = " ".join(string_sent)
        pos_new.append(string_sent)

    # These sentences should have been classified as negative
    misclf_pos = [pos_new[i] for i in pos_ind]

    neg_new = []
    for sentence in neg_sentences:
        string_sent = [x.encode('UTF8') for x in sentence]
        string_sent = " ".join(string_sent)
        neg_new.append(string_sent)

    # These sentences should have been classified as positive
    misclf_neg = [neg_new[i-len(pos_sentences)] for i in neg_ind]

    np.savetxt('./results/test_pos_%s.txt' % (type), misclf_pos,
               delimiter='.', fmt='%s')
    np.savetxt('./results/test_neg_%s.txt' % (type), misclf_neg,
               delimiter='.', fmt='%s')


def PC_comparision(X_train, X_test):
    """
    Preprocesses training and test data with PCA. Using 2, 5, 10, 20, 50, 100,
    150, 200, 250 and 300 PCs.

    :param X_train: array-like, shape (n_samples, n_features)
    :param X_test: array-like, shape (n_samples, n_features)
    :return: dictionary
        Two dictionaries each in the same format. The number of PCs used
        to construct the array as the key and the array itself (test or
        training depending on dictionary) is the value.
    """
    PCs = [2, 5, 10, 20, 50, 100, 150, 200, 250, 300]
    X_train_PCs = {}
    X_test_PCs = {}
    for pc in PCs:
        pca = PCA(n_components=pc)
        pca.fit(X_train)
        X_train1 = pca.transform(X_train)
        X_test1 = pca.transform(X_test)
        X_train_PCs[pc] = X_train1
        X_test_PCs[pc] = X_test1

    print('Collected all PCs')
    return X_train_PCs, X_test_PCs


def process_pca(n_components, X_train, X_test):
    # Optional PCA preprocessing
    pca = PCA(n_components)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    return X_train, X_test

def standard(args):
    # Dictionary {word:vector}, where each word is a key corresponding to a
    # 'n_features'-d row vector, shape (n_features,).
    model = load_glove_model('./data/glove.840B.300d.txt')

    # Load data in to lists of positive and negative examples.
    pos, neg = load_data('SO', remove_stop=False)

    # Shape (n_pos_samples, n_features)
    pos_vectors = np.asarray(sentences2vec(pos, model))
    # Shape (n_neg_samples, n_features)
    neg_vectors = np.asarray(sentences2vec(neg, model))

    # Prepare a matrix containing both positive and negative samples
    X = np.r_[pos_vectors, neg_vectors]
    y = np.zeros(X.shape[0])
    y[:pos_vectors.shape[0]] = 1.0

    X_with_ind = np.c_[
        range(X.shape[0]), X]  # Using 'range' to produce an index

    # Randomly split the X and y arrays into 30/70 test/train split.
    X_train, X_test, y_train, y_test = train_test_split(X_with_ind,
                                                        y,
                                                        test_size=0.3,
                                                        random_state=0)
    # Index to convert from X_test (shuffled data) back to X.
    # For example, if sample 3 in y_test is missclassified, we use the third value
    # from X_ind to get back the original index of the example. If this value is
    # 300, this tells us that the 300th example has been missclassified.
    X_ind = X_test[:, 0]

    X_train = X_train[:, 1:]
    X_test = X_test[:, 1:]

    # Standardize features by removing the mean and scaling to unit variance
    sc = StandardScaler()
    sc.fit(X_train)
    X_train = sc.transform(X_train)
    X_test = sc.transform(X_test)

    # Optional PCA preprocessing
    if args.pca: X_train, X_test = process_pca(args.pca, X_train, X_test)

    return X_train, X_test, y_train, y_test, X_ind

def parse_options():
    # Handles the command-line argument, which specifies the dataset to be used.
    parser = argparse.ArgumentParser(description='Choose data to load into the'
                                                 ' Logistic Regression'
                                                 ' classifier.')
    parser.add_argument('-i','--input',
                        help='Specify input data: "MR", "SO", "CR", "MPQA"',
                        choices=["MR", "SO", "CR", "MPQA"],
                        required=True)
    parser.add_argument('-p','--pca',
                        help='Specify the number of Principal components.',
                        type=int)
    parser.add_argument('-s', '--stop', action='store_true',
                        help='Using the -s or --stop flag, removes the stopword '
                             'from the data.')
    parser.add_argument('-d', '--data',
                        help='Specify the dataset size you would like to use for '
                             'producing learning curves.',
                        type=int,
                        nargs='+')

    return parser.parse_args()
