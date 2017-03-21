To use the script first save a copy of the glove vectors as glove.840B.300d.txt (http://nlp.stanford.edu/data/glove.840B.300d.zip) and place them into the working directory.
Then just type the following in to the command line, replacing ’SO’ with whichever dataset (“SO”, “MR”, “MPQA”, “CR”) you want to run the script on.

Add -s to remove stopwords from the data
Add -p followed by the number of PCs if you would like to preprocess using PCA

$ python binary_supervised_learning.pg -i SO