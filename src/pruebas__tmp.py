from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_selection import chi2
from feature_scoring import ig, bns
import numpy as np
import time


def main():
    newsgoups = fetch_20newsgroups(subset='train', categories=['sci.crypt', 'talk.politics.guns'])

    vectorizer = CountVectorizer()
    vector = vectorizer.fit_transform(newsgoups.data, newsgoups.target)
    vocab = np.array(vectorizer.get_feature_names())
    print "number of positive examples:", np.sum(newsgoups.target)

    t0 = time.time()
    ig_scores, _ = ig(vector, newsgoups.target)
    print "Information Gain top 50  scored terms:"
    print vocab[np.argsort(ig_scores)][-50:]
    print "time: %.4f secs" % (time.time()-t0)

    t0 = time.time()
    bns_scores, _ = bns(vector, newsgoups.target)
    print "Bi-Normal Separation top 50  scored terms:"
    print vocab[np.argsort(bns_scores)][-50:]
    print "time: %.4f secs" % (time.time()-t0)

    t0 = time.time()
    chi2_scores, _ = chi2(vector, newsgoups.target)
    print "Chi Squared top 50  scored terms:"
    print vocab[np.argsort(chi2_scores)][-50:]
    print "time: %.4f secs" % (time.time()-t0)

if __name__ == '__main__':
    main()
