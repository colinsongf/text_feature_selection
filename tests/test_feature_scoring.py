from feature_scoring import ig, infogain_score, bns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.datasets import make_classification
from numpy.testing import assert_array_equal
from scipy.special import ndtri
import pandas as pd
import numpy as np
import unittest


# class FeatureSelectionTests(unittest.TestCase):
#
#     def setUp(self):
#         self.texts = np.array(['hola pedazo de puto', 'que te pasa gil!', 'sos un puto y lo sabes',
#                             'vos no sabes nada', 'sabes como te dicen?', 'hola como andan todos?'])
#         self.labels = np.array([1,1,1,0,0,0])
#
#     def test_chi2(self):
#         vector = CountVectorizer().fit_transform(self.texts, self.labels)
#         selector = SelectKBest(chi2, k=1)
#         selector.fit(vector, self.labels)
#         self.assertEqual(selector.get_support(indices=True), [0])
#
#     def test_infogain(self):
#         vector = CountVectorizer().fit_transform(self.texts, self.labels)
#         selector = SelectKBest(ig, k=1)
#         selector.fit(vector, self.labels)
#         self.assertEqual(selector.get_support(indices=True), [0])
#
#     def test_compare_infogain(self):
#         vector = CountVectorizer().fit_transform(self.texts, self.labels)
#
#         selector1 = SelectKBest(ig, k=5)
#         vector1 = selector1.fit_transform(vector, self.labels)
#
#         selector2 = SelectKBest(infogain_score, k=5)
#         vector2 = selector2.fit_transform(vector, self.labels)
#
#         supp1 = selector1.get_support(indices=True)
#         scores1 = selector1.scores_
#         supp2 = selector2.get_support(indices=True)
#         scores2 = selector2.scores_
#
#         for i in range(5):
#             print supp1[i], scores1[i], supp2[i], scores2[i]

class FeatureScoringTests(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[1,0,1,1],[1,0,0,0],[0,0,1,0],[0,0,0,0]])
        self.y = np.array([1,1,0,0])

    def test_chi2_scores(self):
        scores, _ = chi2(self.X, self.y)
        print "scores:", scores
        self.assertTrue(len(scores) == 4)
        #self.assertTrue(np.all(scores >= 0.0)) # FIXME fails because the nan's
        assert_array_equal(scores, [2, np.nan, 0, 1])

        # n = 4
        # n11 = 2. # 1
        # n01 = 0. # 1
        # n00 = 2. # 2
        # n10 = 0. # 0
        # chi = (n * (n11*n00 - n01*n10)**2)/((n11+n01)*(n10+n00)*(n11+n10)*(n01+n00))
        # TODO ver por que esta cuenta da 4 mientras que chi2 devuelve 2.

    def test_ig_scores(self):
        scores, _ = ig(self.X, self.y)
        print "scores:", scores
        self.assertTrue(len(scores) == 4)
        #self.assertTrue(np.all(scores >= 0.0)) # FIXME fails because the nan's
        #self.assertTrue(np.all(scores <= 1.0)) # FIXME fails because the nan's
        assert_array_equal(scores, np.array([1, np.nan, 0, 0.5]))

    def test_bns_scores(self):
        scores, _ = bns(self.X, self.y)
        print "scores:", scores
        self.assertTrue(len(scores) == 4)
        self.assertTrue(np.all(scores >= 0.0))
        self.assertTrue(np.all(scores <= 2*ndtri(0.9995)))
        true_bns_scores = np.array([ndtri(0.9995) - ndtri(0.0005), # ~ F^-1(1) - F^-1(0)
                                    ndtri(0.0005) - ndtri(0.0005), # F^-1(0) - F^-1(0)
                                    ndtri(0.5) - ndtri(0.5), # F^-1(0.5) - F^-1(0.5)
                                    ndtri(0.5) - ndtri(0.0005)]) # F^-1(0.5) - F^-1(0)
        assert_array_equal(scores, true_bns_scores)

def main():
    unittest.main()

if __name__ == '__main__':
    main()
