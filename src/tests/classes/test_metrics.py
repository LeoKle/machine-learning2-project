import unittest
from math import isclose

from classes.metrics import Metrics


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.tp = 50
        self.tn = 30
        self.fp = 10
        self.fn = 10

    def test_accuracy(self):
        result = Metrics.accuracy(self.tp, self.tn, self.fp, self.fn)
        expected = (
            (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn) * 100
        )  # Convert to percentage
        self.assertTrue(isclose(result, expected))

    def test_error_rate(self):
        result = Metrics.error_rate(self.tp, self.tn, self.fp, self.fn)
        expected = (self.fp + self.fn) / (self.tp + self.tn + self.fp + self.fn)
        self.assertTrue(isclose(result, expected))

    def test_precision(self):
        result = Metrics.precision(self.tp, self.tn, self.fp, self.fn)
        expected = self.tp / (self.tp + self.fp)
        self.assertTrue(isclose(result, expected))

    def test_recall(self):
        result = Metrics.recall(self.tp, self.tn, self.fp, self.fn)
        expected = self.tp / (self.tp + self.fn)
        self.assertTrue(isclose(result, expected))

    def test_specificity(self):
        result = Metrics.specificity(self.tp, self.tn, self.fp, self.fn)
        expected = self.tn / (self.tn + self.fp)
        self.assertTrue(isclose(result, expected))

    def test_negative_predictive_value(self):
        result = Metrics.negative_predictive_value(self.tp, self.tn, self.fp, self.fn)
        expected = self.tn / (self.tn + self.fn)
        self.assertTrue(isclose(result, expected))

    def test_false_positive_rate(self):
        result = Metrics.false_positive_rate(self.tp, self.tn, self.fp, self.fn)
        expected = self.fp / (self.fp + self.tn)
        self.assertTrue(isclose(result, expected))

    def test_false_negative_rate(self):
        result = Metrics.false_negative_rate(self.tp, self.tn, self.fp, self.fn)
        expected = self.fn / (self.fn + self.tp)
        self.assertTrue(isclose(result, expected))

    def test_f1_score(self):
        precision = self.tp / (self.tp + self.fp)
        recall = self.tp / (self.tp + self.fn)
        expected = 2 * precision * recall / (precision + recall)
        result = Metrics.f1_score(self.tp, self.tn, self.fp, self.fn)
        self.assertTrue(isclose(result, expected))

    def test_fbeta_score(self):
        beta = 2
        precision = self.tp / (self.tp + self.fp)
        recall = self.tp / (self.tp + self.fn)
        beta_squared = beta**2
        expected = (
            (1 + beta_squared)
            * precision
            * recall
            / (beta_squared * precision + recall)
        )
        result = Metrics.fbeta_score(self.tp, self.tn, self.fp, self.fn, beta=beta)
        self.assertTrue(isclose(result, expected))

    def test_zero_divisions(self):
        self.assertEqual(Metrics.accuracy(0, 0, 0, 0), 0.0)
        self.assertEqual(Metrics.error_rate(0, 0, 0, 0), 0.0)
        self.assertEqual(Metrics.precision(0, 0, 0, 0), 0.0)
        self.assertEqual(Metrics.recall(0, 0, 0, 0), 0.0)
        self.assertEqual(Metrics.specificity(0, 0, 0, 0), 0.0)
        self.assertEqual(Metrics.negative_predictive_value(0, 0, 0, 0), 0.0)
        self.assertEqual(Metrics.false_positive_rate(0, 0, 0, 0), 0.0)
        self.assertEqual(Metrics.false_negative_rate(0, 0, 0, 0), 0.0)
        self.assertEqual(Metrics.f1_score(0, 0, 0, 0), 0.0)
        self.assertEqual(Metrics.fbeta_score(0, 0, 0, 0, beta=2), 0.0)


if __name__ == "__main__":
    unittest.main()
