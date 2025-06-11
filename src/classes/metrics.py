class Metrics:
    @staticmethod
    def accuracy(tp, tn, fp, fn):
        return ((tp + tn) / (tp + tn + fp + fn) * 100) if (tp + tn + fp + fn) != 0 else 0.0

    @staticmethod
    def error_rate(tp, tn, fp, fn):
        return (fp + fn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0.0

    @staticmethod
    def precision(tp, tn, fp, fn):
        return tp / (tp + fp) if (tp + fp) != 0 else 0.0

    @staticmethod
    def recall(tp, tn, fp, fn):  # Sensitivity
        return tp / (tp + fn) if (tp + fn) != 0 else 0.0

    @staticmethod
    def specificity(tp, tn, fp, fn):
        return tn / (tn + fp) if (tn + fp) != 0 else 0.0

    @staticmethod
    def negative_predictive_value(tp, tn, fp, fn):
        return tn / (tn + fn) if (tn + fn) != 0 else 0.0

    @staticmethod
    def false_positive_rate(tp, tn, fp, fn):
        return fp / (fp + tn) if (fp + tn) != 0 else 0.0

    @staticmethod
    def false_negative_rate(tp, tn, fp, fn):
        return fn / (fn + tp) if (fn + tp) != 0 else 0.0

    @staticmethod
    def f1_score(tp, tn, fp, fn):
        precision = Metrics.precision(tp, tn, fp, fn)
        recall = Metrics.recall(tp, tn, fp, fn)
        return (
            2 * precision * recall / (precision + recall)
            if (precision + recall) != 0
            else 0.0
        )

    @staticmethod
    def fbeta_score(tp, tn, fp, fn, beta=1.0):
        precision = Metrics.precision(tp, tn, fp, fn)
        recall = Metrics.recall(tp, tn, fp, fn)
        beta_squared = beta**2
        numerator = (1 + beta_squared) * precision * recall
        denominator = beta_squared * precision + recall
        return numerator / denominator if denominator != 0 else 0.0
