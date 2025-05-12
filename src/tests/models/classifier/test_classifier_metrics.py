import unittest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix
import seaborn as sns
import torch

from data.mnist import get_mnist_dataloaders
from data.cifar10 import get_cifar10_dataloaders
from classes.metrics import Metrics
from models.classifier.classifier import Classifier
from utils.device import DEVICE
from utils.plotter import Plotter

class TestClassifierMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mnist_model = Classifier(dataset="MNIST").to(DEVICE)
        cls.cifar10_model = Classifier(dataset="CIFAR10").to(DEVICE)
        
        cls.mnist_train, cls.mnist_test = get_mnist_dataloaders(batch_size=1000)
        cls.cifar10_train, cls.cifar10_test = get_cifar10_dataloaders(batch_size=1000)

        cls.output_dir = Path(__file__).parent.parent.parent / 'output'
        cls.output_dir.mkdir(exist_ok=True)

    def test_metrics_calculations(self):
        """Test all metrics calculations with sample values"""
        test_cases = [
            # tp, tn, fp, fn
            (50, 30, 10, 10),  # Normal case
            (0, 0, 0, 0),       # All zeros
            (100, 0, 0, 0),     # Perfect positive prediction
            (0, 100, 0, 0),     # Perfect negative prediction
        ]
        
        for tp, tn, fp, fn in test_cases:
            with self.subTest(tp=tp, tn=tn, fp=fp, fn=fn):
                total = tp + tn + fp + fn
                if total > 0:
                    self.assertAlmostEqual(
                        Metrics.accuracy(tp, tn, fp, fn) + Metrics.error_rate(tp, tn, fp, fn),
                        1.0,
                        places=4
                    )
                
                # Test precision-recall relationship
                precision = Metrics.precision(tp, tn, fp, fn)
                recall = Metrics.recall(tp, tn, fp, fn)
                if precision + recall > 0:
                    f1 = Metrics.f1_score(tp, tn, fp, fn)
                    self.assertAlmostEqual(
                        f1,
                        Metrics.fbeta_score(tp, tn, fp, fn, beta=1.0),
                        places=4
                    )

    def generate_confusion_matrix(self, model, dataloader, dataset_name):
        """Generate and save confusion matrix for a dataset"""
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        cm = confusion_matrix(all_labels, all_preds)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Get class names
        class_names = [str(i) for i in range(10)] if dataset_name == "MNIST" else [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f',
                    xticklabels=class_names,
                    yticklabels=class_names,
                    cmap='Blues')
        plt.title(f'Normalized Confusion Matrix ({dataset_name})')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        plot_path = self.output_dir / f'confusion_matrix_{dataset_name.lower()}.png'
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        return plot_path

    def test_mnist_confusion_matrix(self):
        """Test and save MNIST confusion matrix"""
        plot_path = self.generate_confusion_matrix(
            self.mnist_model,
            self.mnist_test,
            "MNIST"
        )
        print(f"\nMNIST confusion matrix saved to: {plot_path}")
        self.assertTrue(plot_path.exists())

    def test_cifar10_confusion_matrix(self):
        """Test and save CIFAR10 confusion matrix"""
        plot_path = self.generate_confusion_matrix(
            self.cifar10_model,
            self.cifar10_test,
            "CIFAR10"
        )
        print(f"\nCIFAR10 confusion matrix saved to: {plot_path}")
        self.assertTrue(plot_path.exists())

if __name__ == "__main__":
    unittest.main()