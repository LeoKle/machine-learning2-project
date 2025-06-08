import unittest
from pathlib import Path
import torch
from utils.plotter import Plotter
from pipeline.train_classifier_linear import train_classifier_linear
from data.mnist import get_mnist_dataloaders
from data.cifar10 import get_cifar10_dataloaders

class TestLinearOnEncodedTraining(unittest.TestCase):
    def test_training_and_saving(self):
        dataset_type = "MNIST"

        if dataset_type.upper() == "MNIST":
            train_loader, test_loader = get_mnist_dataloaders(batch_size=64)
            dummy_input = torch.randn(1, 1, 28, 28)
            encoder_path = "resultsAECNN2_MNIST/MNIST_encoder_weights.pth"
        elif dataset_type.upper() == "CIFAR10":
            train_loader, test_loader = get_cifar10_dataloaders(batch_size=128)
            dummy_input = torch.randn(1, 3, 32, 32)
            encoder_path = "resultsAECNN2_CIFAR10/CIFAR10_encoder_weights.pth"
        else:
            raise ValueError("Unsupported dataset_type. Use 'MNIST' or 'CIFAR10'.")

        _, pipeline = train_classifier_linear(train_loader, test_loader, dummy_input, encoder_path)

        model_save_dir = Path(f"output/results_linear_encoded_{dataset_type}_paths")
        plot_save_dir = Path(f"output/results_linear_encoded_{dataset_type}_pngs")
        model_save_dir.mkdir(parents=True, exist_ok=True)
        plot_save_dir.mkdir(parents=True, exist_ok=True)

        with open(plot_save_dir / "epoch_metrics.txt", "w") as f:
            #f.write("Epoch | Train Loss | Test Loss | Test Accuracy\n")
            f.write("Epoch | Train Loss | Test Loss  | Accuracy | Error Rate | Precision | Recall | Specificity |   NPV  |   FPR  |   FNR  | F1 Score | Fbeta Score\n")
            #f.write("------|------------|-----------|----------------\n")

        test_losses = []
        extra_epochs = 0
        triggered_extra = False
        epoch = 1

        while True:
            pipeline.current_epoch = epoch
            pipeline.model.train()

            total_loss = 0
            total_batches = 0
            for batch, labels in pipeline.dataloader_train:
                loss = pipeline.train_epoch(batch, labels)
                total_loss += loss
                total_batches += 1

            avg_train_loss = total_loss / total_batches
            pipeline.tracker.track("train_loss", avg_train_loss, epoch)

            test_loss = pipeline.evaluate()
            metrics = pipeline.tracker.get_metrics()
            accuracy     = metrics["accuracy"][-1]
            error_rate   = metrics["error_rate"][-1]
            precision    = metrics["precision"][-1]
            recall       = metrics["recall"][-1]
            specificity  = metrics["specificity"][-1]
            npv          = metrics["NPV"][-1]
            fpr          = metrics["FPR"][-1]
            fnr          = metrics["FNR"][-1]
            f1_score     = metrics["f1_score"][-1]
            fbeta_score  = metrics["fbeta_score"][-1]
            with open(plot_save_dir / "epoch_metrics.txt", "a") as f:
                f.write(f"{epoch:>5} |  {avg_train_loss:.4f}    |   {test_loss:.4f}   |  {accuracy:.2f}%  |   "
                        f"{error_rate:.4f}   |  {precision:.4f}   | {recall:.4f} |   {specificity:.4f}    | "
                        f"{npv:.4f} | {fpr:.4f} | {fnr:.4f} |  {f1_score:.4f}  | {fbeta_score:.4f}\n")
            test_losses.append(test_loss)
            print(f"[{dataset_type}] Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Test Loss = {test_loss:.4f}, Test Acc = {accuracy:.2f}%")

            if epoch % 10 == 0:
                torch.save(pipeline.model.state_dict(), model_save_dir / f"classifier_epoch_{epoch}.pth")

            if not triggered_extra and epoch >= 3:
                if test_losses[-1] > test_losses[-2] > test_losses[-3]:
                    print(f"Test loss increased consecutively. Saving best checkpoint at epoch {epoch}.")
                    torch.save(pipeline.model.state_dict(), model_save_dir / f"best_classifier_epoch_{epoch}.pth")
                    Plotter.plot_loss_progression(pipeline.tracker.get_metrics(), list(range(1, epoch + 1)), plot_save_dir)
                    extra_epochs = 3
                    triggered_extra = True

            if triggered_extra:
                extra_epochs -= 1
                if extra_epochs == 0:
                    print(f"Stopping after {extra_epochs} extra epochs. Final epoch: {epoch}")
                    torch.save(pipeline.model.state_dict(), model_save_dir / f"classifier_final_epoch_{epoch}.pth")
                    Plotter.plot_loss_progression(pipeline.tracker.get_metrics(), list(range(1, epoch + 1)), plot_save_dir)
                    break

            epoch += 1

        Plotter.plot_metrics(pipeline.tracker.get_metrics(), plot_save_dir / "classifier_metrics.png")
        Plotter.plot_accuracy(accuracy_values=pipeline.tracker.get_metrics()["test_accuracy"], output_file_name=plot_save_dir / "classifier_accuracy.png")

        for e in [10]:
            self.assertTrue((model_save_dir / f"classifier_epoch_{e}.pth").exists())

        self.assertTrue((plot_save_dir / "classifier_metrics.png").exists())

if __name__ == "__main__":
    unittest.main()
