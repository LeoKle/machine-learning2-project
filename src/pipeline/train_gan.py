import torch
import torch.nn as nn

from classes.tracker import Tracker
from utils.device import DEVICE
from utils.plotter import Plotter
from metrics.metrics_gan import GanMetrics


class GanTrainingPipeline:
    def __init__(
        self,
        dataloader_train: torch.utils.data.DataLoader,
        dataloader_test: torch.utils.data.DataLoader,
        generator: nn.Module,
        discriminator: nn.Module,
        loss_function_discriminator: torch.nn.modules.loss._Loss,
        loss_function_generator: torch.nn.modules.loss._Loss,
        optimizer_discriminator: torch.optim.Optimizer,
        optimizer_generator: torch.optim.Optimizer,
        output_dir: str = "output",
    ):
        self.dataloader_train = dataloader_train
        self.dataloader_test = dataloader_test

        self.generator = generator.to(DEVICE)
        self.discriminator = discriminator.to(DEVICE)

        self.loss_function_discriminator = loss_function_discriminator
        self.loss_function_generator = loss_function_generator
        self.optimizer_discriminator = optimizer_discriminator
        self.optimizer_generator = optimizer_generator

        self.metrics = GanMetrics()
        self.is_image_count = 5000  # Number of images to compute Inception Score
        self.is_batch_size = 100  # Batch size for generating images for Inception Score

        self.output_dir = output_dir

        self.tracker = Tracker(output=self.output_dir + "/gan_metrics.json")

    def train_discriminator(self, batch: torch.Tensor):
        self.discriminator.zero_grad()
        batch_size = batch.shape[0]

        # train discriminator on real data
        label_real = torch.ones(batch_size, 1, device=DEVICE) * 0.9

        noise_image = batch  # + torch.randn_like(batch) * 0.05
        output_real = self.discriminator(noise_image)

        loss_real = self.loss_function_discriminator(output_real, label_real)

        # train discriminator on generated data
        label_fake = torch.zeros(batch_size, 1, device=DEVICE)  # * 0.1
        latent_tensor = torch.randn(
            batch_size, self.generator.latent_dim, device=DEVICE
        )
        generated_images = self.generator(latent_tensor)

        output_generated = self.discriminator(generated_images)
        loss_generated = self.loss_function_discriminator(output_generated, label_fake)

        total_loss = loss_real + loss_generated
        self.discriminator_loss.append(total_loss)

        total_loss.backward()
        self.optimizer_discriminator.step()

    def train_generator(self, batch: torch.Tensor):
        self.generator.zero_grad()
        batch_size = batch.shape[0]

        latent_tensor = torch.randn(
            batch_size, self.generator.latent_dim, device=DEVICE
        )
        label_real = torch.ones(batch_size, 1, device=DEVICE)

        generator_output = self.generator(latent_tensor)
        discriminator_output = self.discriminator(generator_output)

        loss = self.loss_function_generator(discriminator_output, label_real)
        self.generator_loss.append(loss)

        loss.backward()
        self.optimizer_generator.step()

    def show_current_images(self, epoch):
        with torch.no_grad():
            self.generator.eval()
            image_count = 10
            latent_tensor = torch.randn(
                image_count, self.generator.latent_dim, device=DEVICE
            )

            generator_output = self.generator(latent_tensor)

            Plotter.show_image(
                generator_output, output_file_name=self.output_dir + f"/gan_{epoch}.png"
            )
            Plotter.show_image(generator_output, output_file_name="output/#latest.png")

    def train(self, epoch_count):
        for epoch in range(1, epoch_count + 1):
            print("Training epoch", epoch)
            self.generator.train()
            self.discriminator.train()
            self.metrics.reset()

            self.generator_loss = []
            self.discriminator_loss = []

            for _, (x, _) in enumerate(self.dataloader_train):
                x = x.to(DEVICE)
                self.train_generator(x)
                self.train_discriminator(x)

            generator_loss = torch.tensor(self.generator_loss).mean()
            discriminator_loss = torch.tensor(self.discriminator_loss).mean()

            self.tracker.track("generator_loss", generator_loss.item(), epoch)
            self.tracker.track("discriminator_loss", discriminator_loss.item(), epoch)

            print(f"Generator Loss: {generator_loss}")
            print(f"Discriminator Loss: {discriminator_loss}")

            self.show_current_images(epoch)

            # only evaluate every 10 epochs or on the last epoch
            if epoch % 10 != 0 and epoch != epoch_count:
                continue

            print(f"Evaluating epoch {epoch}...")
            with torch.no_grad():
                self.generator.eval()
                self.discriminator.eval()

                real_preds = []
                real_targets = []

                fake_preds = []
                fake_targets = []

                for _, (x, _) in enumerate(self.dataloader_test):
                    x = x.to(DEVICE)
                    real_out = self.discriminator(x)
                    real_preds.extend((real_out > 0.5).float().cpu())
                    real_targets.extend(torch.ones_like(real_out).cpu())

                    z = torch.randn(x.size(0), self.generator.latent_dim, device=DEVICE)
                    fake_imgs = self.generator(z)
                    fake_out = self.discriminator(fake_imgs)
                    fake_preds.extend((fake_out > 0.5).float().cpu())
                    fake_targets.extend(torch.zeros_like(fake_out).cpu())

                    self.metrics.update_fid(fake_imgs, real=False)
                    self.metrics.update_fid(x, real=True)

                for _ in range(0, self.is_image_count, self.is_batch_size):
                    z = torch.randn(
                        self.is_batch_size, self.generator.latent_dim, device=DEVICE
                    )
                    fake_imgs = self.generator(z)
                    # sum IS score:
                    self.metrics.update_is(fake_imgs)

                # Compute Inception Score
                mean, std, fid_score = self.metrics.compute()
                print(f"Inception Score: {mean:.4f} ± {std:.4f}")
                print(f"FID Score: {fid_score:.4f}")
                self.tracker.track("is_mean", mean, epoch)
                self.tracker.track("is_std", std, epoch)
                self.tracker.track("fid_score", fid_score, epoch)

                preds = torch.cat(real_preds + fake_preds)
                targets = torch.cat(real_targets + fake_targets)

                accuracy = (preds == targets).float().mean().item()
                print(f"Discriminator accuracy on test set: {accuracy:.4f}")
                self.tracker.track("accuracy", accuracy, epoch)

        self.tracker.export_data()
        print(self.tracker.get_metrics())
