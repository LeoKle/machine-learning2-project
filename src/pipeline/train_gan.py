import torch
import torch.nn as nn

from utils.device import DEVICE
from utils.plotter import Plotter


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
    ):
        self.dataloader_train = dataloader_train
        self.dataloader_test = dataloader_test

        self.generator = generator.to(DEVICE)
        self.discriminator = discriminator.to(DEVICE)

        self.loss_function_discriminator = loss_function_discriminator
        self.loss_function_generator = loss_function_generator
        self.optimizer_discriminator = optimizer_discriminator
        self.optimizer_generator = optimizer_generator

    def train_discriminator(self, batch: torch.Tensor):
        self.discriminator.zero_grad()
        batch_size = batch.shape[0]

        batch = batch.to(DEVICE)

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

        latent_tensor = torch.randn(batch_size, self.generator.latent_dim).to(DEVICE)
        label_real = torch.ones(batch_size, 1).to(DEVICE)

        generator_output = self.generator(latent_tensor)
        discriminator_output = self.discriminator(generator_output)

        loss = self.loss_function_generator(discriminator_output, label_real)
        self.generator_loss.append(loss)

        loss.backward()
        self.optimizer_generator.step()

    def train(self, epoch):
        for epoch in range(1, epoch + 1):
            print("Training epoch", epoch)
            self.generator.train()
            self.discriminator.train()

            self.generator_loss = []
            self.discriminator_loss = []

            for _, (x, _) in enumerate(self.dataloader_train):
                self.train_generator(x)
                self.train_discriminator(x)

            generator_loss = torch.tensor(self.generator_loss).mean()
            discriminator_loss = torch.tensor(self.discriminator_loss).mean()

            print(f"Generator Loss: {generator_loss}")
            print(f"Discriminator Loss: {discriminator_loss}")

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

                preds = torch.cat(real_preds + fake_preds)
                targets = torch.cat(real_targets + fake_targets)

                accuracy = (preds == targets).float().mean().item()
                print(f"Discriminator accuracy on test set: {accuracy:.4f}")

                image_count = 1
                latent_tensor = torch.randn(image_count, self.generator.latent_dim).to(
                    DEVICE
                )

                generator_output = self.generator(latent_tensor)

                Plotter.show_image(
                    generator_output[0], output_file_name=f"output/gan_{epoch}.png"
                )
                Plotter.show_image(
                    generator_output[0], output_file_name="output/#latest.png"
                )
