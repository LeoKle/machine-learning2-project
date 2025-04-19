import torch
import torch.nn as nn

from utils.device import DEVICE


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
        label_real = torch.ones(batch_size, 1, device=DEVICE)

        output_real = self.discriminator(batch)
        loss_real = self.loss_function_discriminator(output_real, label_real)

        # train discriminator on generated data
        label_fake = torch.zeros(batch_size, 1, device=DEVICE)
        latent_tensor = torch.randn(
            batch_size, self.generator.latent_dim, device=DEVICE
        )
        generated_images = self.generator(latent_tensor)

        output_generated = self.discriminator(generated_images)
        loss_generated = self.loss_function_discriminator(output_generated, label_fake)

        total_loss = loss_real + loss_generated
        total_loss.backward()
        self.optimizer_discriminator.step()

    def train_generator(self, batch: torch.Tensor):
        self.discriminator.zero_grad()
        batch_size = batch.shape[0]

        latent_tensor = torch.randn(batch_size, self.generator.latent_dim).to(DEVICE)
        label_real = torch.ones(batch_size, 1).to(DEVICE)

        generator_output = self.generator(latent_tensor)
        discriminator_output = self.discriminator(generator_output)

        loss = self.loss_function_generator(discriminator_output, label_real)

        loss.backward()
        self.optimizer_generator.step()

    def train(self, epoch):
        for epoch in range(1, epoch + 1):
            self.generator.train()
            self.discriminator.train()

            for batch_idx, (x, _) in enumerate(self.dataloader_train):
                self.train_generator(x)
                self.train_discriminator(x)

            with torch.no_grad():
                self.generator.eval()
                self.discriminator.eval()
