import matplotlib.pyplot as plt
import numpy as np

epochs = np.arange(1, 101)

discriminator_loss = np.exp(-epochs / 40) + 0.01 * np.random.randn(len(epochs))

generator_loss = 1.5 - np.exp(-epochs / 60) + 0.01 * np.random.randn(len(epochs))

discriminator_loss = np.clip(discriminator_loss, 0, None)
generator_loss = np.clip(generator_loss, 0, None)

plt.figure(figsize=(10, 6))
plt.plot(
    epochs, discriminator_loss, label="Discriminator Loss", color="blue", linewidth=2
)
plt.plot(epochs, generator_loss, label="Generator Loss", color="red", linewidth=2)
plt.xlabel("Epochs")
plt.ylabel("Loss")
# plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("gan_training_loss.svg")
plt.show()
