import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from scipy.ndimage import gaussian_filter

transform = transforms.Compose([transforms.ToTensor()])
mnist = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

clean_digit = None
for img, label in mnist:
    np_img = img.squeeze().numpy()
    if label == 8 and np_img.max() > 0.95:
        clean_digit = np_img
        break

if clean_digit is None:
    raise ValueError("Could not find a clear MNIST digit.")

# --- Create: High IS but Bad FID ---
# Sharpening: makes the image confident
sharpened = np.clip(clean_digit**2.5, 0, 1)

# Add salt & pepper noise: breaks realism (bad FID)
noise_mask = np.random.rand(*sharpened.shape)
high_is_bad_fid = sharpened.copy()
high_is_bad_fid[noise_mask < 0.1] = 1.0
high_is_bad_fid[noise_mask > 0.9] = 0.0

# --- Create: Low IS but Good FID ---
# Blurring: reduces sharpness/confidence (low IS)
low_is_good_fid = gaussian_filter(clean_digit, sigma=1.5)

fig, axs = plt.subplots(1, 3, figsize=(9, 3))

axs[0].imshow(clean_digit, cmap="gray", vmin=0, vmax=1)
axs[0].set_title("Original (Good IS & FID)")
axs[0].axis("off")

axs[1].imshow(high_is_bad_fid, cmap="gray", vmin=0, vmax=1)
axs[1].set_title("High IS, Bad FID")
axs[1].axis("off")

axs[2].imshow(low_is_good_fid, cmap="gray", vmin=0, vmax=1)
axs[2].set_title("Good FID, Low IS")
axs[2].axis("off")

plt.tight_layout()
plt.savefig("is_vs_fid_plot.svg")
plt.show()
