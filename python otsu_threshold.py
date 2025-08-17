import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt

def otsu_threshold(image_uint8):
    """
    Compute Otsu's threshold for a uint8 grayscale image and return (threshold, binary_mask).
    """
    # Histogram (256 bins for uint8)
    hist, bin_edges = np.histogram(image_uint8.ravel(), bins=256, range=(0, 256))
    total = image_uint8.size

    # Probabilities of each intensity level
    p = hist.astype(np.float64) / total

    # Cumulative sums (class probabilities) and cumulative means
    w0 = np.cumsum(p)                   # class 0..k
    w1 = 1.0 - w0                       # class k+1..255
    mu_k = np.cumsum(p * np.arange(256))  # cumulative mean up to k
    mu_T = mu_k[-1]                       # global mean

    # Between-class variance σ_b^2(k) = (μ_T*w0 - μ_k)^2 / (w0*(1-w0))
    eps = 1e-12
    numerator = (mu_T * w0 - mu_k) ** 2
    denom = w0 * w1
    sigma_b2 = numerator / (denom + eps)

    # Maximize between-class variance
    k_star = int(np.argmax(sigma_b2))  # threshold in [0..255]

    # Binarize (>= matches common MATLAB behavior)
    bw = (image_uint8 >= k_star).astype(np.uint8)

    return k_star, bw

if __name__ == "__main__":
    # ---- Load image (grayscale) ----
    I = imageio.imread("polymersomes.tif")
    if I.ndim == 3:  # convert RGB to grayscale if needed
        I = (0.2989*I[...,0] + 0.5870*I[...,1] + 0.1140*I[...,2]).astype(np.uint8)
    else:
        # ensure uint8 range
        if I.dtype != np.uint8:
            # scale safely to 0..255 if it's not uint8
            I = (255 * (I.astype(np.float64) - I.min()) / (I.max() - I.min() + 1e-12)).astype(np.uint8)

    T, BW = otsu_threshold(I)
    print(f"Otsu threshold: {T}")

    # ---- Show result ----
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.imshow(I, cmap="gray")
    plt.title("Input")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(BW, cmap="gray")
    plt.title(f"Binarized (T={T})")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
