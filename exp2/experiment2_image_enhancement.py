"""
Experiment 2: Image Enhancement Based on OpenCV
=================================================
Course     : Digital Image Processing (Shanghai Maritime University)
Objective  : Master the basic usage of OpenCV, and use computer-vision
             libraries to perform smoothing / filtering operations on
             images so as to achieve image enhancement.

Pipeline   :
    1. Import the required packages.
    2. Read the original image and convert between colour spaces.
    3. Add Salt-and-Pepper noise and Gaussian noise.
    4. Suppress noise with Mean / Median / Gaussian filters.
    5. Manually implement a per-channel median filter for colour images.
    6. Save every result into ./output_images and ./result.

Dependencies: opencv-python, scikit-image, numpy, matplotlib
Run        : python experiment2_image_enhancement.py
"""

import os
import matplotlib
matplotlib.use('Agg')          # non-interactive backend (no GUI required)
import cv2
import numpy as np
from skimage.util import random_noise
from matplotlib import pyplot as plt

# Make sure the output folders exist
os.makedirs('output_images', exist_ok=True)
os.makedirs('result', exist_ok=True)


# =====================================================================
# 2.1  Read the original image and perform colour-space conversion
# =====================================================================

# ---- Read the local image (BGR layout inside OpenCV) ----
img = cv2.imread('p1.jpg')
# Resize to a moderate size, preserving the aspect ratio, so that the
# manual (Python-loop) median filter stays computationally tractable.
h, w = img.shape[:2]
scale = 512 / max(h, w)
img = cv2.resize(img, (int(w * scale), int(h * scale)),
                 interpolation=cv2.INTER_AREA)

# Get the (B, G, R) values of the pixel located at (row=100, col=100)
(b, g, r) = img[100, 100]
print('Pixel (100, 100) -> BGR =', b, g, r)

# ---- Show the original image (OpenCV reads it as BGR) ----
plt.imshow(img)
plt.title('Original Image (BGR)')
plt.axis('off')
plt.savefig('output_images/original_bgr.jpg', dpi=200, bbox_inches='tight')
plt.close()

# ---- BGR -> RGB: swap the Blue and Red channels ----
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(rgb_img)
plt.title('RGB Image (BGR->RGB)')
plt.axis('off')
plt.savefig('output_images/rgb_image.jpg', dpi=200, bbox_inches='tight')
plt.close()

# ---- BGR -> Gray: collapse the three channels into one ----
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_img, cmap='gray')
plt.title('Gray Image (BGR->GRAY)')
plt.axis('off')
plt.savefig('output_images/gray_image.jpg', dpi=200, bbox_inches='tight')
plt.close()


# =====================================================================
# 2.2  Add noise to the image
# =====================================================================

# Salt-and-Pepper noise: ~40% of the pixels are randomly turned into
# white (salt) or black (pepper) values.
sp_noise_img = random_noise(rgb_img, mode='s&p', amount=0.4)

# Gaussian noise: a random deviation drawn from a Gaussian distribution
# (mean = 0.2, variance = 0.03) is added to every pixel.
gus_noise_img = random_noise(rgb_img, mode='gaussian', mean=0.2, var=0.03)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(rgb_img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(sp_noise_img, cmap='gray')
plt.title('Salt-and-Pepper Noise')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(gus_noise_img, cmap='gray')
plt.title('Gaussian Noise')
plt.axis('off')

plt.tight_layout()
plt.savefig('output_images/noise_comparison.jpg', dpi=200, bbox_inches='tight')
plt.close()


# =====================================================================
# 2.3  Image filtering with the three OpenCV filters
# =====================================================================

# ---- Mean filter (cv2.blur) on both noisy images ----
mean_sp = cv2.blur(sp_noise_img, (5, 5))
mean_gus = cv2.blur(gus_noise_img, (5, 5))

# ---- Median filter (cv2.medianBlur) - requires uint8 input ----
mid_sp = cv2.medianBlur((sp_noise_img * 255).astype(np.uint8), 5)
mid_gus = cv2.medianBlur((gus_noise_img * 255).astype(np.uint8), 5)

# ---- Gaussian filter (cv2.GaussianBlur) ----
gauss_sp = cv2.GaussianBlur((sp_noise_img * 255).astype(np.uint8), (5, 5), 0)
gauss_gus = cv2.GaussianBlur((gus_noise_img * 255).astype(np.uint8), (5, 5), 0)

# ---- 2 x 3 comparison grid ----
plt.figure(figsize=(13, 8))
# Row 1: Salt-and-Pepper noise
plt.subplot(2, 3, 1)
plt.imshow(mean_sp)
plt.title('S&P noise + Mean Filter')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(mid_sp)
plt.title('S&P noise + Median Filter')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(gauss_sp)
plt.title('S&P noise + Gaussian Filter')
plt.axis('off')

# Row 2: Gaussian noise
plt.subplot(2, 3, 4)
plt.imshow(mean_gus)
plt.title('Gaussian noise + Mean Filter')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(mid_gus)
plt.title('Gaussian noise + Median Filter')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(gauss_gus)
plt.title('Gaussian noise + Gaussian Filter')
plt.axis('off')

plt.tight_layout()
plt.savefig('result/filter_results_2x3.jpg', dpi=200, bbox_inches='tight')
plt.close()


# =====================================================================
# 2.4  Manually implement a median filter for colour images
# =====================================================================

def manual_median_filter_color(image, kernel_size=5):
    """
    Manually implement a median filter for a colour (RGB) image.

    Parameters
    ----------
    image       : input colour image, numpy array of shape (H, W, 3).
    kernel_size : size of the (square) filtering window.

    Returns
    -------
    filtered_img : the median-filtered colour image.
    """
    pad = kernel_size // 2                       # half window, e.g. 2 for 5
    filtered_img = np.zeros_like(image)

    # Process each colour channel independently
    for c in range(3):                           # R, G, B channels
        channel = image[:, :, c]
        padded_channel = np.pad(channel, pad_width=pad, mode='edge')
        for i in range(channel.shape[0]):
            for j in range(channel.shape[1]):
                # Extract the kxk neighbourhood window
                region = padded_channel[i:i + kernel_size, j:j + kernel_size]
                filtered_img[i, j, c] = np.median(region)

    return filtered_img


# Apply the manual median filter to the salt-and-pepper noisy image
manual_mid = manual_median_filter_color(
    (sp_noise_img * 255).astype(np.uint8), kernel_size=5
)

# ---- Side-by-side comparison ----
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(sp_noise_img, cmap='gray')
plt.title('Salt-and-Pepper Noisy Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(manual_mid, cmap='gray')
plt.title('Manual Median Filter (colour)')
plt.axis('off')

plt.tight_layout()
plt.savefig('result/manual_median.jpg', dpi=200, bbox_inches='tight')
plt.close()

print('All result images have been saved to ./output_images and ./result')
