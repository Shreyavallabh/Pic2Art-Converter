import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load and resize image
img = cv2.imread("1.jpg")
img = cv2.resize(img, (600, 600))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Pencil Sketch
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
inv = 255 - gray
blur = cv2.GaussianBlur(inv, (21, 21), 0)
sketch = cv2.divide(gray, 255 - blur, scale=256.0)

# Step 1: Apply bilateral filter for edge-preserving smoothing
smooth = cv2.bilateralFilter(img, d=9, sigmaColor=300, sigmaSpace=300)

# Step 2: Detect edges
gray_blur = cv2.medianBlur(gray, 7)
edges = cv2.adaptiveThreshold(gray_blur, 255,
                              cv2.ADAPTIVE_THRESH_MEAN_C,
                              cv2.THRESH_BINARY, 9, 2)

# Step 3: Color quantization using k-means clustering (reduces colors like cartoons)
Z = img.reshape((-1, 3))
Z = np.float32(Z)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 9  # Number of color clusters
_, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
quantized = res.reshape((img.shape))

# Step 4: Combine quantized colors with edges
cartoon = cv2.bitwise_and(quantized, quantized, mask=edges)
cartoon_rgb = cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB)

# Plot all outputs
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(img_rgb)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Pencil Sketch")
plt.imshow(sketch, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Cartoon")
plt.imshow(cartoon_rgb)
plt.axis('off')

plt.tight_layout()
plt.show()