import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from sklearn.cluster import KMeans
from tkinter import filedialog
from tkinter import Tk

# 📁 File dialog to choose image
Tk().withdraw()
file_path = filedialog.askopenfilename(title="Select a Car Image")
if not file_path:
    print("❌ No file selected. Exiting.")
    exit()

# 📷 Read and convert image
bgr = cv2.imread(file_path)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
rgb = cv2.resize(rgb, (800, 600))  # Resize for better viewing

# 🎯 Use KMeans to find dominant color
pixels = rgb.reshape(-1, 3)
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(pixels)
dominant_color = kmeans.cluster_centers_[0].astype(np.uint8)

print(f"🎯 Automatically detected dominant color: {dominant_color}")

# 🎨 Convert reference to LAB
ref_rgb = dominant_color.reshape(1, 1, 3)
ref_lab = color.rgb2lab(ref_rgb)

# 🎨 Convert full image to LAB
lab_img = color.rgb2lab(rgb)

# 📏 Compute Delta E
delta_e = color.deltaE_cie76(lab_img, ref_lab)

# 🧪 Mark pixels with ∆E > 2 as defects
defect_mask = delta_e > 2
highlighted_img = rgb.copy()
highlighted_img[defect_mask] = [0, 255, 0]  # Green for defect zones

# 📊 Show results
plt.figure(figsize=(15, 6))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(rgb)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Delta E Map")
plt.imshow(delta_e, cmap='magma')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Defects Highlighted")
plt.imshow(highlighted_img)
plt.axis('off')

plt.tight_layout()
plt.show()
