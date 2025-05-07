# 🚗 Car Paint Defect Detection using Segment-anything meta AI + Delta E (CIE76)

This project detects paint defects in car surfaces using **Meta's Segment Anything Model (SAM)** for car body segmentation and **Delta E (CIE76)** for detecting color inconsistencies. It extracts the dominant car color and identifies defect zones where the paint significantly deviates.

---

## 📌 Key Features

- ✅ Automatic car body segmentation using **SAM**
- 🎯 Dominant color extraction using **KMeans**
- 🌈 Color difference calculation using **∆E (Delta E - CIE76)** in Lab color space
- 🟩 Defects are highlighted where Delta E exceeds 2 (∆E > 2)

---

## 🧠 Tech Stack

- Python
- OpenCV
- NumPy
- scikit-image
- scikit-learn
- matplotlib
- PyTorch
- Meta AI's **Segment Anything Model (SAM)**

---

## 🛠 How TO run :
1)just run this on google colab
2) then upload image 
3) you will get results






