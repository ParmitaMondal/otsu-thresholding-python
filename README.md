# Otsu's Thresholding in Python

This repository contains a **from-scratch implementation of Otsu’s thresholding** algorithm in Python using NumPy.  
It replicates the classic MATLAB code flow: histogram → cumulative sums → class means → maximize between-class variance → compute threshold → binarize.

## 📂 Features
- Loads an input grayscale image (e.g., `polymersomes.tif`)
- Computes histogram and probabilities
- Iteratively calculates class statistics
- Finds optimal threshold by maximizing between-class variance
- Produces a binary mask similar to MATLAB’s `graythresh` + `im2bw`
- Displays both the input and thresholded images

## 🚀 Usage
Clone the repo and install dependencies:

```bash
git clone https://github.com/<your-username>/otsu-thresholding-python.git
cd otsu-thresholding-python
pip install -r requirements.txt
