"""
preprocessing.py
-----------------
This file acts as a library for baseline_logR.py, baseline_gda.py,
and baseline_naive_bayes.py to do its pre-processing on a given image.
"""
import cv2

# converts image into vector representing a color histogram of the image.
def extract_color_histogram(image, bins=(8, 8, 8)):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# converts a batch of images from (4032 by 3024 by 3) arrays to
# (4032 * 3024 * 3) length vectors.
def flatten(images):
    return images.view(images.size(0), -1)