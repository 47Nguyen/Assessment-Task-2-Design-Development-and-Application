#Load modules
import numpy as np
from skimage.feature import hog
from skimage.color import rgb2gray, rgb2hsv


#Shape/edge descriptor - one HOG vector per image
def hog_features(X):
    vectors = []
    for image in X:
        grey = rgb2gray(image)
        vectors.append(hog(grey, pixels_per_cell=(8, 8), cells_per_block=(2, 2)))
    return np.array(vectors, dtype=np.float32)


#Colour descriptor - 16-bin histogram per HSV channel, concatenated
def colour_hist(X, bins=16):
    vectors = []
    for image in X:
        hsv = rgb2hsv(image)
        counts = []
        for c in range(3):
            hist, _ = np.histogram(hsv[:, :, c], bins=bins, range=(0, 1))
            counts.append(hist / hist.sum())
        vectors.append(np.concatenate(counts))
    return np.array(vectors, dtype=np.float32)
