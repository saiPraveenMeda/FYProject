
import matplotlib.pyplot as plt

from skimage import data
from skimage.filters import threshold_otsu, threshold_adaptive

warped = threshold_adaptive("warped.jpg", 250, offset = 10)
warped = warped.astype("uint8") * 255

# get areas where we can split image on whitespace to make OCR more accurate
color_level = np.array([np.sum(line) / len(line) for line in warped])
cuts = []
i = 0
while(i < len(color_level)):
    if color_level[i] > 250:
        begin = i
        while(color_level[i] > 250):
            i += 1
        cuts.append((i + begin)/2) # middle of the whitespace region
    else:
        i += 1
