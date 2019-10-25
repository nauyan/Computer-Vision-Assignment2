from skimage import feature
from skimage import io, color
import sklearn
import glob
import numpy as np

train_positive = []
for x in glob.glob("./Dataset/Train/pos/*"):
    img = io.imread(x)
    img = color.rgb2gray(img)
    temp = np.zeros(2,)
    temp[0] = 66
    train_positive.append(temp)
    # print(img.shape)
arr = np.asarray(train_positive)
print(arr.shape)
